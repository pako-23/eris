"""
This script defines the ERIS client node used in decentralized federated learning experiments.
Each client performs local model training with optional differential privacy (DP), pruning, or sparsification,
and evaluates privacy leakage through membership inference attacks (MIAs) depending on the experiment configuration.

Clients communicate with aggregators and support multiple privacy-preserving mechanisms, 
gradient compression techniques (e.g., shifted-k sparsification), and audit tools.
Designed to be launched alongside coordinator.py for full experiment orchestration.

This is code is set to be used locally, but it can be used in a distributed environment by changing the IP address of 
each client.
"""

#!/usr/bin/env python3

import numpy as np
from eris import ErisClient, ShiftedCompression
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Subset
from torch.utils.data import DataLoader
import sys
import os
import argparse
import time
import copy
import opacus # type: ignore

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import models
from public import utils
from public import config as cfg


class ExampleClient(ErisClient):
    def __init__(
        self,
        router_address: str,
        subscribe_address: str,
        model=None,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        criterion=None,
        num_examples=None,
        client_id=None,
        train_fn=None,
        evaluate_fn=None,
        device=None,
        config=None,
        exp_n: int = 0,
    ):
        # Initialize the superclass with only the required positional arguments
        super().__init__(router_address, subscribe_address)

        # Initialize additional attributes
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_examples = num_examples
        self.client_id = client_id
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.device = device
        self.dataset_name = config["dataset"]
        self.predictor_name = config["model_name"] if model else None
        self.config = config
        self.current_round = 0
        
        # privacy audit parameters
        self.split_mask = None
        self.n_split = None
        self.k_plus = cfg.k_plus
        self.k_min = cfg.k_min
        self.p_value = cfg.p_value
        self.canary_frac = cfg.canary_frac
        self.privacy_audit = cfg.privacy_audit
        self.privacy_estimate = -1
        self.accuracy_mia = -1
        self.acc_privacy_estimate = -1
        self.acc_accuracy_mia = -1
        self.privacy_estimate_mean = -1
        self.accuracy_mia_mean = -1
        self.acc_privacy_estimate_mean = -1
        self.acc_accuracy_mia_mean = -1
        self.acc_scores = None
        self.local_dp = cfg.local_dp
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.exp_n = exp_n
        self.reference_s = []
        for p in self.get_parameters():
            self.reference_s.append(np.zeros_like(p))

        # prepare dataset auditing
        if self.privacy_audit:
            canaries, non_canaries = random_split(self.train_loader.dataset, [self.canary_frac, 1 - self.canary_frac])
            self.n_canaries = len(canaries)
            self.scores = np.zeros(self.n_canaries)

            # subsample canaries & make new dataloader
            true_in_out = torch.distributions.bernoulli.Bernoulli(torch.ones(self.n_canaries) * 0.5).sample()
            self.true_in_out = true_in_out.numpy()
            canaries_in_idx = torch.nonzero(true_in_out)
            subsampled_train_data = torch.utils.data.ConcatDataset([
                non_canaries,
                torch.utils.data.Subset(canaries, canaries_in_idx)
            ])
            self.subsampled_train_loader = DataLoader(subsampled_train_data, batch_size=self.config['batch'], shuffle=True)
            self.canary_loader = DataLoader(canaries, batch_size=self.config['batch'], shuffle=False)

        # local differential privacy initialization
        if cfg.local_dp:
            # Calculate sample rate = (batch_size / total_number_of_samples)
            if cfg.privacy_audit:
                sample_rate = min(1.0, self.subsampled_train_loader.batch_size / len(self.subsampled_train_loader.dataset))
            else:
                sample_rate = min(1.0, self.train_loader.batch_size / len(self.train_loader.dataset))

            self.sigma = opacus.accountants.utils.get_noise_multiplier(
                target_epsilon=cfg.epsilon,
                target_delta=cfg.delta,
                sample_rate=sample_rate,
                epochs=int(self.config['epochs']), 
                accountant='rdp',  
            ) 

            self.privacy_engine = opacus.privacy_engine.PrivacyEngine(accountant='rdp', secure_mode=False)
            if cfg.privacy_audit:
                self.model, self.optimizer, self.subsampled_train_loader = self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.subsampled_train_loader,
                    noise_multiplier=self.sigma,
                    max_grad_norm=cfg.sensitivity,
                    )
            else:
                self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=self.sigma,
                    max_grad_norm=cfg.sensitivity,
                    )           
            
            if client_id == 1:
                print(f"\n\033[94mLocal Differential Privacy with introduced noise_value_sd: {self.sigma}\033[0m\n")

    @property
    def gamma(self):
        n = 2
        self.k = int(self.n_params / (n * np.log2(self.config['rounds'][self.exp_n])))
        # self.k = k = int(self.n_params * cfg.k_sparsity)
        w = (self.n_params / self.k) - 1
        return np.sqrt((1 + 2 * w) / (2 * (1 + w)**3))

    def get_parameters(self):
        self.model.to("cpu")
        return [param.data.numpy() for param in self.model.parameters()]


    def set_parameters(self, parameters):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(parameters[i])


    def fit(self):    
        # get initial parameters
        params_in = self.get_parameters()
        self.model.train(True)
        self.model.to(self.device)
        self.current_round += 1
        
        # privacy auditing
        if self.privacy_audit:
            if cfg.local_dp:
                """
                Local Differential Privacy (DP) training
                """
                models.train_with_opacus(
                    self.model, 
                    self.device, 
                    self.subsampled_train_loader, 
                    self.optimizer, 
                    self.criterion, 
                    self.sigma, 
                    self.config["epochs"], 
                    self.client_id
                    )
            else:
                """
                Traditional training without DP
                """
                for epoch in range(self.config["epochs"]):
                    self.train_fn(
                        self.model, 
                        self.device, 
                        self.subsampled_train_loader, 
                        self.optimizer, 
                        self.criterion, 
                        epoch, 
                        self.client_id
                        )
                
            if cfg.pruning:
                """
                prune k% of the largest gradients [PriPrune FL] 
                """
                # calculate gradients
                params_out = self.get_parameters()
                grads = [param_out - param_in for param_in, param_out in zip(params_in, params_out)]
                
                # prune the largest gradients
                pruned_grads = self.prune_largest_grads(
                    grads=grads,
                    pruning_rate = cfg.pruning_rate
                    )
                
                # update model parameters
                params_out = [param_in + grad for param_in, grad in zip(params_in, pruned_grads)]
            
                # update model weights
                self.set_parameters(params_out)
                
            elif cfg.k_sparsification:
                """
                k-random sparsification on the gradients [Part of SOTERIAFL]
                """
                # calculate gradients
                params_out = self.get_parameters()
                grads = [param_out - param_in for param_in, param_out in zip(params_in, params_out)]
                    
                # k-sparsification on the gradients
                sparse_grads = self.compress_parameters(
                    grads,
                    k = self.k
                    )
                
                # update model parameters
                params_out = [param_in + grad for param_in, grad in zip(params_in, sparse_grads)]
                
                # update model weights
                self.set_parameters(params_out)
            
            elif cfg.shifted_k_sparsification:
                """
                shifted k-random sparsification on the gradients [SOTERIAFL]
                """
                # calculate gradients
                params_out = self.get_parameters()
                grads = [param_out - param_in for param_in, param_out in zip(params_in, params_out)]
                
                # shifted gradient
                shifted_grads = [grad - s for grad, s in zip(grads, self.reference_s)] 
                
                # k-sparsification on the gradients
                shifted_sparse_grads = self.compress_parameters(
                    shifted_grads,
                    k = self.k
                    )
                
                # update reference vector
                self.reference_s = [s + self.gamma * sparse_grad for s, sparse_grad in zip(self.reference_s, shifted_sparse_grads)]
                
                # update model parameters
                params_out = [param_in + grad for param_in, grad in zip(params_in, shifted_sparse_grads)]
                
                # update model weights
                self.set_parameters(params_out)
            
            else:
                params_out = self.get_parameters()

            # evaluation of privacy leakage per split 
            accuracy_mia_list, acc_accuracy_mia_list = [], [] 
            privacy_estimate_list, acc_privacy_estimate_list = [], [] 
            for i in range(self.n_split):
                params_out_only = copy.deepcopy(params_in)
                for j in range(len(params_out_only)):
                    mask = (self.split_mask[j] == i)
                    params_out_only[j][mask] = params_out[j][mask]
                # params_out_only = copy.deepcopy(params_out) 

                # set params to the model
                self.set_parameters(params_out_only)
                    
                # normalize client update vector
                client_update = utils.parameters_to_1d(params_out_only) - utils.parameters_to_1d(params_in)
                client_update = client_update / np.linalg.norm(client_update)

                # compute scores for each canary, used to predict membership            
                scores = []
                # canary_loader = torch.utils.data.DataLoader(canaries, batch_size=cfg.batch_size, shuffle=False)
                if cfg.score_fn == 'whitebox':
                    self.set_parameters(params_in)
                    for samples, targets in self.canary_loader:
                        scores.extend(self.score_with_pseudograd_batch(samples, targets, client_update))
                    self.set_parameters(params_out_only)
                if cfg.score_fn == 'blackbox':
                    for samples, targets in self.canary_loader:
                        scores.extend(self.score_blackbox_batch(samples, targets))
                else:
                    NotImplementedError(f'score function {cfg.score_fn} is not known')

                # accumulative leakage
                if self.acc_scores is None:
                    self.acc_scores = copy.deepcopy(scores)
                else:
                    self.acc_scores = self.acc_scores + np.asarray(scores)

                # lower-bound privacy budget evaluation
                accuracy_mia, privacy_estimate = self.evaluate_privacy(scores)
                acc_accuracy_mia, acc_privacy_estimate = self.evaluate_privacy(self.acc_scores)
                accuracy_mia_list.append(accuracy_mia)
                privacy_estimate_list.append(privacy_estimate)
                acc_accuracy_mia_list.append(acc_accuracy_mia)
                acc_privacy_estimate_list.append(acc_privacy_estimate)
            
            # max metrics
            self.accuracy_mia = max(accuracy_mia_list)
            self.privacy_estimate = max(privacy_estimate_list)
            self.acc_accuracy_mia = max(acc_accuracy_mia_list)
            self.acc_privacy_estimate = max(acc_privacy_estimate_list)
            
            # mean
            self.accuracy_mia_mean = np.mean(accuracy_mia_list)
            self.privacy_estimate_mean = np.mean(privacy_estimate_list)
            self.acc_accuracy_mia_mean = np.mean(acc_accuracy_mia_list)
            self.acc_privacy_estimate_mean = np.mean(acc_privacy_estimate_list)
            
            utils.save_audit_metrics(
                round_num=self.current_round, 
                accuracy=self.accuracy_mia, 
                privacy_estimate=self.privacy_estimate, 
                acc_accuracy=self.acc_accuracy_mia, 
                acc_privacy_estimate=self.acc_privacy_estimate,
                accuracy_mean=self.accuracy_mia_mean, 
                privacy_estimate_mean=self.privacy_estimate_mean, 
                acc_accuracy_mean=self.acc_accuracy_mia_mean, 
                acc_privacy_estimate_mean=self.acc_privacy_estimate_mean, 
                client_id=self.client_id,
                history_folder=f"histories/{self.config['model_name']}/{self.config['dataset']}/"
                )
            
            # reset client model 
            self.set_parameters(params_out)
 
        else: # NO AUDITING
            if cfg.local_dp:   
                # Local Differential Privacy
                models.train_with_opacus(self.model, 
                    self.device, 
                    self.train_loader, 
                    self.optimizer, 
                    self.criterion, 
                    self.sigma, 
                    self.config["epochs"], 
                    self.client_id
                    )
            else:
                for epoch in range(self.config["epochs"]):
                    self.train_fn(
                        self.model, 
                        self.device, 
                        self.train_loader, 
                        self.optimizer, 
                        self.criterion, 
                        epoch, 
                        self.client_id
                        )
                    
            if cfg.pruning:
                """
                prune k% of the largest gradients [PriPrune FL] 
                """
                # calculate gradients
                params_out = self.get_parameters()
                grads = [param_out - param_in for param_in, param_out in zip(params_in, params_out)]
                
                # prune the largest gradients
                pruned_grads = self.prune_largest_grads(
                    grads=grads,
                    pruning_rate = cfg.pruning_rate
                    )
                
                # update model parameters
                params_out = [param_in + grad for param_in, grad in zip(params_in, pruned_grads)]
            
                # update model weights
                self.set_parameters(params_out)

            elif cfg.k_sparsification:
                """
                k-random sparsification on the gradients [Part of SOTERIAFL]
                """
                # calculate gradients
                params_out = self.get_parameters()
                grads = [param_out - param_in for param_in, param_out in zip(params_in, params_out)]
                    
                # k-sparsification on the gradients
                sparse_grads = self.compress_parameters(
                    grads,
                    k = self.k
                    )
                
                # update model parameters
                params_out = [param_in + grad for param_in, grad in zip(params_in, sparse_grads)]
                
                # update model weights
                self.set_parameters(params_out)

            elif cfg.shifted_k_sparsification:
                """
                shifted k-random sparsification on the gradients [SOTERIAFL]
                """
                # calculate gradients
                params_out = self.get_parameters()
                grads = [param_out - param_in for param_in, param_out in zip(params_in, params_out)]
                
                # shifted gradient
                shifted_grads = [grad - s for grad, s in zip(grads, self.reference_s)] 
                
                # k-sparsification on the gradients
                shifted_sparse_grads = self.compress_parameters(
                    shifted_grads,
                    k = self.k
                    )
                
                # update reference vector
                self.reference_s = [s + self.gamma * sparse_grad for s, sparse_grad in zip(self.reference_s, shifted_sparse_grads)]
                
                # update model parameters
                params_out = [param_in + grad for param_in, grad in zip(params_in, shifted_sparse_grads)]
                
                # update model weights
                self.set_parameters(params_out)
                
            else:
                params_out = self.get_parameters()

        return params_out, self.num_examples["train"]

    def evaluate(self):
        # save previous aggregated model if client 1
        if self.client_id == 1:
            if cfg.local_dp:
                # Rename keys by removing the '_module.' prefix
                state_dict = {}
                for key, weight in self.model.state_dict().items():
                    if key.startswith('_module.'):
                        new_key = key.replace('_module.', '', 1)
                        state_dict[new_key] = weight
                    else:
                        state_dict[key] = weight
            else:
                state_dict = self.model.state_dict()
            # save
            torch.save(state_dict, f"checkpoints/{self.predictor_name}/{self.config["dataset"]}/model_{self.current_round}.pth")

        self.model.eval()
        self.model.to(self.device)

        loss, accuracy, f1_score = self.evaluate_fn(
            self.model, self.device, self.val_loader, self.criterion, self.client_id
        )
        
        # save loss and accuracy client
        utils.save_client_metrics(
            self.current_round,
            loss,
            accuracy,
            f1_score,
            client_id=self.client_id,
            history_folder=f"histories/{self.predictor_name}/{self.dataset_name}/",
        )


    def evaluate_privacy(self, scores):
        ground_truth = copy.deepcopy(self.true_in_out)
        score_indices_sorted = np.argsort(scores)[::-1]
        classified_in = score_indices_sorted[:int(self.n_canaries * self.k_plus + 1)]
        classified_out = score_indices_sorted[int(self.n_canaries * (1 - self.k_min) + 1):]
        abstained = np.setdiff1d(score_indices_sorted, np.concatenate((classified_in, classified_out)))
        classification = np.zeros(self.n_canaries)
        classification[classified_in] = 1
        classification[abstained] = 2
        ground_truth[abstained] = 2
        W = ground_truth == classification
        num_correct = W.sum() - len(abstained)
        accuracy_mia = num_correct / (self.n_canaries - len(abstained))
        
        # tpr = np.sum(classification == true_in_out) / len(canaries_in_idx)
        # tnr = np.sum((1 - classification) == (1 - true_in_out)) / len(canaries_out_idx)
        # fpr = np.sum(classification == (1 - true_in_out)) / len(canaries_out_idx)
        # fnr = np.sum((1 - classification) == true_in_out) / len(canaries_in_idx)

        # compute empirical privacy estimate, which should be < epsilon w/ high probability
        privacy_estimate = utils.get_eps_audit(
            m=self.n_canaries,
            r=self.n_canaries - len(abstained),
            v=num_correct,
            delta=cfg.delta,
            p=0.05)
        
        # Kairouz privacy estimate from https://proceedings.mlr.press/v37/kairouz15.html
        # privacy_estimate = np.max([np.log(1 - cfg.delta - fpr) - np.log(fnr), 
                            # np.log(1 - cfg.delta - fnr) - np.log(fpr)])
                            
        return accuracy_mia, privacy_estimate
    
    
    def score_with_pseudograd_batch(self, samples, targets, client_update):
        '''
        Computes membership inference attack scores for a batch by 
        computing the inner product between the 'pseudogradient'
        represented by client update and the true gradients
        for each sample in the batch.
        '''
        self.model.to(self.device)  # Ensure model is on the correct device
        samples = samples.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        predictions = self.model(samples)
        losses = torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
        
        scores = []
        for loss in losses:
            # Compute gradients for each sample
            audit_grad = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
            # audit_grad = parameters_to_1d(audit_grad)
            audit_grad = np.concatenate([x.cpu().flatten() for x in audit_grad])
            score = np.dot(client_update, - audit_grad)
            scores.append(score)
        
        return scores


    def score_blackbox_batch(self, samples, targets):
        with torch.no_grad():
            self.model.to(self.device)  # Ensure model is on the correct device
            samples = samples.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(samples)
            losses = torch.nn.functional.cross_entropy(predictions, targets, reduction='none').cpu()

            return -losses

 
    def prune_params_basedon_grads(self, grads, params, pruning_rate=0.3):
        """
        Prunes 'pruning_rate' fraction (e.g., 0.3) of the weights with the largest gradients.
        Args:
            grads (List[np.ndarray]): List of NumPy arrays representing gradients.
            params (List[np.ndarray]): List of NumPy arrays representing model parameters.
            pruning_rate (float): Fraction of weights to prune (e.g., 0.3 for 30%).

        Returns:
            List[np.ndarray]: Pruned parameters with the largest weights set to zero.
        """

        assert len(grads) == len(params), "Number of gradients and parameters must match."
        assert pruning_rate > 0 and pruning_rate < 1, "Pruning rate must be in (0, 1)."

        # Flatten all parameters (excluding scalars) into a single array
        flattened_params = np.concatenate([p.flatten() for p in params if p.ndim > 0])
        flattened_grads = np.concatenate([np.abs(g.flatten()) for g in grads if g.ndim > 0])

        # Determine the threshold for pruning
        threshold = np.percentile(flattened_grads, 100 * (1 - pruning_rate))

        # Create a mask for weights below the threshold
        mask = flattened_grads <= threshold  # Keep small or midrange gradients
        # print(f"Pruned parameters: {np.sum(mask)} out of {len(flattened_params)}, pruning rate {pruning_rate}.")

        # Apply the mask to set pruned weights to zero
        pruned_params = flattened_params * mask

        # Reconstruct the parameter shapes
        pruned_params_list = []
        start_idx = 0
        for p in params:
            if p.ndim == 0:
                # If it's a scalar, just keep it uncompressed
                pruned_params_list.append(p)
            else:
                flat_len = p.size
                sliced = pruned_params[start_idx:start_idx + flat_len]
                pruned_params_list.append(sliced.reshape(p.shape))
                start_idx += flat_len

        return pruned_params_list

    # def prune_parameters(self, params, pruning_rate=0.3):
    #     """
    #     Prunes the largest weights in the parameters based on the pruning rate.

    #     Args:
    #         params (List[np.ndarray]): List of NumPy arrays representing model parameters.
    #         pruning_rate (float): Fraction of weights to prune (e.g., 0.3 for 30%).

    #     Returns:
    #         List[np.ndarray]: Pruned parameters with the largest weights set to zero.
    #     """
    #     pruned_params = []
    #     for idx, param in enumerate(params):
    #         if param.ndim == 0:
    #             # Skip scalar parameters if any
    #             pruned_params.append(param)
    #             continue

    #         # Compute the absolute values and flatten the array
    #         abs_param = np.abs(param)
    #         flat_param = abs_param.flatten()

    #         # Determine the threshold for pruning
    #         threshold = np.percentile(flat_param, 100 * (1 - pruning_rate))

    #         # Create a mask for weights above the threshold
    #         mask = abs_param >= threshold

    #         # Apply the mask to set pruned weights to zero
    #         pruned_param = param * mask

    #         pruned_params.append(pruned_param)
    #         # print(f"Layer {idx}: Pruned {pruning_rate * 100}% of weights.")

    #     return pruned_params
    

    def compress_parameters(self, params, k):
        """
        Compresses the model parameters using random-k sparsification.

        Args:
            params (List[np.ndarray]): List of NumPy arrays representing model parameters.
            k (int): Number of coordinates to retain during compression.

        Returns:
            List[np.ndarray]: Compressed parameters where only k coordinates are retained
                            (and scaled by d/k according to the random-k operator).
        """
        # Flatten all parameters (excluding scalars) into a single array
        flattened_params = np.concatenate([p.flatten() for p in params if p.ndim > 0])
        d = flattened_params.size

        # If k >= d, no compression happens (just return original parameters)
        if k >= d:
            print(f"No compression applied: k ({k}) >= d ({d}).")
            return params
        assert k > 0, "k must be a positive integer."

        # Randomly select k indices out of d
        indices = np.random.choice(d, k, replace=False)

        # Create a boolean mask of size d with exactly k entries as True
        mask = np.zeros(d, dtype=bool)
        mask[indices] = True

        # Apply the mask and scale by d/k
        scaling_factor = d / k
        compressed_flattened = scaling_factor * flattened_params * mask
        # print(f"\033[93mCompressed parameters: kept {np.sum(mask)} out of {len(flattened_params)}, compression rate {k/d}.\033[0m")

        # Reconstruct the parameter shapes
        sparsified_params = []
        start_idx = 0
        for p in params:
            if p.ndim == 0:
                # If it's a scalar, just keep it uncompressed
                sparsified_params.append(p)
            else:
                flat_len = p.size
                sliced = compressed_flattened[start_idx:start_idx + flat_len]
                sparsified_params.append(sliced.reshape(p.shape))
                start_idx += flat_len

        return sparsified_params


    def prune_params_basedon_grads(self, grads, params, pruning_rate=0.3):
        """
        Prunes 'pruning_rate' fraction (e.g., 0.3) of the weights with the largest gradients.
        Args:
            grads (List[np.ndarray]): List of NumPy arrays representing gradients.
            params (List[np.ndarray]): List of NumPy arrays representing model parameters.
            pruning_rate (float): Fraction of weights to prune (e.g., 0.3 for 30%).

        Returns:
            List[np.ndarray]: Pruned parameters with the largest weights set to zero.
        """
        
        assert len(grads) == len(params), "Number of gradients and parameters must match."
        assert pruning_rate > 0 and pruning_rate < 1, "Pruning rate must be in (0, 1)."
        
        # Flatten all parameters (excluding scalars) into a single array
        flattened_params = np.concatenate([p.flatten() for p in params if p.ndim > 0])
        flattened_grads = np.concatenate([np.abs(g.flatten()) for g in grads if g.ndim > 0])

        # Determine the threshold for pruning
        threshold = np.percentile(flattened_grads, 100 * (1 - pruning_rate))

        # Create a mask for weights below the threshold
        mask = flattened_grads <= threshold  # Keep small or midrange gradients
        # print(f"Pruned parameters: {np.sum(mask)} out of {len(flattened_params)}, pruning rate {pruning_rate}.")

        # Apply the mask to set pruned weights to zero
        pruned_params = flattened_params * mask

        # Reconstruct the parameter shapes
        pruned_params_list = []
        start_idx = 0
        for p in params:
            if p.ndim == 0:
                # If it's a scalar, just keep it uncompressed
                pruned_params_list.append(p)
            else:
                flat_len = p.size
                sliced = pruned_params[start_idx:start_idx + flat_len]
                pruned_params_list.append(sliced.reshape(p.shape))
                start_idx += flat_len

        return pruned_params_list


    def prune_largest_grads(self, grads, pruning_rate=0.3):
        """
        Prunes 'pruning_rate' fraction (e.g., 0.3) of the largest gradients.
        Args:
            grads (List[np.ndarray]): List of NumPy arrays representing gradients.
            pruning_rate (float): Fraction of weights to prune (e.g., 0.3 for 30%).

        Returns:
            List[np.ndarray]: Pruned parameters with the largest weights set to zero.
        """
        
        assert pruning_rate > 0 and pruning_rate < 1, "Pruning rate must be in (0, 1)."
        
        # Flatten all parameters (excluding scalars) into a single array
        flattened_grads = np.concatenate([g.flatten() for g in grads if g.ndim > 0])
        flattened_abs_grads = np.concatenate([np.abs(g.flatten()) for g in grads if g.ndim > 0])

        # Determine the threshold for pruning
        threshold = np.percentile(flattened_abs_grads, 100 * (1 - pruning_rate))

        # Create a mask for weights below the threshold
        mask = flattened_abs_grads <= threshold  # Keep small or midrange gradients
        # print(f"Pruned parameters: {np.sum(mask)} out of {len(flattened_params)}, pruning rate {pruning_rate}.")

        # Apply the mask to set pruned weights to zero
        pruned_grads = flattened_grads * mask

        # Reconstruct the parameter shapes
        pruned_grads_list = []
        start_idx = 0
        for g in grads:
            if g.ndim == 0:
                # If it's a scalar, just keep it uncompressed
                pruned_grads_list.append(g)
            else:
                flat_len = g.size
                sliced = pruned_grads[start_idx:start_idx + flat_len]
                pruned_grads_list.append(sliced.reshape(g.shape))
                start_idx += flat_len

        return pruned_grads_list













def parse_args():
    parser = argparse.ArgumentParser(
        description="Start an Eris client configured with the options for the given experiment"
    )
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 101),
        required=False,
        default=1,
        help="Specifies the artificial data partition",
    )
    parser.add_argument(
        "--aggregator",
        action=argparse.BooleanOptionalAction,
        help="Start node as aggregator",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="mnist",
        choices=["mnist", "cifar10", "imdb", "fmnist"],
    )
    parser.add_argument(
        "--shard",
        type=str,
        help="Path to the dataset portion",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 20),
        default=1,
        help="Specifies the fold to be used",
    )
    parser.add_argument(
        "--exp_n",
        type=int,
        help="exp number",
        default=0,
    )
    
    return parser.parse_args()


def start_node(
    is_aggr=False,
    model=None,
    train_loader=None,
    val_loader=None,
    optimizer=None,
    criterion=None,
    num_examples=None,
    client_id=None,
    train_fn=None,
    evaluate_fn=None,
    device=None,
    config=None,
    exp_n=None,
):
    # Initialize the ExampleClient with positional and keyword arguments
    client = ExampleClient(
        "tcp://127.0.0.1:50051",
        "tcp://127.0.0.1:5555",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_examples=num_examples,
        client_id=client_id,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
        device=device,
        config=config,
        exp_n=exp_n,
    )

    # Configure aggregator if ports are provided
    if is_aggr:
        client.set_aggregator_config("127.0.0.1")
        if cfg.shifted_k_sparsification:
            client.set_aggregation_strategy(ShiftedCompression(client.gamma))

    # Start training
    start_time = time.time()
    training_success = client.join()
    if training_success:
        client.split_mask = client.get_split_mask()
        client.n_split = max(x.max() for x in client.split_mask) + 1
        training_success = client.train()

    if training_success:
        print("Client finished the training successfully")
        
        # Check if this is client 1 to perform testing
        if client_id == 1:
            # evaluation on the test set
            print("Client 1 is performing final model testing...")
            time.sleep(1)
            
            # aggregated metrics
            aggregated_metrics = utils.aggregate_client_data(config)
            # utils.print_max_metrics(aggregated_metrics)
            
            # plot and print
            best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(aggregated_metrics, config, exp_n=exp_n, fold=config['fold'], show=False, eris=True)

            # Load the test set
            test_dataset = torch.load(f"../data/datasets/{config['dataset']}_test.pt", weights_only=False)

            # Create the data loader
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config["batch_test"], 
                shuffle=False
            )

            # Reinitialize the model (ensure it matches the trained model architecture)
            test_model = models.model_dict[config["dataset"]](config["model_args"]).to(device)

            # Construct the checkpoint path
            checkpoint_path = f"checkpoints/{config["model_name"]}/{config['dataset']}/model_{best_loss_round}.pth"
            test_model.load_state_dict(torch.load(checkpoint_path,  weights_only=False))
            
            # Define the number of parameters in the model
            num_params = sum(p.numel() for p in test_model.parameters())
            print(f"\033[93mTotal number of parameters in the model: {num_params}\033[0m")

            # Evaluate the model on the test set
            loss_test, accuracy_test, metric_test = evaluate_fn(test_model, device, test_loader, criterion)
            training_time = round((time.time() - start_time)/60, 2)
            print(f"\n\033[93mTest Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}%, F1 Score: {metric_test*100:.2f}%\033[0m")

            # Save metrics as numpy array
            metrics = {
                "loss": loss_test,
                "accuracy": accuracy_test,
                'f1_score': metric_test,
                "time": training_time,
            }
            if cfg.privacy_audit:
                metrics["max_accuracy_mia"] = aggregated_metrics["MIA Accuracy"].max()
                metrics["max_privacy_estimate"] = aggregated_metrics["Privacy"].max()
                metrics["max_acc_accuracy_mia"] = aggregated_metrics["Accumulative MIA Accuracy"].max()
                metrics["max_acc_privacy_estimate"] = aggregated_metrics["Accumulative Privacy"].max()
                metrics["max_accuracy_mia_mean"] = aggregated_metrics["MIA Accuracy Mean"].max()
                metrics["max_privacy_estimate_mean"] = aggregated_metrics["Privacy Mean"].max()
                metrics["max_acc_accuracy_mia_mean"] = aggregated_metrics["Accumulative MIA Accuracy Mean"].max()
                metrics["max_acc_privacy_estimate_mean"] = aggregated_metrics["Accumulative Privacy Mean"].max()

                metrics_to_print = [
                    ("Max MIA Accuracy", "max_accuracy_mia"),
                    ("Max Privacy Estimate", "max_privacy_estimate"),
                    ("Max Accumulative MIA Accuracy", "max_acc_accuracy_mia"),
                    ("Max Accumulative Privacy Estimate", "max_acc_privacy_estimate"),
                    ("Max MIA Accuracy Mean", "max_accuracy_mia_mean"),
                    ("Max Privacy Estimate Mean", "max_privacy_estimate_mean"),
                    ("Max Accumulative MIA Accuracy Mean", "max_acc_accuracy_mia_mean"),
                    ("Max Accumulative Privacy Estimate Mean", "max_acc_privacy_estimate_mean")
                ]
                output = "\n".join([f"{label} {metrics[key]}" for label, key in metrics_to_print])
                print(f'\n\033[93m{output}\033[0m\n')
            
            # Print training time in minutes (grey color)
            print(f"\033[90mTraining time: {training_time} minutes\033[0m")
    
            np.save(f'test_metrics_fold_{config['fold']}.npy', metrics)

        return 0

    return 1


def main():
    # Arguments
    args = parse_args()
    
    # Check GPU and set manual seed
    device = utils.check_gpu(seed=cfg.seed, client_id=args.id)
    utils.set_seed(cfg.seed)
    config = cfg.experiments[args.dataset]
    config['fold'] = args.fold

    # Initialize model
    model = models.model_dict[config["dataset"]](config["model_args"]).to(device)

    # Load data
    data = torch.load(args.shard, weights_only=False)

    # Split the dataset
    train_size = config['client_train_samples'][args.exp_n]
    val_size = int(train_size * 0.3) # 30% for validation
    total_requested = train_size + val_size
    if total_requested > len(data):
        raise ValueError(
            f"Requested train+val samples ({total_requested}) exceed dataset size ({len(data)})!"
        )
    torch.manual_seed(cfg.seed)
    indices = torch.randperm(len(data))[:total_requested]
    subset_data = Subset(data, indices)
    train_dataset, val_dataset = random_split(
        subset_data, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    num_examples = {"train": train_size, "val": val_size}
    print(f"Num samples: {num_examples}")


    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_test"], shuffle=False)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = F.mse_loss if config["n_classes"] == 1 else F.cross_entropy

    # Create directories and delede old files
    if args.id == 1:
        utils.create_delede_folders(config)

    if args.aggregator:
        return start_node(
            is_aggr=True,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_examples=num_examples,
            client_id=args.id,
            train_fn=models.simple_train,
            evaluate_fn=models.simple_test,
            device=device,
            config=config,
            exp_n=args.exp_n,
        )
    else:
        return start_node(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_examples=num_examples,
            client_id=args.id,
            train_fn=models.simple_train,
            evaluate_fn=models.simple_test,
            device=device,
            config=config,
            exp_n=args.exp_n
        )
    

if __name__ == "__main__":
    sys.exit(main())
