"""
This script defines a Flower client for federated learning experiments reflecting SoteriaFL implementation
with LLMs. Each client locally trains a model using the received global parameters, add differential privacy and 
perform a shifted compression of the gradients before sending it to the server for aggregation. The client 
also evaluates the model's performance and privacy leakage (MIA).
It can be deployed locally or across distributed systems by adjusting `server_address`.

Clients are assigned dataset shards, train models locally, and optionally evaluate privacy leakage
using canary samples. After training, results and plots are saved to track performance and leakage.
"""

# Arguments
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 101),
        required=True,
        help="Specifies the artificial data partition",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="mnist",
        choices=["mnist", "cifar10", "imdb", "fmnist"],
    )
    parser.add_argument(
        "--exp_n",
        type=int,
        help="exp number",
        default=0,
    )
    
    return parser.parse_args()
args = parse_args()

# set device for the client
if args.id % 2 == 0:
    device = '2'
else:
    device = '3'
    
# Import Libraies
from collections import OrderedDict
import torch
import flwr as fl
import numpy as np
import copy
import opacus # type: ignore
import gc

from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from datasets import concatenate_datasets, Dataset, load_from_disk # type: ignore
from transformers import ( # type: ignore
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = device  # select the gpu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import utils
from public import models
from public import config as cfg




# Define Flower client 
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,                
                 model: torch.nn.Module,
                 train_data: Dataset,
                 val_data: Dataset,
                 num_examples: dict, 
                 client_id: int,
                 training_args: TrainingArguments, 
                 privacy_audit: bool = True,
                 canary_frac: float = 0.2, 
                 p_value: float = 0.05,
                 k_plus: float = 1 / 3, 
                 k_min: float = 1 / 3,
                 config: dict = {'dataset':'imdb', 'batch':4},
                 exp_n: int = 0  
                ):
        
        # Define the client
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.num_examples = num_examples
        self.client_id = client_id
        self.training_args = training_args
        self.canary_frac = canary_frac
        self.p_value = p_value
        self.k_plus = k_plus
        self.k_min = k_min
        self.privacy_audit = privacy_audit
        self.privacy_estimate = -1
        self.accuracy_mia = -1
        self.acc_privacy_estimate = -1
        self.acc_accuracy_mia = -1
        self.config = config
        self.acc_scores = None
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.exp_n = exp_n
        
        # initialize reference vector
        self.reference_s = []
        for p in self.get_parameters():
            self.reference_s.append(np.zeros_like(p))
        
        # initialize k for sparsity
        self.k = int(self.n_params / np.log2(self.config['rounds'][exp_n]))  # as in SoteriaFL
        w = (self.n_params / self.k) - 1
        self.gamma = np.sqrt((1 + 2 * w) / (2 * (1 + w)**3))

        # prepare dataset auditing
        if self.privacy_audit:
            
            self.n_canaries = int(len(self.train_data) * canary_frac)
            self.canaries = self.train_data.select(range(0, self.n_canaries))
            non_canaries = self.train_data.select(range(self.n_canaries, len(self.train_data)))
            self.scores = np.zeros(self.n_canaries)

            self.canaries.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            non_canaries.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

            # subsample canaries & make new dataloader
            true_in_out = torch.distributions.bernoulli.Bernoulli(torch.ones(self.n_canaries) * 0.5).sample()
            self.true_in_out = true_in_out.numpy()
            canaries_in_idx = torch.nonzero(true_in_out.clone().detach())
            
            # concatenate non_canaries data with samples from canaries with canaries_in_idx
            self.subsampled_train_data = concatenate_datasets([non_canaries, self.canaries.select(canaries_in_idx)])
            self.subsampled_train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

            # Trainer initialization using only the IN set for training
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.subsampled_train_data,
                # eval_dataset=test_data,  # Normal evaluation on the official test set (not pass it in FL)
                compute_metrics=utils.compute_metrics,
            )
        else:
            # Trainer initialization using the full training set
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_data,
                # eval_dataset=test_data,  # Normal evaluation on the official test set (not pass it in FL)
                compute_metrics=utils.compute_metrics,
            )

        if cfg.local_dp:
            # Calculate sample rate = (batch_size / total_number_of_samples)
            if cfg.privacy_audit:
                sample_rate = min(1.0, self.training_args.per_device_train_batch_size / len(self.subsampled_train_data))
            else:
                sample_rate = min(1.0, self.training_args.per_device_train_batch_size / len(self.train_data))

            # Create the dataloader
            self.train_loader_dp = DataLoader(
                self.subsampled_train_data if cfg.privacy_audit else self.train_data,
                batch_size=self.training_args.per_device_train_batch_size,
                shuffle=True,
            )
            
            self.sigma = opacus.accountants.utils.get_noise_multiplier(
                target_epsilon=cfg.epsilon,
                # target_delta=cfg.delta,
                target_delta = 1 / len(self.train_loader_dp),
                sample_rate=sample_rate,
                epochs=int(self.training_args.num_train_epochs), 
                accountant='rdp',  
            ) 
            print(f"Client {self.client_id} - Noise multiplier: {self.sigma}")

            
            # Optimizer and scheduler
            self.optimizer_dp = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_args.learning_rate,
                betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
                eps=self.training_args.adam_epsilon,
                weight_decay=self.training_args.weight_decay,
            )
                
            # Opacus privacy engine
            self.model.train()
            self.privacy_engine = opacus.PrivacyEngine(accountant='rdp', secure_mode=False)
            self.model, self.optimizer_dp, self.train_loader_dp = self.privacy_engine.make_private(
                                module=self.model,
                                optimizer=self.optimizer_dp,
                                data_loader=self.train_loader_dp,
                                noise_multiplier=self.sigma,
                                max_grad_norm=cfg.sensitivity,
                                poisson_sampling=False,
                            )     
            
            if client_id == 1:
                print(f"\n\033[94mLocal Differential Privacy with introduced noise_value_sd: {self.sigma}\033[0m\n")


    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, params_in, config):
        self.set_parameters(params_in)  

        # privacy auditing
        if self.privacy_audit:
            if True:
                """
                Training with DP
                """                
                models.train_llm_with_opacus(
                    self.model, 
                    self.trainer.args.device, 
                    self.subsampled_train_data,  
                    self.training_args, 
                    self.sigma,
                    1 / len(self.train_loader_dp), 
                    client_id=self.client_id
                    ) 
            # else:
            #     """
            #     Traditional training without DP
            #     """
            #     self.trainer.train()
            #     training_loss = [log['loss'] for log in self.trainer.state.log_history if 'loss' in log]

                
            """
            shifted k-random sparsification on the gradients [SOTERIAFL]
            """
            # calculate gradients
            params_out = self.get_parameters(config)
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
            
            # normalize client update vector
            client_update = utils.parameters_to_1d(params_out) - utils.parameters_to_1d(params_in)
            client_update = client_update / np.linalg.norm(client_update)

            # compute scores for each canary, used to predict membership            
            scores = []
            # canary_loader = torch.utils.data.DataLoader(canaries, batch_size=cfg.batch_size, shuffle=False)
            # if cfg.score_fn == 'whitebox':
            #     self.set_parameters(params_in)
            #     for samples, targets in self.canary_loader:
            #         scores.extend(self.score_with_pseudograd_batch(samples, targets, client_update))
            #     self.set_parameters(params_out)
            if cfg.score_fn == 'blackbox':
                # self.set_parameters(params_in)  # TO REMOVE
                scores = self.score_blackbox_batch(self.canaries)
                # self.set_parameters(params_out) # TO REMOVE
            else:
                NotImplementedError(f'score function {cfg.score_fn} is not known')

            # accumulative leakage
            if self.acc_scores is None:
                self.acc_scores = copy.deepcopy(scores)
            else:
                self.acc_scores = self.acc_scores + np.asarray(scores)

            # lower-bound privacy budget evaluation
            self.accuracy_mia, self.privacy_estimate = self.evaluate_privacy(scores)
            self.acc_accuracy_mia, self.acc_privacy_estimate = self.evaluate_privacy(self.acc_scores)
            
            utils.save_audit_metrics(
                config["current_round"], 
                self.accuracy_mia, 
                self.privacy_estimate, 
                self.acc_accuracy_mia, 
                self.acc_privacy_estimate, 
                self.accuracy_mia, 
                self.privacy_estimate, 
                self.acc_accuracy_mia, 
                self.acc_privacy_estimate, 
                client_id=self.client_id,
                history_folder=f"histories/{self.config['model_name']}/{self.config['dataset']}/"
                )
        
        else: # NO AUDITING
            if True:   
                """
                Training with DP
                """                
                models.train_llm_with_opacus(
                    self.model, 
                    self.trainer.args.device, 
                    self.train_data,  
                    self.training_args, 
                    self.sigma, 
                    1 / len(self.train_loader_dp), 
                    client_id=self.client_id
                    )
            # else:
            #     """
            #     Traditional training without DP
            #     """
            #     self.trainer.train()
            #     training_loss = [log['loss'] for log in self.trainer.state.log_history if 'loss' in log]
                
                    
            """
            k-random sparsification on the gradients [Part of SOTERIAFL]
            """
            # calculate gradients
            params_out = self.get_parameters(config)
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
            

        gc.collect() 
        torch.cuda.empty_cache() 
        if self.client_id == 1:
            print(f"\033[91mTraining Round: {config["current_round"]}\033[0m")
        return params_out, self.num_examples["train"], {}
    
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.trainer.model = self.model
        eval_results = self.trainer.evaluate(eval_dataset=self.val_data)
        loss = eval_results.get("eval_loss", None)
        accuracy = eval_results.get("eval_accuracy", None)
        f1_score = eval_results.get("eval_f1", None)
        
        # save loss and accuracy client
        utils.save_client_metrics(config["current_round"], loss, accuracy, f1_score, client_id=self.client_id,
                                    history_folder=f"histories/{self.config['model_name']}/{self.config['dataset']}/")
        
        return float(loss), self.num_examples["val"], {
            "accuracy": float(accuracy),
            "f1_score": float(f1_score),
            "privacy_estimate": self.privacy_estimate,
            "accuracy_mia": self.accuracy_mia,
            "accumulative_privacy_estimate": self.acc_privacy_estimate,
            "accumulative_accuracy_mia": self.acc_accuracy_mia,
            "privacy_estimate_mean": self.privacy_estimate,
            "accuracy_mia_mean": self.accuracy_mia,
            "accumulative_privacy_estimate_mean": self.acc_privacy_estimate,
            "accumulative_accuracy_mia_mean": self.acc_accuracy_mia
        }


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
    
    
    # def score_with_pseudograd_batch(self, samples, targets, client_update):
    #     '''
    #     Computes membership inference attack scores for a batch by 
    #     computing the inner product between the 'pseudogradient'
    #     represented by client update and the true gradients
    #     for each sample in the batch.
    #     '''
    #     self.model.to(self.device)  # Ensure model is on the correct device
    #     samples = samples.to(self.device)
    #     targets = targets.to(self.device)
        
    #     # Forward pass
    #     predictions = self.model(samples)
    #     losses = torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
        
    #     scores = []
    #     for loss in losses:
    #         # Compute gradients for each sample
    #         audit_grad = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
    #         # audit_grad = parameters_to_1d(audit_grad)
    #         audit_grad = np.concatenate([x.cpu().flatten() for x in audit_grad])
    #         score = np.dot(client_update, - audit_grad)
    #         scores.append(score)
        
    #     return scores


    def score_blackbox_batch(self, data):        
        prediction_output = self.trainer.predict(data)
        logits = torch.tensor(prediction_output.predictions)
        labels = torch.tensor(prediction_output.label_ids)

        # Compute per-sample loss
        losses = cross_entropy(logits, labels, reduction='none')

        # Return scores
        return -losses.cpu().numpy()


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
    









# main
def main(args)->None:
    # Arguments
    args = parse_args()

    # check gpu and set manual seed
    device = utils.check_gpu(seed=cfg.seed, client_id=args.id)
    utils.set_seed(cfg.seed)
    config = cfg.experiments[args.dataset]

    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained(config["model_name"], num_labels=config["n_classes"])
    
    if args.id == 1:
        utils.print_num_parameters(model)
        
    # Load data
    data = load_from_disk(f"../data/client_datasets/IID_data_client_{args.id}")
    
    # Split the dataset
    train_size = config['client_train_samples'][args.exp_n]
    val_size = int(train_size * 0.3) # 30% for validation
    total_requested = train_size + val_size
    if total_requested > len(data):
        raise ValueError(
            f"Requested train+val samples ({total_requested}) exceed dataset size ({len(data)})!"
        )

    # select the first 1000 samples for the sub
    torch.manual_seed(cfg.seed)
    # shuffle
    data = data.shuffle(seed=cfg.seed)
    # select data
    train_data = data.select(range(0, train_size))
    val_data = data.select(range(train_size, total_requested))    
    num_examples = {"train": train_size,"val": val_size}
    print(f"Num samples: {num_examples}")

    # Start Flower client    
    client = FlowerClient(
                        model, 
                        train_data=train_data, 
                        val_data=val_data, 
                        num_examples=num_examples, 
                        client_id=args.id,
                        training_args=config["training_args"], 
                        privacy_audit=cfg.privacy_audit,
                        canary_frac=cfg.canary_frac,
                        p_value=cfg.p_value,
                        k_plus=cfg.k_plus,
                        k_min=cfg.k_min,
                        config=config, 
                        exp_n=args.exp_n                        
                        ).to_client()
    fl.client.start_client(server_address="[::]:8098", client=client, max_wait_time=40) # local host
    
    # read saved data and plot
    utils.plot_client_metrics(args.id, config, show=False)
    






if __name__ == "__main__":
    main(args)
