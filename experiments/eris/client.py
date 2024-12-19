#!/usr/bin/env python3

import numpy as np
from eris import ErisClient
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import sys
import os
import argparse
import time
import copy

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
        self.k_plus = cfg.k_plus
        self.k_min = cfg.k_min
        self.p_value = cfg.p_value
        self.canary_frac = cfg.canary_frac
        self.privacy_audit = cfg.privacy_audit
        self.privacy_estimate = -1
        self.accuracy_mia = -1
        self.acc_privacy_estimate = -1
        self.acc_accuracy_mia = -1
        self.acc_scores = None

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

        # local differential privacy
        self.local_dp = cfg.local_dp
        if self.local_dp:
            self.dp_funtion = utils.LocalDpMod(cfg.clipping_norm, cfg.epsilon, cfg.delta, self.client_id)
            noise_value_sd = (cfg.sensitivity * np.sqrt(2 * np.log(1.25 / cfg.delta)) / cfg.epsilon)
            if client_id == 1:
                print(f"\n\033[94mLocal Differential Privacy with noise_value_sd: {noise_value_sd}\033[0m\n")


    def get_parameters(self):
        self.model.to("cpu")
        return [param.data.numpy() for param in self.model.parameters()]


    def set_parameters(self, parameters):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(parameters[i])


    def fit(self):    
        # get initial parameters
        params_in = self.get_parameters()
             
        # save previous aggregated model if client 1
        if self.client_id == 1:
            torch.save(self.model.state_dict(), f"checkpoints/{self.predictor_name}/{self.config["dataset"]}/model_{self.current_round}.pth")

        self.model.train(True)
        self.model.to(self.device)
        self.current_round += 1
        
        # privacy auditing
        if self.privacy_audit:
            # train
            for epoch in range(self.config["epochs"]):
                self.train_fn(
                    self.model,
                    self.device,
                    self.subsampled_train_loader,
                    self.optimizer,
                    self.criterion,
                    epoch,
                    self.client_id,
                )

            # Local Differential Privacy
            if cfg.local_dp:
                params_out = self.dp_funtion(params_in, self.get_parameters())
                self.set_parameters(params_out)
            else:
                params_out = self.get_parameters()

            # normalize client update vector
            client_update = utils.parameters_to_1d(params_out) - utils.parameters_to_1d(params_in)
            client_update = client_update / np.linalg.norm(client_update)

            # compute scores for each canary, used to predict membership            
            scores = []
            # canary_loader = torch.utils.data.DataLoader(canaries, batch_size=cfg.batch_size, shuffle=False)
            if cfg.score_fn == 'whitebox':
                self.set_parameters(params_in)
                for samples, targets in self.canary_loader:
                    scores.extend(self.score_with_pseudograd_batch(samples, targets, client_update))
                self.set_parameters(params_out)
            if cfg.score_fn == 'blackbox':
                for samples, targets in self.canary_loader:
                    scores.extend(self.score_blackbox_batch(samples, targets, client_update))
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
            
            utils.save_audit_metrics(self.current_round, self.accuracy_mia, self.privacy_estimate, self.acc_accuracy_mia, 
                                    self.acc_privacy_estimate, client_id=self.client_id,
                                    history_folder=f"histories/{self.config['model_name']}/{self.config['dataset']}/"
                                    )
        
        else:
            # train
            for epoch in range(self.config["epochs"]):
                self.train_fn(
                    self.model,
                    self.device,
                    self.train_loader,
                    self.optimizer,
                    self.criterion,
                    epoch,
                    self.client_id,
                )
            
            # Local Differential Privacy
            if cfg.local_dp:
                params_out = self.dp_funtion(params_in, self.get_parameters())
                self.set_parameters(params_out)


    def evaluate(self):
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


    def score_blackbox_batch(self, samples, targets, client_update):
        with torch.no_grad():
            self.model.to(self.device)  # Ensure model is on the correct device
            samples = samples.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(samples)
            losses = torch.nn.functional.cross_entropy(predictions, targets, reduction='none').cpu()

            return -losses


def start_node(
    aggr_rpc_port=None,
    aggr_publish_port=None,
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
    )

    # Configure aggregator if ports are provided
    if aggr_rpc_port is not None and aggr_publish_port is not None:
        client.set_aggregator_config("127.0.0.1", aggr_rpc_port, aggr_publish_port)

    # Start training
    start_time = time.time()
    training_success = client.join()
    if training_success:
        client.split_mask = client.get_split_mask()
        training_success = client.train()

    if training_success:
        print("Client finished the training successfully")
        
        # Check if this is client 1 to perform testing
        if client_id == 1:
            print("Client 1 is performing final model testing...")
            
            # Load the test set
            test_dataset = torch.load(f"../data/datasets/{config['dataset']}_test.pt", weights_only=False)

            # Create the data loader
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config["batch_test"], 
                shuffle=False
            )

            # Reinitialize the model (ensure it matches the trained model architecture)
            test_model = config["model"](config["model_args"]).to(device)

            # Determine the best loss round
            # You need to implement logic to retrieve `best_loss_round`
            # This could be stored in a file, returned by `client.train()`, or managed within `config`
            best_loss_round = config['rounds']-1  # Replace with actual logic

            # Construct the checkpoint path
            checkpoint_path = f"checkpoints/{config["model_name"]}/{config['dataset']}/model_{best_loss_round}.pth"
            test_model.load_state_dict(torch.load(checkpoint_path,  weights_only=False))

            # Evaluate the model on the test set
            loss_test, accuracy_test, metric_test = evaluate_fn(test_model, device, test_loader, criterion)
            print(f"\n\033[93mTest Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}%, F1 Score: {metric_test*100:.2f}%\033[0m\n")
            
            print(f"\n\033[93mTest Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}, F1 Score: {metric_test*100:.2f} \
                            Max MIA Accuracy {max(metrics_distributed["accuracy_mia"])} \
                            Max Privacy Estimate {max(metrics_distributed["privacy_estimate"])} \
                            Max Accumulative MIA Accuracy {max(metrics_distributed["accumulative_accuracy_mia"])} \
                            Max Accumulative Privacy Estimate {max(metrics_distributed["accumulative_privacy_estimate"])} \033[0m\n")

            # Print training time in minutes (grey color)
            training_time = round((time.time() - start_time)/60, 2)
            print(f"\033[90mTraining time: {training_time} minutes\033[0m")
    
            # Save metrics as numpy array
            metrics = {
                "loss": loss_test,
                "accuracy": accuracy_test,
                'f1_score': metric_test,
                "time": training_time,
            }
            np.save(f'test_metrics_fold_{config['fold']}.npy', metrics)

        return 0

    return 1


def main():
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
        "--submit-port",
        type=int,
        help="Aggregator submit port",
    )
    parser.add_argument(
        "--publish-port",
        type=int,
        help="Aggregator publish port",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="mnist",
        choices=list(cfg.experiments.keys()),
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
    args = parser.parse_args()
    
    # Check GPU and set manual seed
    device = utils.check_gpu(seed=cfg.seed)
    utils.set_seed(cfg.seed)
    config = cfg.experiments[args.dataset]
    config['fold'] = args.fold

    # Initialize model
    model = config["model"](config["model_args"]).to(device)

    # Load data
    data = torch.load(args.shard, weights_only=False)

    # Split the dataset
    val_size = int(len(data) * 0.2)  # 20% for validation
    train_size = len(data) - val_size  # 80% for training
    torch.manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(data, [train_size, val_size])
    num_examples = {"train": train_size, "val": val_size}

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_test"], shuffle=False)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = F.mse_loss if config["n_classes"] == 1 else F.cross_entropy

    # Create directories and delede old files
    if args.id == 1:
        utils.create_delede_folders(config)

    if args.submit_port and args.publish_port:
        return start_node(
            aggr_rpc_port=args.submit_port,
            aggr_publish_port=args.publish_port,
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
        )
    

if __name__ == "__main__":
    sys.exit(main())
