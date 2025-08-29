"""
This script implements a Flower-based Federated Learning server using the FedAvg strategy.
The server coordinates the training process by:
- Waiting for clients to connect.
- Distributing the current global model to clients.
- Aggregating client updates after local training.
- Saving model checkpoints and tracking evaluation metrics (e.g., accuracy, F1, privacy leakage).

Key Features:
- Supports custom aggregation and evaluation logic.
- Tracks and logs privacy metrics including membership inference attack (MIA) success and audit estimates.
- Automatically saves the best-performing model and generates training plots.
- Designed for both local and distributed deployments (server IP address configurable).

This server must be started before clients connect. Compatible with client.py or FlowerClient-based implementations.
"""

# Libraries
import flwr as fl
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Parameters, Scalar, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from logging import WARNING
from flwr.common.logger import log
from collections import OrderedDict
import json
import time
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common import NDArrays
from functools import reduce

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import models
from public import utils
from public import config as cfg


# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
    }
    return config
    
# Custom weighted average function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    # accuracy_mia = [num_examples * m["accuracy_mia"] for num_examples, m in metrics]
    accuracy_mia_list = [m["accuracy_mia"] for _, m in metrics]
    accuracy_mia = max(accuracy_mia_list)
    privacy_estimate_list = [m["privacy_estimate"] for _, m in metrics]
    privacy_estimate = max(privacy_estimate_list)
    # validities = [num_examples * m["validity"] for num_examples, m in metrics]
    acc_accuracy_mia_list = [m["accumulative_accuracy_mia"] for _, m in metrics]
    acc_accuracy_mia = max(acc_accuracy_mia_list)
    acc_privacy_estimate_list = [m["accumulative_privacy_estimate"] for _, m in metrics]
    acc_privacy_estimate = max(acc_privacy_estimate_list)
    # mean privacy metrics
    accuracy_mia_mean = [m["accuracy_mia_mean"] for _, m in metrics]
    privacy_estimate_mean = [m["privacy_estimate_mean"] for _, m in metrics]
    accumulative_accuracy_mia_mean = [m["accumulative_accuracy_mia_mean"] for _, m in metrics]
    accumulative_privacy_estimate_mean = [m["accumulative_privacy_estimate_mean"] for _, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "f1_score": sum(f1_scores) / sum(examples),
        # "accuracy_mia": sum(accuracy_mia) / sum(examples) if accuracy_mia[0] > 0 else None,
        "accuracy_mia": accuracy_mia if accuracy_mia > 0 else None,
        "privacy_estimate": privacy_estimate if privacy_estimate > -0.5 else None,
        "accumulative_accuracy_mia": acc_accuracy_mia if acc_accuracy_mia > 0 else None,
        "accumulative_privacy_estimate": acc_privacy_estimate if acc_privacy_estimate > -0.5 else None,
        "accuracy_mia_mean": sum(accuracy_mia_mean) / len(accuracy_mia_mean),
        "privacy_estimate_mean": sum(privacy_estimate_mean) / len(privacy_estimate_mean),
        "accumulative_accuracy_mia_mean": sum(accumulative_accuracy_mia_mean) / len(accumulative_accuracy_mia_mean),
        "accumulative_privacy_estimate_mean": sum(accumulative_privacy_estimate_mean) / len(accumulative_privacy_estimate_mean),
        }

def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.config = config

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        
        ################################################################################
        # Federated averaging aggregation
        ################################################################################
        # Federated averaging - from traditional code
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))   # Global aggregation - traditional - no clustering
        
        # Aggregate custom metrics if aggregation fn was provided   NO FIT METRICS AGGREGATION FN PROVIDED - SKIPPED FOR NOW
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
            
        ################################################################################
        # Save model
        ################################################################################
        if aggregated_parameters_global is not None:

            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(self.model.state_dict(), f"checkpoints/{self.config["model_name"]}/{self.config['dataset']}/model_{server_round}.pth")
        
        return aggregated_parameters_global, aggregated_metrics
    
    
    ############################################################################################################
    # Aggregate evaluation results
    ############################################################################################################
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
            
        print(f"\033[92mRound {server_round} - Aggregated loss: {loss_aggregated:.3f} - Aggregated accuracy: {metrics_aggregated['accuracy']*100:.2f}\033[0m")

        return loss_aggregated, metrics_aggregated
    


    



# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 20),
        default=1,
        help="Specifies the fold to be used",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="mnist",
        choices=list(cfg.experiments.keys()),
    )
    parser.add_argument(
        "--exp_n",
        type=int,
        help="exp number",
        default=0,
    )
    args = parser.parse_args()
    
    # Start time
    start_time = time.time()
    config = cfg.experiments[args.dataset]
    
    # Load the test set
    test_dataset = torch.load(f'../data/datasets/{args.dataset}_test.pt', weights_only=False)

    # Create the data loaders
    test_loader = DataLoader(test_dataset, batch_size=config['batch_test'], shuffle=False)

    # model and history folder
    device = utils.check_gpu(seed=cfg.seed, print_info=True)
    utils.set_seed(cfg.seed)

    # model and history folder    
    # model = config["model"](config["model_args"]).to(device)
    model = models.model_dict[config["dataset"]](config["model_args"]).to(device)

    # Define the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\033[93mTotal number of parameters in the model: {num_params}\033[0m")

    # Create directories and delede old files
    utils.create_delede_folders(config)

    # Define strategy
    strategy = SaveModelStrategy(
        model=model, # model to be trained
        min_fit_clients=config['clients'],  # Never sample less than 10 clients for training
        min_evaluate_clients=config['clients'],   # Never sample less than 5 clients for evaluation
        min_available_clients=config['clients'],  # Wait until all 10 clients are available
        fraction_fit=1.0, # Sample 100 % of available clients for training
        fraction_evaluate=1.0, # Sample 100 % of available clients for evaluation
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        config=config
    )
        
    print(f"\033[94mTraining {config["model_name"]} on {args.dataset} with {config['clients']} clients\033[0m\n")

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address="0.0.0.0:8086",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=config['rounds'][args.exp_n]),
        strategy=strategy,
    )
    # convert history to list
    metrics_distributed = {
        'loss': [k[1] for k in history.losses_distributed],
        'accuracy': [k[1] for k in history.metrics_distributed['accuracy']],
        'f1_score': [k[1] for k in history.metrics_distributed['f1_score']],
        'accuracy_mia': [k[1] for k in history.metrics_distributed['accuracy_mia']],
        'privacy_estimate': [k[1] for k in history.metrics_distributed['privacy_estimate']],
        'accumulative_accuracy_mia': [k[1] for k in history.metrics_distributed['accumulative_accuracy_mia']],
        'accumulative_privacy_estimate': [k[1] for k in history.metrics_distributed['accumulative_privacy_estimate']],
        'accuracy_mia_mean': [k[1] for k in history.metrics_distributed['accuracy_mia_mean']],
        'privacy_estimate_mean': [k[1] for k in history.metrics_distributed['privacy_estimate_mean']],
        'accumulative_accuracy_mia_mean': [k[1] for k in history.metrics_distributed['accumulative_accuracy_mia_mean']],
        'accumulative_privacy_estimate_mean': [k[1] for k in history.metrics_distributed['accumulative_privacy_estimate_mean']]
    }

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder: histories/{config["model_name"]}/{args.dataset}/distributed_metrics_{args.fold}.json")
    with open(f"histories/{config["model_name"]}/{args.dataset}/distributed_metrics_exp_{args.exp_n}_fold_{args.fold}.json", "w") as f:
        json.dump(metrics_distributed, f)

    # Single Plot
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(metrics_distributed, config, exp_n=args.exp_n, fold=args.fold, show=False)
    # best_loss_round = config['rounds'] - 1 # take the last round model
    
    # Privacy estimate plot
    # utils.plot_audit_metrics(client_id, model_name, dataset_name, show=True):

    # Load the best model
    model.load_state_dict(torch.load(f"checkpoints/{config["model_name"]}/{args.dataset}/model_{best_loss_round}.pth", weights_only=False))

    # Evaluate the model on the test set
    criterion = F.mse_loss if config['n_classes'] == 1 else F.cross_entropy
    loss_test, accuracy_test, metric_test = models.simple_test(model, device, test_loader, criterion)
    print(f"\n\033[93mTest Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}, F1 Score: {metric_test*100:.2f} \033[0m")
    
    if cfg.privacy_audit:
        metrics_to_print = [
            ("Max MIA Accuracy", "accuracy_mia"),
            ("Max Privacy Estimate", "privacy_estimate"),
            ("Max Accumulative MIA Accuracy", "accumulative_accuracy_mia"),
            ("Max Accumulative Privacy Estimate", "accumulative_privacy_estimate"),
            ("Max MIA Accuracy Mean", "accuracy_mia_mean"),
            ("Max Privacy Estimate Mean", "privacy_estimate_mean"),
            ("Max Accumulative MIA Accuracy Mean", "accumulative_accuracy_mia_mean"),
            ("Max Accumulative Privacy Estimate Mean", "accumulative_privacy_estimate_mean")
        ]
        output = "\n".join([f"{label} {max(metrics_distributed[key])}" for label, key in metrics_to_print])
        print(f'\n\033[93m{output}\033[0m\n')

    # Print training time in minutes (grey color)
    training_time = round((time.time() - start_time)/60, 2)
    print(f"\033[90mTraining time: {training_time} minutes\033[0m")
    time.sleep(1)
    
    # Save metrics as numpy array
    metrics = {
        "loss": loss_test,
        "accuracy": accuracy_test,
        "f1_score": metric_test,
        "time": training_time,
    }
    if cfg.privacy_audit:
        metrics["max_accuracy_mia"] = max(metrics_distributed["accuracy_mia"])
        metrics["max_privacy_estimate"] = max(metrics_distributed["privacy_estimate"])
        metrics["max_acc_accuracy_mia"] = max(metrics_distributed["accumulative_accuracy_mia"])
        metrics["max_acc_privacy_estimate"] = max(metrics_distributed["accumulative_privacy_estimate"])
        metrics["max_accuracy_mia_mean"] = max(metrics_distributed["accuracy_mia_mean"])
        metrics["max_privacy_estimate_mean"] = max(metrics_distributed["privacy_estimate_mean"])
        metrics["max_acc_accuracy_mia_mean"] = max(metrics_distributed["accumulative_accuracy_mia_mean"])
        metrics["max_acc_privacy_estimate_mean"] = max(metrics_distributed["accumulative_privacy_estimate_mean"])

    np.save(f'test_metrics_fold_{args.fold}.npy', metrics)
    
if __name__ == "__main__":
    main()
