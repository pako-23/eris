#!/usr/bin/env python3

# import numpy as np
# from eris import ErisClient
# import torch
# import sys


# class Client(ErisClient):
#     def __init__(self, router_address, subscribe_address):
#         super().__init__(router_address, subscribe_address)
#         # TODO: add model

#     def get_parameters(self):
#         return [param.data.numpy() for param in self.model.parameters()]

#     def set_parameters(self, parameters):
#         for i, param in enumerate(self.model.parameters()):
#             param.data = torch.from_numpy(parameters[i])

#     def fit(self):
#         # TODO: train model
#         pass

#     def evaluate(self):
#         pass


# def start_node(aggr_rpc_port=None, aggr_publish_port=None):
#     client = ExampleClient("tcp://127.0.0.1:50051", "tcp://127.0.0.1:5555")

#     if aggr_rpc_port is not None and aggr_publish_port is not None:
#         client.set_aggregator_config("127.0.0.1", aggr_rpc_port, aggr_publish_port)

#     if client.train():
#         print("Client finished the training successfully")
#         return 0

#     return 1


# def main():
#     if len(sys.argv) == 1:
#         return start_node()
#     elif len(sys.argv) == 3:
#         return start_node(int(sys.argv[1]), int(sys.argv[2]))
#     else:
#         print(
#             f"Usage: {sys.argv[0]} [<aggregator submit port> <aggregator publish port>]",
#             file=sys.stderr,
#         )
#         return 1


# if __name__ == "__main__":
#     sys.exit(main())


import numpy as np
from eris import ErisClient
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import models
from public import utils
from public import config as cfg


class ExampleClient(ErisClient):
    # def __init__(self, router_address, subscribe_address, model):
    #     super().__init__(router_address, subscribe_address, model)
    def __init__(self, 
                 *args, 
                 model, 
                 train_loader, 
                 val_loader,
                 optimizer, 
                 criterion,
                 num_examples, 
                 client_id,
                 train_fn,
                 evaluate_fn,
                 device,
                 **kwargs):
        super().__init__(*args, **kwargs)
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
        self.dataset_name = cfg.dataset_name
        self.predictor_name = model.__class__.__name__
        
    def get_parameters(self):
        return [param.data.numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(parameters[i])

    def fit(self):
        self.model.train(True)

        # for data in self.train_data:
        #     x, y = data
        #     self.model.optimizer.zero_grad()
        #     pred = self.model.forward(x)
        #     loss = self.model.criterion(pred, y)
        #     loss.backward()
        #     self.model.optimizer.step()
        for epoch in range(2):
            self.train_fn(self.model, self.device, self.train_loader, self.optimizer, self.criterion, epoch, self.client_id)
            

    def evaluate(self):
        pass


def start_node(
    aggr_rpc_port=None, 
    aggr_publish_port=None, 
    model = None,
    train_loader = None,
    val_loader = None,
    optimizer = None,
    criterion = None,
    num_examples = None,
    client_id = None,
    train_fn = None,
    evaluate_fn = None,
    device = None,
    ):
    
    client = ExampleClient(
        "tcp://127.0.0.1:50051", 
        "tcp://127.0.0.1:5555", 
        model,
        train_loader, 
        val_loader,
        optimizer, 
        criterion,
        num_examples, 
        client_id,
        train_fn,
        evaluate_fn,
        device,
        )

    if aggr_rpc_port is not None and aggr_publish_port is not None:
        client.set_aggregator_config("127.0.0.1", aggr_rpc_port, aggr_publish_port)

    if client.train():
        print("Client finished the training successfully")
        return 0

    return 1


def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 101),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()

    # check gpu and set manual seed
    device = utils.check_gpu(seed=cfg.seed)
    utils.set_seed(cfg.seed)

    # model and history folder
    model = models.model_dict[cfg.dataset_name](
        models.model_args[cfg.dataset_name]).to(device)

    # Load data
    data = torch.load(f'../data/client_datasets/IID_data_client_{args.id}.pt', weights_only=False)
    
    # Split the dataset
    val_size = int(len(data) * 0.2)  # 20% for validation
    train_size = len(data) - val_size  # 80% for training
    torch.manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    num_examples = {
        "train": train_size,
        "val": val_size
    }

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = F.mse_loss if cfg.n_classes_dict[cfg.dataset_name]==1 else F.cross_entropy

    if len(sys.argv) == 1:
        return start_node()
    elif len(sys.argv) == 3:
        return start_node(
            int(sys.argv[1]), 
            int(sys.argv[2]), 
            model,
            train_loader, 
            val_loader,
            optimizer, 
            criterion,
            num_examples, 
            args.id,
            models.simple_train,
            models.simple_test,
            device,
        )
            
    else:
        print(
            f"Usage: {sys.argv[0]} [<aggregator submit port> <aggregator publish port>]",
            file=sys.stderr,
        )
        return 1




if __name__ == "__main__":
    sys.exit(main())
