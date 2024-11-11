#!/usr/bin/env python3

import numpy as np
from eris import ErisClient
import sys


class Client(ErisClient):
    def __init__(self, router_address, subscribe_address):
        super().__init__(router_address, subscribe_address)
        # TODO: add model

    def get_parameters(self):
        return [param.data.numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(parameters[i])

    def fit(self):
        # TODO: train model
        pass

    def evaluate(self):
        pass


def start_node(aggr_rpc_port=None, aggr_publish_port=None):
    client = ExampleClient("tcp://127.0.0.1:50051", "tcp://127.0.0.1:5555")

    if aggr_rpc_port is not None and aggr_publish_port is not None:
        client.set_aggregator_config("127.0.0.1", aggr_rpc_port, aggr_publish_port)

    if client.train():
        print("Client finished the training successfully")
        return 0

    return 1


def main():
    if len(sys.argv) == 1:
        return start_node()
    elif len(sys.argv) == 3:
        return start_node(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print(
            f"Usage: {sys.argv[0]} [<aggregator submit port> <aggregator publish port>]",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
