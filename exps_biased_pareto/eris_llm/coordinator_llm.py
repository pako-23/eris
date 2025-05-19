"""
This script launches the central ERIS coordinator responsible for orchestrating communication across
distributed clients and aggregators in decentralized federated learning experiments.

It configures the training setup (number of communication rounds, model splits, and clients) based on
the specified dataset and experiment index, then initializes and starts the coordinator node. Designed 
to work jointly with client.py.
"""

#!/usr/bin/env python3

import argparse
import sys
import os

from eris import ErisCoordinatorConfig, ErisCoordinator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import public.config as cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start an Eris coordinator configured with the options for the given experiment"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to run the experiment on",
        choices=list(cfg.experiments.keys()),
    )
    parser.add_argument(
        "--exp_n",
        type=int,
        help="exp number",
        default=0,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = cfg.experiments[args.dataset_name]

    builder = ErisCoordinatorConfig()
    builder.set_rounds(int(config["rounds"][args.exp_n]))
    builder.set_splits(int(config["splits"]))
    builder.set_min_clients(int(config["clients"]))
    # builder.set_publish_port(50051)
    # builder.set_subscribe_port(5555)

    coordinator = ErisCoordinator(builder)
    coordinator.start()


if __name__ == "__main__":
    main()
