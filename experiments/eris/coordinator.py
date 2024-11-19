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
    return parser.parse_args()


def main():
    args = parse_args()
    config = cfg.experiments[args.dataset_name]

    builder = ErisCoordinatorConfig()
    builder.set_rounds(int(config["rounds"]))
    builder.set_splits(int(config["splits"]))
    builder.set_min_clients(int(config["clients"]))

    coordinator = ErisCoordinator(builder)
    coordinator.start()


if __name__ == "__main__":
    main()
