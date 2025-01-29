#!/usr/bin/env python3

from eris import ErisCoordinatorConfig, ErisCoordinator


def main():
    builder = ErisCoordinatorConfig()
    builder.set_rounds(5)
    builder.set_splits(5)
    builder.set_min_clients(10)

    coordinator = ErisCoordinator(builder)
    coordinator.start()


if __name__ == "__main__":
    main()
