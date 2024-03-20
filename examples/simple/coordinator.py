#!/usr/bin/env python3

from eris import ErisCoordinator, ErisCoordinatorBuilder


def main():
    builder = ErisCoordinatorBuilder()
    builder.add_rounds(1)
    builder.add_splits(4)
    coordinator = ErisCoordinator(builder)
    coordinator.start()


if __name__ == "__main__":
    main()
