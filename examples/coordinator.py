#!/usr/bin/env python3

from eris import ErisCoordinator, ErisCoordinatorBuilder


def main():
    builder = ErisCoordinatorBuilder()
    coordinator = ErisCoordinator(builder)
    coordinator.start()


if __name__ == "__main__":
    main()
