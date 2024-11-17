#!/usr/bin/env python3

# import argparse
# import sys
# import os

# from eris import ErisCoordinatorConfig, ErisCoordinator

# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # from public.config import experiments


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Start an eris coordinator configured with the options for the given experiment"
#     )
#     parser.add_argument(
#         "--experiment",
#         type=str,
#         default="mnist",
#         help="The experiment to run",
#         choices=list(experiments.keys()),
#     )
#     return parser.parse_args()


# def main():
#     # args = parse_args()
#     # config = experiments[args.experiment]
#     builder = ErisCoordinatorConfig()
#     builder.set_rounds(5)
#     builder.set_splits(5)
#     builder.set_min_clients(10)

#     coordinator = ErisCoordinator(builder)
#     coordinator.start()


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3

from eris import ErisCoordinatorConfig, ErisCoordinator


def main():
    builder = ErisCoordinatorConfig()
    builder.set_rounds(2)
    builder.set_splits(5)
    builder.set_min_clients(10)

    coordinator = ErisCoordinator(builder)
    coordinator.start()


if __name__ == "__main__":
    main()
