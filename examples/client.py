#!/usr/bin/env python3

from eris import ErisClient


class Client(ErisClient):
    def get_parameters(self):
        pass

    def fit(self):
        pass

    def evaluate(self):
        pass


def main():
    c = ErisClient("127.0.0.1:50051", "0.0.0.0", 50052)
    c.run()


if __name__ == "__main__":
    main()
