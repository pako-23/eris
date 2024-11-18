# Eris

Eris is a framework to build federated AI systems without the need of
a centralized federated server.

---

## To build eris

To build the Python package, install the requirement dependencies
as follows

``` shell
pip install -r requirements.txt
```

Then, you can start a release build as follows

```shell
./setup.py bdist_wheel
```

---

## To run the eris experiments

To run the eris experiments, start by setting up the `eris` package as
follows

``` shell
pip install -r requirements.txt
pip install -r experiments/requirements.txt
./setup.py bdist_wheel
pip install $(find dist/ -name '*.whl') --force-reinstall
```

Then, run the experiments as follows
TODO: setup experiments ....
