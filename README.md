# ERIS: Enhancing Privacy and Communication Efficiency in Serverless Federated Learning

## 🌍 Overview

**ERIS** is a scalable serverless Federated Learning (FL) framework that removes the server bottleneck while preserving FedAvg utility. The core idea is to partition each client update across multiple client-side aggregators so that (i) aggregation is fully distributed and network load is balanced, and (ii) no single entity ever observes a full client update—only a small, randomized subset—yielding inherent privacy benefits. ERIS further integrates a distributed shifted compression mechanism to drastically reduce the number of transmitted (and exposed) parameters.


<p align="center">
  <img src="plots/eris_overview.png" alt="FLUX Overview" width="100%"/>
</p>


## 📦 Key Features
- **Exact serverless aggregation via gradient partitioning.** ERIS introduces a novel gradient partitioning scheme that balances network load and remains mathematically equivalent to FedAvg updates, while amplifying privacy on client updates by limiting the information available to any single observer.

- **Distributed Shifted Compression.** Applies a shift-and-compress strategy to each client gradient, reducing transmitted parameters to less than $3.3\%$ of the model size and cuts distribution time by up to $10^3\times$ in the worst case. Besides improving communication efficiency, compression further limits exposure by shrinking the parameter subset observed per round.

- **Theory and large-scale validation.** We provide convergence guarantees and information-theoretic privacy bounds showing that leakage decreases with the number of aggregators and the compression level. Extensive experiments on three image and two text datasets—from small networks to modern LLMs—and under two threat models against six SOTA baselines confirm ERIS’s strong privacy–utility–efficiency trade-off.


## 🚀 Installation
1. Create and activate a Python environment.
2. Install dependencies.
3. Build the ERIS extension.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```


## 🏃‍♂️ Running Experiments
Each experiment suite has the same structure:
- `public/config.py` contains the main settings.
- `data/` contains dataset download and splitting scripts.
- `eris/`, `fedavg/`, and `soterafl/` (plus `_llm` variants) contain runnable scripts.

General flow:
1. Edit the suite config: `exps_*/public/config.py`.
2. Run a strategy from its folder.

Example (ERIS with unbiased gradient estimator):
```bash
cd exps_unbiased_g_estimator/eris
bash run.sh
```

Run a baseline by switching folders:
```bash
cd exps_unbiased_g_estimator/fedavg
bash run.sh
```

LLM variants use the `_llm` folders:
```bash
cd exps_unbiased_g_estimator/eris_llm
bash run.sh
```

## ⚙️ Experiment Suites
| Suite | Goal | Run From |
| --- | --- | --- |
| `exps_unbiased_g_estimator` | ERIS with unbiased gradient estimator | `eris/`, `fedavg/`, `soterafl/` (+ `_llm`) |
| `exps_biased_g_estimator` | ERIS with biased gradient estimator | `eris/`, `fedavg/`, `soterafl/` (+ `_llm`) |
| `exps_unbiased_pareto` | Pareto front (privacy vs utility), unbiased | `eris/`, `fedavg/`, `soterafl/` (+ `_llm`) |
| `exps_biased_pareto` | Pareto front (privacy vs utility), biased | `eris/`, `fedavg/`, `soterafl/` (+ `_llm`) |
| `exps_dra` | Data Reconstruction Attacks (DLG/iDLG) | `python main.py` |
| `exps_GPT` | GPT-scale experiments and baselines | `bash run.sh` |

## 🗂️ Datasets
- Image, tabular, time-series, and text datasets are downloaded on first run via `data/client_datasets_split.py`.
- To pre-download datasets for a suite:
```bash
cd exps_unbiased_g_estimator/data
python download_datasets.py
```
- Supported datasets for each suite are listed in its `public/config.py`.


## Citation
If you use ERIS, please cite the current preprint:
```
@misc{fenoglio2026eris,
  title        = {ERIS: Enhancing Privacy and Communication Efficiency in Serverless Federated Learning},
  author       = {Dario Fenoglio and Pasquale Polverino and Jacopo Quizi and Martin Gjoreski and Marc Langheinrich},
  year         = {2026},
  note         = {Preprint, February 9, 2026}
}
```

## License
This project is licensed under the GNU General Public License. See `LICENSE` for details.
