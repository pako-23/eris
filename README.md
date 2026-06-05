# ERIS: Enhancing Privacy and Scalability in Federated Learning via Federated Shard Aggregation

## 🌍 Overview

**ERIS** is a scalable Federated Learning (FL) framework designed to jointly address privacy, scalability, and utility. The core idea is Federated Shard Aggregation (FSA): instead of sending a full client update to a central server, each client partitions its update into non-overlapping shards and distributes their aggregation across multiple client-side aggregators. This removes the central aggregation bottleneck, balances communication and computation across participants, and ensures that no single observer sees a complete client update.

Because the shards are disjoint and complete, clients can reassemble the same global update induced by centralized FedAvg-style aggregation, preserving utility without relying on heavy cryptography, noise injection, or perturbations that degrade learning. ERIS can further integrate Distributed Shifted Compression (DSC) as a pre-processing layer to reduce transmitted payloads and exposed coordinates, strengthening both scalability and privacy.

<p align="center">
  <img src="plots/eris_overview.png" alt="FLUX Overview" width="100%"/>
</p>


## 📦 Key Features

- **Federated Shard Aggregation.** ERIS introduces FSA, a distributed aggregation mechanism that partitions each client update into non-overlapping shards and assigns their aggregation to multiple client-side aggregators. This removes the central server bottleneck, limits the information visible to any single observer, and preserves the centralized FL update after reassembly.
  
- **Optional Distributed Shifted Compression.** ERIS naturally supports DSC as a pre-processing layer before shard aggregation. DSC reduces the number of transmitted parameters and exposed coordinates, further improving communication scalability and privacy while retaining the utility guarantees provided by FSA.

- **Privacy, scalability, and utility guarantees.** ERIS provides convergence guarantees under standard assumptions and information-theoretic privacy bounds showing that leakage decreases with the observable fraction of each update, the number of aggregators, and, when DSC is enabled, the compression level. Experiments across four image and two text datasets—from small models to modern LLMs—and under two threat models against six SOTA baselines confirm ERIS’s strong privacy–utility–scalability trade-off.


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
  title        = {ERIS: Enhancing Privacy and Scalability in Federated Learning via Federated Shard Aggregation},
  author       = {Dario Fenoglio and Pasquale Polverino and Jacopo Quizi and Martin Gjoreski and Akash Dhasade and Marc Langheinrich},
  year         = {2026},
  eprint       = {2602.08617},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  note         = {Preprint, version 2, May 11, 2026}
}
```

## License
This project is licensed under the GNU General Public License. See `LICENSE` for details.
