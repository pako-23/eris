# ERIS: Enhancing Privacy and Communication Efficiency in Decentralized Federated Learning

ERIS is a decentralized Federated Learning (FL) framework that jointly addresses the challenges of communication bottlenecks and gradient-based privacy attacks without sacrificing model accuracy. By partitioning model updates across multiple client-side aggregators and employing a distributed shifted compression mechanism, ERIS eliminates the server bottleneck, provably converges under standard assumptions, and bounds mutual information leakage—establishing a new Pareto frontier for scalable, privacy-preserving FL on large models.

## 🚀 Key Features
- **Decentralized Aggregation:** Distributes aggregation workload across \(A\) client-side aggregators, balancing network load and removing the single-server bottleneck.
- **Distributed Shifted Compression:** Applies a shift-and-compress strategy to each client gradient, reducing transmitted parameters to less than 6% in the worst case while preserving convergence.
- **Model Partitioning:** Splits compressed gradients into disjoint shards sent to different aggregators, ensuring no single entity sees full updates.
- **Theoretical Guarantees:** Proves convergence rate matching FedAvg and an information-theoretic bound on privacy leakage that scales inversely with the number of aggregators.
- **Strong Empirical Performance:** Matches state-of-the-art accuracy on image and text benchmarks (MNIST, CIFAR-10, LFW, IMDB) while reducing membership inference success from ~83% to ~65%, cutting communication cost by >94%, and speeding distribution by up to 1000×.


## 📦 Installation
1. Clone the repository
    ``` shell
    git clone https://github.com/...
    cd eris
    ```
2. Install dependencies
    ``` shell
    pip install -r requirements.txt
    ```
3. Build and install the package
    ```shell
    ./setup.py bdist_wheel
    pip install $(find dist/ -name '*.whl') --force-reinstall
    ```


## ⚙️ Configuration
Experiments are organized under method-specific folders: `eris/`, `fedavg/`, `soteriafl/`, and their respective variations with LLMs. Each contains a `public/config.py` and a `run.sh` script. Additionally, dedicated `exps_*` folders group privacy and attack scenarios:
- `exps_unbiased_g_estimator` and `exps_unbiased_pareto`: Membership Inference Attacks (MIAs) with unbiased gradient estimator $\mathbb{E}_t[\tilde{\mathbf{g}}_k^t]\!=\!\nabla f_k(\mathbf{x^t})$; varying local sample counts and privacy mechanism strengths to study privacy–utility trade-offs and construct Pareto fronts.
- `exps_biased_g_estimator` and `exps_biased_pareto`: MIAs with biased gradient estimator $\mathbb{E}_t[\tilde{\mathbf{g}}_k^t]\!=\!\nabla f_k(\mathbf{x^t}) + C$; analogous analyses under biased gradients.
- `exps_dra`: Data Reconstruction Attacks (DRAs) analysis.

Key configurable parameters in each MIA method folder’s `public/config.py` include:
- **dataset:** MNIST, CIFAR10, LFW, or IMDB.
- **model:** e.g., LeNet5, ResNet9, or DistilBERT.
- **aggregators (A):** Number of client-side aggregators to split updates, denoted as `splits`.
- **compression:** Compression constant and sparsification type (shifted vs. simple).
- **privacy mechanism:** Options like pruning, LDP, (for both ERIS and baselines) or native ERIS partitioning.
- **privacy audit**: Membership inference or reconstruction attack settings.
- **training**: Learning rate, batch size, local epochs, communication rounds, etc.
Customize any parameter in `public/config.py` to reproduce different scenarios.

## 🏃‍♂️ Running Experiments
Once configured, to run ERIS-specific experiments:
```bash
cd 'strategy'
bash run.sh
```
Results—including metrics, training history, visualizations, and model checkpoints—are saved under the `strategy/results`, `strategy/history`, `strategy/images`, and `strategy/checkpoints` directories, respectively.

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE – see the LICENSE file for details.

## Citation
If you use ERIS in your research, please cite our NeurIPS 2025 paper:
```
@inproceedings{anonymous2025flux,
  title        = {{ERIS}: {Enhancing Privacy and Communication Efficiency in Decentralized Federated Learning}},
  author       = {Anonymous Authors},
  year         = {2025}
}
```
