### Gradient Obfuscation Gives a False Sense of Security in Federated Learning
![](https://img.shields.io/badge/Python-3-blue) ![](https://img.shields.io/badge/Pytorch-1.9.0-blue)

This code is an adaptation of the original repo of the paper. \


#### Prerequisites

- install Python packages
    ```bash
    pip3 install -r requirements.txt
    ```

- Download the pretrained models and put them under `model_zoos` ([link](https://huggingface.co/erickyue/rog_modelzoo/tree/main))

- Download the csv file (https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv) and put it under `data` folder

- The images for a minimal runnable example has been included under `data` folder. The ImageNet validation dataset can be used for a full test. 

<br />

#### Example
- Run prepare_dataset.py to generate the dataset for CIFAR10
- Run train_post_cifar10.py to train the generative model needed for reconstruction on CIFAR10

- Run the example with QSGD: 
    ```bash
    python3 main.py
    ```
    The script will load the configuration file `config.yaml`. The results will be stored under `experiments`.

- Run the example with FedAvg:
    ```bash
    python3 attack_fedavg.py
    ```
    The script will load the configuration file `config_fedavg.yaml`. 

    You can change the settings in the configuration file. For example, use a different compression scheme with 
    ```
    compress: topk
    ```

<br />

