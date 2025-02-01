#!/usr/bin/env python3

# Libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from tslearn.datasets import UCR_UEA_datasets  # type: ignore
from ucimlrepo import fetch_ucirepo  # type: ignore
import os
import numpy as np
import shutil
from transformers import ( # type: ignore
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)


########################################################################################
# MNIST - IMAGE CLASSIFICATION
########################################################################################
def download_mnist():
    print(
        "\033[93m\nDownloading MNIST dataset...\033[0m"
    )  # Define transformations for the datasets
    transform_mnist = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Normalize with mean and std for MNIST
        ]
    )

    # Download and load the MNIST training and test datasets
    trainset_mnist = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform_mnist
    )
    testset_mnist = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform_mnist
    )

    # Save the dataset
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    torch.save(trainset_mnist, "datasets/mnist_train.pt")
    torch.save(testset_mnist, "datasets/mnist_test.pt")
    print("MNIST training dataset saved correctly as numpy file.")

    # Remove 'data' folder
    shutil.rmtree("data")


########################################################################################
# CIFAR-10 - IMAGE CLASSIFICATION
########################################################################################
def download_cifar10():
    print("\033[93m\nDownloading CIFAR-10 dataset...\033[0m")
    # Define transformations for the datasets
    transform_cifar10 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),  # Normalize with mean and std for CIFAR-10
        ]
    )

    # Download and load the CIFAR-10 training and test datasets
    trainset_cifar10 = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_cifar10
    )
    testset_cifar10 = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_cifar10
    )

    # Save as torch tensor
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    torch.save(trainset_cifar10, "datasets/cifar10_train.pt")
    torch.save(testset_cifar10, "datasets/cifar10_test.pt")
    print("CIFAR-10 training dataset saved correctly as numpy file.")

    # Remove 'data' folder
    shutil.rmtree("data")


########################################################################################
# F-MNIST
########################################################################################
def download_fashion_mnist():
    print("\033[93m\nDownloading Fashion-MNIST dataset...\033[0m")

    # Define transformations for the dataset
    transform_fashion_mnist = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.2860406,), (0.35302424,)
            ),  # Normalize with mean and std for Fashion-MNIST
        ]
    )

    # Download and load the Fashion-MNIST training and test datasets
    trainset_fashion = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform_fashion_mnist
    )
    testset_fashion = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform_fashion_mnist
    )

    # Save the dataset
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    torch.save(trainset_fashion, "datasets/fmnist_train.pt")
    torch.save(testset_fashion, "datasets/fmnist_test.pt")
    print("Fashion-MNIST training and test datasets saved correctly as torch files.")

    # Remove 'data' folder
    shutil.rmtree("data")


########################################################################################
# AIRLINE PASSENGERS - TIME SERIES
########################################################################################
# Function to create a dataset where X is the number of passengers at t, t-1, ..., t-n and Y is the passengers at t+1
# def create_dataset(data, window_size=1):
#     X, Y = [], []
#     for i in range(len(data) - window_size):
#         X.append(data[i:(i + window_size)])
#         Y.append(data[i + window_size])
#     return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# def download_airline():
#     print("\nDownloading Airline Passengers dataset...")
#     # Load the Air Passenger dataset
#     url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
#     data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

#     # Scale the data
#     scaler = StandardScaler()
#     data['Passengers'] = scaler.fit_transform(data['Passengers'].values.reshape(-1, 1))

#     # Define the window size
#     window_size = 30

#     # Create the dataset
#     X, Y = create_dataset(data['Passengers'].values, window_size)

#     # Split the data into training and test sets
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

#     # Create TensorDataset for training and testing
#     train_dataset = TensorDataset(X_train, Y_train)
#     test_dataset = TensorDataset(X_test, Y_test)

#     # Save the dataset as torch tensor
#     if not os.path.exists('datasets'):
#         os.makedirs('datasets')

#     torch.save(train_dataset, 'datasets/airline_train.pt')
#     torch.save(test_dataset, 'datasets/airline_test.pt')
#     print("Airline Passengers dataset saved correctly as csv file.")


def create_window(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        _x = data[i : (i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def download_airline():
    print("\033[93m\nDownloading Airline Passengers dataset...\033[0m")
    # Load the Air Passenger dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    data = pd.read_csv(url)
    data = data.iloc[:, 1:2].values

    # Scale the data
    sc = StandardScaler()
    training_data = sc.fit_transform(data)

    # Create windows
    window_size = 30
    x, y = create_window(training_data, window_size)

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.20, shuffle=False
    )

    # Create TensorDataset for training and testing
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )

    # Save the dataset as torch tensor
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    torch.save(train_dataset, "datasets/airline_train.pt")
    torch.save(test_dataset, "datasets/airline_test.pt")
    print("Airline Passengers dataset saved correctly as csv file.")


########################################################################################
# ADULT - TABULAR CLASSIFICATION
########################################################################################
def download_adult():
    print("\033[93m\nDownloading Adult dataset...\033[0m")
    # Load the dataset from UCI Machine Learning Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    data = pd.read_csv(url, header=None, names=column_names, na_values=" ?")

    # Check for missing values
    # print("\nMissing values per column:")
    # print(data.isnull().sum())

    # Preprocessing
    # Separate features and target variable
    X = data.drop("income", axis=1)
    y = data["income"]

    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Preprocessing for numerical and categorical features
    numerical_features = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing for both numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Preprocess the data
    X_preprocessed = preprocessor.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_preprocessed.toarray(), y, test_size=0.2, random_state=42
    )

    # Create TensorDataset for training and testing
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )

    # Save the dataset as torch tensor
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    torch.save(train_dataset, "datasets/adult_train.pt")
    torch.save(test_dataset, "datasets/adult_test.pt")
    print("Adult dataset saved correctly as torch tensor.")


########################################################################################
# Breast Cancer Wisconsin (Diagnostic) - TABULAR CLASSIFICATION
########################################################################################
def download_breast(preprocess=True):
    print("\033[93m\nDownloading Breast Cancer Wisconsin dataset..\033[0m")

    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    y.loc[:, "Diagnosis"] = y["Diagnosis"].map({"M": 1, "B": 0})

    # # metadata and variable information
    # print(f"Metadata: {breast_cancer_wisconsin_diagnostic.metadata}")
    # print(f"Info variables: {breast_cancer_wisconsin_diagnostic.variables}")

    # Preprocessing dataframe X
    if preprocess:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y["Diagnosis"].values, test_size=0.2, random_state=42
    )
    Y_train = Y_train.astype(float)
    Y_test = Y_test.astype(float)

    # Create TensorDataset for training and testing
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )

    # Save the dataset as torch tensor
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    torch.save(train_dataset, "datasets/breast_train.pt")
    torch.save(test_dataset, "datasets/breast_test.pt")


########################################################################################
# Diabetes - TABULAR CLASSIFICATION
################################################################################
def download_diabetes(preprocess=True):
    print("\033[93m\nDownloading Diabetes Health Indicators dataset...\033[0m")

    # fetch dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

    # data (as pandas dataframes)
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets

    # # metadata and variable information
    # print(f"Metadata: {breast_cancer_wisconsin_diagnostic.metadata}")
    # print(f"Info variables: {breast_cancer_wisconsin_diagnostic.variables}")

    # Preprocessing dataframe X
    if preprocess:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y["Diabetes_binary"].values, test_size=0.2, random_state=42
    )
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Create TensorDataset for training and testing
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )

    # Save the dataset as torch tensor
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    torch.save(train_dataset, "datasets/diabetes_train.pt")
    torch.save(test_dataset, "datasets/diabetes_test.pt")


########################################################################################
# LSST - TIME SERIES
########################################################################################
def download_lsst():
    print("\033[93m\nDownloading LSST dataset...\033[0m")
    # Load the dataset from UCR Time Series Classification Archive
    ucr_loader = UCR_UEA_datasets()

    # Load a specific dataset
    X_train, y_train, X_test, y_test = ucr_loader.load_dataset("LSST")

    # Remove the labels (92 and 95) from the dataset - few samples, errors when splitting
    mask = y_test != "92"
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = y_train != "92"
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = y_test != "95"
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = y_train != "95"
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Encode the string labels
    label_encoder = LabelEncoder()

    # Assuming y_train, y_val, y_test are originally string labels
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Create TensorDataset for training and testing
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    # Save the dataset as torch tensor
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    torch.save(train_dataset, "datasets/lsst_train.pt")
    torch.save(test_dataset, "datasets/lsst_test.pt")
    print("LSST dataset saved correctly as torch tensor.")


########################################################################################
# IMDB Movies Dataset - TEXT CLASSIFICATION
########################################################################################
def download_imdb():
    # 1. Load the IMDb dataset
    dataset = load_dataset("imdb")
    
    # split the dataset into training and test sets
    train_data = dataset["train"]
    train_data = train_data.shuffle(seed=42)
    test_data = dataset["test"]

    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Preprocessing (tokenization) function
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # Tokenize the datasets
    train_data = train_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)

    # Set formats for PyTorch
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Save the datasets 
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    train_data.save_to_disk("datasets/imdb_train")
    test_data.save_to_disk("datasets/imbd_test")
    print("IMDb dataset saved correctly as torch files.")






if __name__ == "__main__":
    print("\033[93m\n--> Start downloading all datasets <--\033[0m")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != script_dir:
        os.chdir(script_dir)

    download_mnist()
    download_cifar10()
    download_fashion_mnist()
    # download_adult()
    download_breast()
    download_diabetes()
    download_airline()
    download_lsst()
