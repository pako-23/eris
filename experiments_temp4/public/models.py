import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import opacus # type: ignore
from torch.utils.data import DataLoader
from transformers import get_scheduler # type: ignore
from tqdm import tqdm
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import config as cfg
import warnings
warnings.filterwarnings(
    "ignore",
    message="Secure RNG turned off.*",
    module="opacus.privacy_engine"
)
# Suppress the "Optimal order is the largest alpha" warning
warnings.filterwarnings(
    "ignore",
    message="Optimal order is the largest alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
)
warnings.filterwarnings(
    "ignore",
    message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated",
    module="torch.nn.modules.module"
)


#############################################################################################################
# Classifier models 
#############################################################################################################

# 1) LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self, model_args):
        """
        Initializes the LeNet-5 model.
        
        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes in the dataset.
            input_size (tuple): Size of the input images.       
        """
        super(LeNet5, self).__init__()
        
        in_channels = model_args["in_channels"]
        num_classes = model_args["num_classes"]
        input_size = model_args["input_size"]
        
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)  # Convolutional layer with 6 feature maps of size 5x5
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 6 feature maps of size 2x2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # Convolutional layer with 16 feature maps of size 5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 16 feature maps of size 2x2
        
        # Dinamically calculate the size of the features after convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
        self.feature_size = prod(dummy_output.size()[1:])

        self.fc1 = nn.Linear(self.feature_size, 120)  # Fully connected layer, output size 120
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer, output size 84
        self.fc3 = nn.Linear(84, num_classes)  # Fully connected layer, output size num_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU after conv1
        x = self.pool1(x)  # Apply subsampling pool1
        x = F.relu(self.conv2(x))  # Apply ReLU after conv2
        x = self.pool2(x)  # Apply subsampling pool2
        x_l = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x_l))  # Apply ReLU after fc1
        x_rec = F.relu(self.fc2(x))  # Apply ReLU after fc2
        logits = self.fc3(x_rec)    # Output: num_classes
        return logits


# class LeNet5(nn.Module):
#     def __init__(self, model_args):
#         """
#         Initializes the LeNet-5 model with a scaling factor for the network size.

#         Args:
#             model_args (dict): A dictionary containing:
#                 - in_channels (int): Number of input channels.
#                 - num_classes (int): Number of classes in the dataset.
#                 - input_size (tuple): Size of the input images (H, W).
#                 - scale_factor (float): Factor by which to scale the network parameters.
#         """
#         super(LeNet5, self).__init__()
        
#         in_channels = model_args["in_channels"]
#         num_classes = model_args["num_classes"]
#         input_size = model_args["input_size"]
#         scale_factor = 0.5

#         # Convert original layer sizes to scaled versions
#         # Round or cast to int to avoid fractional channel sizes
#         conv1_out = int(6 * scale_factor)
#         conv2_out = int(16 * scale_factor)
#         fc1_out = int(120 * scale_factor)
#         fc2_out = int(84 * scale_factor)

#         self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=5, stride=1, padding=2)
#         self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5, stride=1)
#         self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

#         # Dynamically calculate the size of the features after the convolutional layers
#         dummy_input = torch.zeros(1, in_channels, *input_size)
#         dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
#         feature_size = prod(dummy_output.size()[1:])

#         self.fc1 = nn.Linear(feature_size, fc1_out)
#         self.fc2 = nn.Linear(fc1_out, fc2_out)
#         self.fc3 = nn.Linear(fc2_out, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         logits = self.fc3(x)
#         return logits


# 2) Resnet-9 model 
def residual_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        """
        Initializes the ResNet-9 model.
        
        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes in the dataset.
            input_size (tuple): Size of the input images.
        """
        
        in_channels = model_args["in_channels"]
        num_classes = model_args["num_classes"]
        input_size = model_args["input_size"]
        scaling = 2
        
        self.prep = residual_block(in_channels, 64//scaling)
        self.layer1_head = residual_block(64//scaling, 128//scaling, pool=True)
        self.layer1_residual = nn.Sequential(residual_block(128//scaling, 128//scaling), residual_block(128//scaling, 128//scaling))
        self.layer2 = residual_block(128//scaling, 256//scaling, pool=True)
        self.layer3_head = residual_block(256//scaling, 512//scaling, pool=True)
        self.layer3_residual = nn.Sequential(residual_block(512//scaling, 512//scaling), residual_block(512//scaling, 512//scaling))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Changed to adaptive average pooling:         self.MaxPool2d = nn.Sequential(nn.MaxPool2d(4))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the features after the convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool(self.layer3_head(self.layer2(self.layer1_head(self.prep(dummy_input)))))
        self.feature_size = dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)

        # Output layer
        self.linear = nn.Linear(self.feature_size, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.pool(x)  # Changed to adaptive average pooling
        x_rec = x.view(x.size(0), -1)
        x = self.linear(x_rec)
        return x
    


# 3) MLP model  
class MLP(nn.Module):
    def __init__(self, model_args):
        """
        Initializes the MLP model.

        Args:
            input_size
            num_classes
            hidden_dim
        """  

        hidden_dim = model_args["hidden_dim"]
        num_classes = model_args["num_classes"]
        input_size = model_args["input_size"]

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        # Consider adding batch normalization and dropout if needed
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        # x = self.bn1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x_rec = self.relu2(x)
        x = self.fc3(x_rec)
        return x



# 4 Transformer
class TransformerModelFlexible(nn.Module):
    def __init__(self, model_args):
        """
        Initializes a flexible Transformer model for classification or regression tasks.

        Args:
            dataset_name (str): The name of the dataset, used to configure the model.
            num_heads (int): The number of attention heads in the Transformer model.
            num_layers (int): The number of Transformer layers.
            hidden_dim (int): The hidden dimension of the Transformer model.
        """
        super(TransformerModelFlexible, self).__init__()
        
        num_classes = model_args["num_classes"]
        input_size = model_args["input_size"]
        sequence_length = model_args["sequence_length"]
        num_heads = model_args["num_heads"]
        num_layers = model_args["num_layers"]
        hidden_dim = model_args["hidden_dim"]

        # Transformer Components
        self.embedding = nn.Linear(input_size, hidden_dim)  # Input_dim=1 -> hidden_dim=64
        self.pos_encoder = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))  # Positional encoding for sequence length 30
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)  

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, hidden_dim)
        x = x + self.pos_encoder[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer(x)  # Shape: (batch_size, sequence_length, hidden_dim)
        x = x.mean(dim=1)  # Global average pooling to get (batch_size, hidden_dim)
        x = self.fc(x)  # Shape: (batch_size, output_dim) for classification or (batch_size, 1) for regression
        
        return x





#########################################################################################
# Linear regression model
#########################################################################################
class LinearModel(nn.Module):
    def __init__(self, model_args):
        """
        Initializes the LinearModel.

        Args:
            input_size
            num_classes
        """
        super(LinearModel, self).__init__()
        
        input_size = model_args["input_size"] 
        num_classes = model_args["num_classes"]
        
        self.output_layer = nn.Linear(input_size, num_classes) 

    def forward(self, x):
        x = self.output_layer(x.squeeze(-1))
        
        return x




#############################################################################################################
# Helper functions 
#############################################################################################################
## Predictor 
# simple train function
def simple_train(model, device, train_loader, optimizer, criterion, epoch, client_id=None):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 10 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
        #           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        loss_list.append(loss.item())
    # print(f'Client: {client_id} - Train Epoch: {epoch} \tLoss: {sum(loss_list)/len(loss_list):.6f}')


def train_with_opacus(model, device, train_loader, optimizer, criterion, sigma, epochs=1, client_id=None): 
    model.train()

    privacy_engine = opacus.privacy_engine.PrivacyEngine(accountant='rdp', secure_mode=False,)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=cfg.sensitivity,
        )

    # 4) Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            
            # Opacus will do the clipping + noise under the hood
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # 5) Check the privacy budget (epsilon) at the end of each epoch
        cur_epsilon = privacy_engine.get_epsilon(delta=cfg.delta)
        # print(f"Client {client_id} - Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | ε = {cur_epsilon:.2f} (δ={cfg.delta})")

    print(f"Client {client_id} - Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | ε = {cur_epsilon:.2f} (δ={cfg.delta})")

    del privacy_engine
    

# def train_llm_with_opacus(model, device, train_loader, optimizer, scheduler, privacy_engine, training_args, sigma, client_id=None): 
def train_llm_with_opacus(model, device, train_data, training_args, sigma, delta, client_id=None): 
    # Create the dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * training_args.num_train_epochs,
    )
        
    # Freeze the word and position embeddings - few params (cannot be optimized with opacus)
    model.train()
    model.distilbert.embeddings.position_embeddings.weight.requires_grad = False

    # Opacus privacy engine
    privacy_engine = opacus.PrivacyEngine(accountant='rdp', secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private(
                        module=model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        noise_multiplier=sigma,
                        max_grad_norm=cfg.sensitivity,
                        poisson_sampling=False,
                    )

    # 4) Training loop
    training_loss_history = []
    for epoch in range(training_args.num_train_epochs):
        # print(f"Epoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            bar_format="{percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} - {postfix} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        for step, batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # (Optional) Gradient clipping used in Trainer (already in PrivacyEngine)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            training_loss_history.append(loss.item())

            # Update the postfix inline every 100 steps
            if (step + 1) % 100 == 0:
                avg_loss = np.mean(training_loss_history[-100:])
                progress_bar.set_postfix_str(f"Epoch {epoch+1} - Step {step+1} - Loss: {avg_loss:.4f}")
    
    cur_epsilon = privacy_engine.get_epsilon(delta=delta)
    print(f"Client {client_id} - ε = {cur_epsilon:.2f} (δ={delta})")
    
    # Set the position embeddings back to trainable
    model.distilbert.embeddings.position_embeddings.weight.requires_grad = True
 
    del privacy_engine


# simple test function
def simple_test(model, device, test_loader, criterion, client_id=None):
    model.eval()
    test_loss = 0
    correct = 0
    y_true_all, y_pred_all, y_pred_all_digits = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            y_true_all.extend(target.cpu().numpy())
            y_pred_all.extend(pred.cpu().numpy())
            y_pred_all_digits.extend(output.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    
    if criterion == F.cross_entropy:
        accuracy = correct / len(test_loader.dataset)
        f1_score_trad = f1_score(y_true_all, y_pred_all, average='weighted') # Calculate metrics for each label, and find their average weighted by support. NOT traditional F1-score
        print(f'Validation set (Client {client_id}): Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
            f'({100. * correct / len(test_loader.dataset):.0f}%)')
        return test_loss, accuracy, f1_score_trad 

    else:
        mae = mean_absolute_error(y_true_all, y_pred_all_digits)
        mse = mean_squared_error(y_true_all, y_pred_all_digits)
        print(f'Validation set (Client {client_id}): Average loss: {test_loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}')
        return test_loss, mae, mse






#############################################################################################################
# Config
#############################################################################################################
model_dict = {
    "mnist": LeNet5,
    "cifar10": ResNet9,
    "fmnist":LeNet5,
    "breast": MLP,  
    "diabetes": MLP,
    "adult": MLP,
    "airline":LinearModel,
    "lsst": TransformerModelFlexible,
}

model_args = {
    "mnist": 
        {
            "in_channels": 1, 
            "num_classes": 10, 
            "input_size": (28, 28),
        },
    "cifar10": 
        {
            "in_channels": 3, 
            "num_classes": 10, 
            "input_size": (32, 32),    
        },
    "fmnist":
        {
            "in_channels": 1, 
            "num_classes": 10, 
            "input_size": (28, 28),   
        },
    "breast":
        {
            "input_size": 30, 
            "num_classes": 2, 
            "hidden_dim": 128
        },
    "diabetes":
        {
            "input_size": 21, 
            "num_classes": 2, 
            "hidden_dim": 128
        },
    "adult":
        {
            "input_size": 105, 
            "num_classes": 2, 
            "hidden_dim": 128
        },
    "airline":
        {
            "input_size": 30, 
            "num_classes": 1, 
        },
    "lsst":
        {
            "input_size": 6,
            "sequence_length": 36,
            "num_classes": 12,
            "num_heads": 2,
            "num_layers": 2,
            "hidden_dim": 64,
        }
}






# OLD VERSION

# config = {
#     'mnist': {
#         'task_type': 'classification',
#         'num_classes': 10,
#         'LinearModel': {
#             'input_dim': 28 * 28
#         },
#         'MLP': {
#             'input_dim': 28 * 28,
#         },
#         'CNN': {
#             'input_channels': 1
#         },
#         'LeNet5Flexible': {
#             'input_channels': 1
#         },
#         'ResNet9': {
#             'input_channels': 1
#         },
#         'TransformerModelFlexible': {
#             'input_dim': 1,
#             'sequence_length': 28 * 28 
#         },
#         'MultiLayerLSTM': {
#             'input_dim': 1,
#             'sequence_length': 28 * 28 
#         }
#     },
#     'cifar10': {
#         'task_type': 'classification',
#         'num_classes': 10,
#         'LinearModel': {
#             'input_dim': 32 * 32 * 3
#         },
#         'MLP': {
#             'input_dim': 32 * 32 * 3,
#         },
#         'CNN': {
#             'input_channels': 3
#         },
#         'LeNet5Flexible': {
#             'input_channels': 3
#         },
#         'ResNet9': {
#             'input_channels': 3
#         },
#         'TransformerModelFlexible': {
#             'input_dim': 1,
#             'sequence_length': 32 * 32 * 3,  
#         },
#         'MultiLayerLSTM': {
#             'input_dim': 1,
#             'sequence_length': 32 * 32 * 3
#         }
#     },
#     'airline': {
#         'task_type': 'regression',
#         'num_classes': None,
#         'LinearModel': {
#             'input_dim': 30
#         },
#         'MLP': {
#             'input_dim': 30,
#         },
#         'CNN': {
#             'input_channels': 1
#         },
#         'LeNet5Flexible': {
#             'input_channels': 1
#         },
#         'ResNet9': {
#             'input_channels': 1
#         },
#         'TransformerModelFlexible': {  
#            'input_dim': 1,
#            'sequence_length': 30
#         },
#         'MultiLayerLSTM': {
#             'input_dim': 1,
#             'sequence_length': 30
#         }
#     },
#     'adult': {
#         'task_type': 'classification',
#         'num_classes': 2,
#         'LinearModel': {
#             'input_dim': 105
#         },
#         'MLP': {
#             'input_dim': 105,
#         },
#         'CNN': {
#             'input_channels': 1
#         },
#         'LeNet5Flexible': {
#             'input_channels': 1
#         },
#         'ResNet9': {
#             'input_channels': 1
#         },
#         'TransformerModelFlexible': {
#             'input_dim': 1, 
#             'sequence_length': 105
#         },
#         'MultiLayerLSTM': {
#             'input_dim': 1,
#             'sequence_length': 105
#         }    
#     },
#     'LSST': {
#         'task_type': 'classification',
#         'num_classes': 12,
#         'LinearModel': {
#             'input_dim': 36 * 6
#         },
#         'MLP': {
#             'input_dim': 36 * 6,
#         },
#         'CNN': {
#             'input_channels': 1
#         },
#         'LeNet5Flexible': {
#             'input_channels': 1
#         },
#         'ResNet9': {
#             'input_channels': 1
#         },
#         'TransformerModelFlexible': {
#             'input_dim': 6,
#             'sequence_length': 36
#         },
#         'MultiLayerLSTM': {
#             'input_dim': 6,
#             'sequence_length': 36
#         }
#     }
# }

# # define device
# def check_gpu(manual_seed=True, print_info=True):
#     if manual_seed:
#         torch.manual_seed(0)
#     if torch.cuda.is_available():
#         if print_info:
#             print("CUDA is available")
#         device = 'cuda:1'
#         torch.cuda.manual_seed_all(0) 
#     elif torch.backends.mps.is_available():
#         if print_info:
#             print("MPS is available")
#         device = torch.device("mps")
#         torch.mps.manual_seed(0)
#     else:
#         if print_info:
#             print("CUDA is not available")
#         device = 'cpu'
#     return device


# #########################################################################################
# # Linear regression model
# #########################################################################################
# class LinearModel(nn.Module):
#     def __init__(self, dataset_name):
#         """
#         Initializes the LinearModel.

#         Args:
#             dataset_name (str): The name of the dataset, used to configure the model.
#         """
#         super(LinearModel, self).__init__()
        
#         self.task_type = config[dataset_name]['task_type']
#         num_classes = config[dataset_name]['num_classes']
#         input_dim = config[dataset_name]['LinearModel']['input_dim']

#         self.criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
                
#         if self.task_type == 'regression':
#             self.output_layer = nn.Linear(input_dim, 1)  # Single output for regression
#         elif self.task_type == 'classification':
#             if num_classes is None:
#                 raise ValueError("num_classes must be specified for classification tasks.")
#             self.output_layer = nn.Linear(input_dim, num_classes)  # Output layer for classification
#         else:
#             raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")

#     def forward(self, x):
#         x = self.output_layer(x)
        
#         return x

# # Example usage:
# # model = LinearModel(input_dim=30, task_type='regression') # For regression tasks (use MSE loss)
# # model = LinearModel(input_dim=30, task_type='classification', num_classes=1) # For binary classification tasks (use BCE loss)
# # model = LinearModel(input_dim=30, task_type='classification', num_classes=5) # For multiclass classification tasks (use CrossEntropy loss)




# #########################################################################################
# # Multilayer Perceptron (MLP) model
# #########################################################################################
# class MLP(nn.Module):
#     def __init__(self, dataset_name, hidden_dim=128):
#         """
#         Initializes the MLP model.

#         Args:
#             dataset_name (str): The name of the dataset, used to configure the model.
#             hidden_dim (int): The number of hidden units in the MLP.
#         """
#         super(MLP, self).__init__()
        
#         self.task_type = config[dataset_name]['task_type']
#         output_dim = config[dataset_name]['num_classes']
#         input_dim = config[dataset_name]['MLP']['input_dim']
        
#         self.criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
#         if self.task_type == 'regression':
#             self.fc3 = nn.Linear(hidden_dim, 1)  # Single output for regression
#         elif self.task_type == 'classification':
#             if output_dim is None:
#                 raise ValueError("output_dim must be specified for classification tasks.")
#             self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer for classification
#         else:
#             raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
        
#         return x

# # Example usage:
# # model = MLP(input_dim=30, hidden_dim=128, task_type='regression') # For regression tasks (use MSE loss)
# # model = MLP(input_dim=30, hidden_dim=128, task_type='classification', output_dim=1) # For binary classification tasks (use BCE loss)
# # model = MLP(input_dim=30, hidden_dim=128, task_type='classification', output_dim=5) # For multiclass classification tasks (use CrossEntropy loss)


# #########################################################################################
# # Convolutional Neural Network (CNN) model
# #########################################################################################
# class CNN(nn.Module):
#     def __init__(self, dataset_name):
#         """
#         Initializes the CNN model for classification tasks.

#         Args:
#             dataset_name (str): The name of the dataset, used to configure the
#         """
#         super(CNN, self).__init__()
        
#         self.task_type = config[dataset_name]['task_type']
#         num_classes = config[dataset_name]['num_classes']
#         input_channels = config[dataset_name]['CNN']['input_channels']
        
#         self.criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
                
#         self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
#         # The fully connected layer will be defined dynamically later based on the input image size
#         self.fc1 = None
#         self.fc2 = None
#         self.num_classes = num_classes

#     def forward(self, x):
#         x = self.pool(nn.ReLU()(self.conv1(x)))
#         x = self.pool(nn.ReLU()(self.conv2(x)))

#         # Flatten the feature map
#         x = x.view(x.size(0), -1)

#         # Initialize fully connected layers if not done already
#         if self.fc1 is None:
#             num_features = x.size(1)
#             self.fc1 = nn.Linear(num_features, 128)
#             if self.task_type == 'regression':
#                 self.fc2 = nn.Linear(128, 1)
#             elif self.task_type == 'classification':
#                 if self.num_classes is None:
#                     raise ValueError("num_classes must be specified for classification tasks.")
#                 self.fc2 = nn.Linear(128, self.num_classes)
#             else:
#                 raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")
#             # Move to the same device as the input
#             self.fc1.to(x.device)
#             self.fc2.to(x.device)
        
#         x = nn.ReLU()(self.fc1(x))
#         x = self.fc2(x)
        
#         return x

# # Example usage:
# # model = CNN(input_channels=3, num_classes=10)  # For example, 3 channels (RGB) and 10 classes (e.g., CIFAR-10)


# # LENET 5
# class LeNet5Flexible(nn.Module):
#     def __init__(self, dataset_name):
#         """
#         Initializes a flexible CNN model inspired by LeNet-5 for classification tasks.

#         Args:
#             dataset_name (str): The name of the dataset, used to configure the model.
#         """
#         super(LeNet5Flexible, self).__init__()
        
#         self.task_type = config[dataset_name]['task_type']
#         num_classes = config[dataset_name]['num_classes']
#         input_channels = config[dataset_name]['LeNet5Flexible']['input_channels']
        
#         self.criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
        
#         # LeNet-5 Inspired Convolutional Layers
#         self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=2)  # LeNet-5 uses 6 filters
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # LeNet-5 uses 16 filters
        
#         # MaxPooling Layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Fully connected layers (will be dynamically initialized based on the input image size)
#         self.fc1 = None
#         self.fc2 = None
#         self.fc3 = None
#         self.num_classes = num_classes

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # First Conv + Pooling
#         x = self.pool(F.relu(self.conv2(x)))  # Second Conv + Pooling

#         # Flatten the feature map
#         x = x.view(x.size(0), -1)
        
#         # Dynamically initialize fully connected layers
#         if self.fc1 is None:
#             num_features = x.size(1)
#             self.fc1 = nn.Linear(num_features, 120)  # LeNet-5 uses 120 neurons
#             self.fc2 = nn.Linear(120, 84)  # LeNet-5 uses 84 neurons
#             if self.task_type == 'regression':
#                 self.fc3 = nn.Linear(84, 1)
#             elif self.task_type == 'classification':
#                 if self.num_classes is None:
#                     raise ValueError("num_classes must be specified for classification tasks.")
#                 self.fc3 = nn.Linear(84, self.num_classes)
#             else:
#                 raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")
            
#             # Move fully connected layers to the same device as input
#             self.fc1.to(x.device)
#             self.fc2.to(x.device)
#             self.fc3.to(x.device)

#         x = F.relu(self.fc1(x))  # First Fully Connected Layer
#         x = F.relu(self.fc2(x))  # Second Fully Connected Layer
#         x = self.fc3(x)  # Output Layer
        
#         return x


# # RESNET9
# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
    
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class ResNet9(nn.Module):
#     def __init__(self, dataset_name):
#         super(ResNet9, self).__init__()
        
#         self.task_type = config[dataset_name]['task_type']
#         num_classes = config[dataset_name]['num_classes']
#         input_channels = config[dataset_name]['ResNet9']['input_channels']
        
#         self.criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
        
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
        
#         self.layer1 = BasicBlock(64, 128, stride=2)
#         self.layer2 = BasicBlock(128, 256, stride=2)
#         self.layer3 = BasicBlock(256, 512, stride=2)
        
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         if self.task_type == 'regression':
#             self.fc = nn.Linear(512, 1)
#         elif self.task_type == 'classification':
#             if num_classes is None:
#                 raise ValueError("num_classes must be specified for classification tasks.")
#             self.fc = nn.Linear(512, num_classes)
#         else:
#             raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")
    
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
        
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
        
#         return x
    


# #########################################################################################
# # Transformer model
# #########################################################################################
# class TransformerModelFlexible(nn.Module):
#     def __init__(self, dataset_name, num_heads=2, num_layers=2, hidden_dim=64):
#         """
#         Initializes a flexible Transformer model for classification or regression tasks.

#         Args:
#             dataset_name (str): The name of the dataset, used to configure the model.
#             num_heads (int): The number of attention heads in the Transformer model.
#             num_layers (int): The number of Transformer layers.
#             hidden_dim (int): The hidden dimension of the Transformer model.
#         """
#         super(TransformerModelFlexible, self).__init__()

#         self.task_type = config[dataset_name]['task_type']
#         output_dim = config[dataset_name]['num_classes']
#         input_dim = config[dataset_name]['TransformerModelFlexible']['input_dim']
#         self.sequence_length = config[dataset_name]['TransformerModelFlexible']['sequence_length']

#         self.criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()

#         # Transformer Components
#         self.embedding = nn.Linear(input_dim, hidden_dim)  # Input_dim=1 -> hidden_dim=64
#         self.pos_encoder = nn.Parameter(torch.zeros(1, self.sequence_length, hidden_dim))  # Positional encoding for sequence length 30
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         if self.task_type == 'regression':
#             self.fc = nn.Linear(hidden_dim, 1)  # Single output for regression
#         elif self.task_type == 'classification':
#             if output_dim is None:
#                 raise ValueError("output_dim must be specified for classification tasks.")
#             self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer for classification
#         else:
#             raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")

#     def forward(self, x):
#         x = self.embedding(x)  # Shape: (batch_size, sequence_length, hidden_dim)
#         x = x + self.pos_encoder[:, :x.size(1), :]  # Add positional encoding
#         x = self.transformer(x)  # Shape: (batch_size, sequence_length, hidden_dim)
#         x = x.mean(dim=1)  # Global average pooling to get (batch_size, hidden_dim)
#         x = self.fc(x)  # Shape: (batch_size, output_dim) for classification or (batch_size, 1) for regression
        
#         return x


# # Example usage:
# # model = TransformerModelFlexible(dataset_name=dataset, num_heads=2, num_layers=2, hidden_dim=64).to(DEVICE)




# #########################################################################################
# # Long Short-Term Memory (LSTM) model
# #########################################################################################
# class MultiLayerLSTM(nn.Module):
#     def __init__(self, dataset_name, hidden_dim=64, num_layers=2):
#         super(MultiLayerLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         self.task_type = config[dataset_name]['task_type']
#         output_dim = config[dataset_name]['num_classes']
#         input_dim = config[dataset_name]['MultiLayerLSTM']['input_dim']

#         self.criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()

#         # LSTM Layer
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

#         if self.task_type == 'regression':
#             self.fc = nn.Linear(hidden_dim, 1)  # Single output for regression
#         elif self.task_type == 'classification':
#             if output_dim is None:
#                 raise ValueError("output_dim must be specified for classification tasks.")
#             self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer for classification
#         else:
#             raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")

#     def forward(self, x):
#         # Set initial hidden and cell states
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

#         # Forward propagate LSTM
#         out, _ = self.lstm(x, (h0, c0))

#         # Decode the hidden state of the last time step
#         out = self.fc(out[:, -1, :])
#         return out




# #########################################################################################
# # Scalable networks
# #########################################################################################
# class ScalableMLP(nn.Module):
#     def __init__(self, dataset_name, hidden_dim=64, num_layers=2):
#         super(ScalableMLP, self).__init__()
        
#         output_dim = config[dataset_name]['num_classes']
#         input_dim = config[dataset_name]['MLP']['input_dim']
#         self.criterion = nn.CrossEntropyLoss()
#         self.task_type = 'classification'
        
#         layers = []
#         layers.append(nn.Linear(input_dim, hidden_dim))
#         layers.append(nn.ReLU())
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#         layers.append(nn.Linear(hidden_dim, output_dim))
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)

# # Example usage:
# # Start small and scale up
# # model_small = ScalableMLP(input_dim=10, hidden_dim=32, num_layers=2, output_dim=1)  # A small MLP
# # model_large = ScalableMLP(input_dim=10, hidden_dim=256, num_layers=10, output_dim=1)  # A larger MLP

# class ScalableCNN(nn.Module):
#     def __init__(self, dataset_name, base_filters=16, num_conv_layers=2):
#         super(ScalableCNN, self).__init__()
        
#         self.num_classes = config[dataset_name]['num_classes']    
#         in_channels = config[dataset_name]['CNN']['input_channels']
#         self.criterion = nn.CrossEntropyLoss()
#         self.task_type = 'classification'
        
#         layers = []
#         for i in range(num_conv_layers):
#             out_channels = base_filters * (2 ** i)
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(nn.ReLU())
#             layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#             in_channels = out_channels

#         self.conv_layers = nn.Sequential(*layers)
#         # self.fc1 = nn.Linear(out_channels * 8 * 8, 128)
#         # self.fc2 = nn.Linear(128, num_classes)
#         self.fc1 = None
#         self.fc2 = None

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         if self.fc1 is None:
#             self.fc1 = nn.Linear(x.size(1), 128)
#             self.fc2 = nn.Linear(128, self.num_classes)
#             self.fc1.to(x.device)
#             self.fc2.to(x.device)
#         x = nn.ReLU()(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Example usage:
# # Start small and scale up
# # model_small = ScalableCNN(input_channels=3, num_classes=10, base_filters=8, num_conv_layers=2)  # A small CNN
# # model_large = ScalableCNN(input_channels=3, num_classes=10, base_filters=64, num_conv_layers=8)  # A larger CNN
