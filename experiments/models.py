import torch
import torch.nn as nn
import torch.nn.functional as F


config = {
    'mnist': {
        'task_type': 'classification',
        'num_classes': 10,
        'input_dim': 28 * 28,
        'input_channels': 1
    },
    'cifar10': {
        'task_type': 'classification',
        'num_classes': 10,
        'input_dim': 32 * 32 * 3,
        'input_channels': 3
    },
    'airline': {
        'task_type': 'regression',
        'num_classes': None,
        'input_dim': 30,
    },
    'adult': {
        'task_type': 'classification',
        'num_classes': 2,
        'input_dim': 105,
    }
}

# define device
def check_gpu(manual_seed=True, print_info=True):
    if manual_seed:
        torch.manual_seed(0)
    if torch.cuda.is_available():
        if print_info:
            print("CUDA is available")
        device = 'cuda:1'
        torch.cuda.manual_seed_all(0) 
    elif torch.backends.mps.is_available():
        if print_info:
            print("MPS is available")
        device = torch.device("mps")
        torch.mps.manual_seed(0)
    else:
        if print_info:
            print("CUDA is not available")
        device = 'cpu'
    return device


#########################################################################################
# Linear regression model
#########################################################################################
class LinearModel(nn.Module):
    def __init__(self, dataset_name):
        """
        Initializes the LinearModel.

        Args:
            input_dim (int): The number of input features.
            task_type (str): The type of task ('regression' or 'classification').
            num_classes (int, optional): The number of classes for classification. Required if task_type is 'classification'.
        """
        super(LinearModel, self).__init__()
        
        self.task_type = config[dataset_name]['task_type']
        input_dim = config[dataset_name]['input_dim']
        num_classes = config[dataset_name]['num_classes']
        
        self.criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
                
        if self.task_type == 'regression':
            self.output_layer = nn.Linear(input_dim, 1)  # Single output for regression
        elif self.task_type == 'classification':
            if num_classes is None:
                raise ValueError("num_classes must be specified for classification tasks.")
            self.output_layer = nn.Linear(input_dim, num_classes)  # Output layer for classification
        else:
            raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")

    def forward(self, x):
        x = self.output_layer(x)
        
        return x

# Example usage:
# model = LinearModel(input_dim=30, task_type='regression') # For regression tasks (use MSE loss)
# model = LinearModel(input_dim=30, task_type='classification', num_classes=1) # For binary classification tasks (use BCE loss)
# model = LinearModel(input_dim=30, task_type='classification', num_classes=5) # For multiclass classification tasks (use CrossEntropy loss)





#########################################################################################
# Multilayer Perceptron (MLP) model
#########################################################################################
class MLP(nn.Module):
    def __init__(self, dataset_name, hidden_dim=128):
        """
        Initializes the MLP model.

        Args:
            input_dim (int): The number of input features.
            hidden_dim (int): The number of neurons in the hidden layers.
            task_type (str): The type of task ('regression' or 'classification').
            output_dim (int, optional): The number of output neurons. For regression, this is typically 1.
                                        For classification, this corresponds to the number of classes.
        """
        super(MLP, self).__init__()
        
        self.task_type = config[dataset_name]['task_type']
        input_dim = config[dataset_name]['input_dim']
        output_dim = config[dataset_name]['num_classes']
        
        self.criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if self.task_type == 'regression':
            self.fc3 = nn.Linear(hidden_dim, 1)  # Single output for regression
        elif self.task_type == 'classification':
            if output_dim is None:
                raise ValueError("output_dim must be specified for classification tasks.")
            self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer for classification
        else:
            raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x

# Example usage:
# model = MLP(input_dim=30, hidden_dim=128, task_type='regression') # For regression tasks (use MSE loss)
# model = MLP(input_dim=30, hidden_dim=128, task_type='classification', output_dim=1) # For binary classification tasks (use BCE loss)
# model = MLP(input_dim=30, hidden_dim=128, task_type='classification', output_dim=5) # For multiclass classification tasks (use CrossEntropy loss)


#########################################################################################
# Convolutional Neural Network (CNN) model
#########################################################################################
class CNN(nn.Module):
    def __init__(self, dataset_name):
        """
        Initializes the CNN model for classification tasks.

        Args:
            input_channels (int): The number of input channels (e.g., 3 for RGB images).
            num_classes (int): The number of output classes for classification.
        """
        super(CNN, self).__init__()
        
        input_channels = config[dataset_name]['input_channels']
        num_classes = config[dataset_name]['num_classes']
        
        self.criterion = nn.CrossEntropyLoss()
        self.task_type = 'classification'
                
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # The fully connected layer will be defined dynamically later based on the input image size
        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Initialize fully connected layers if not done already
        if self.fc1 is None:
            num_features = x.size(1)
            self.fc1 = nn.Linear(num_features, 128)
            self.fc2 = nn.Linear(128, self.num_classes)
            # Move to the same device as the input
            self.fc1.to(x.device)
            self.fc2.to(x.device)
        
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Example usage:
# model = CNN(input_channels=3, num_classes=10)  # For example, 3 channels (RGB) and 10 classes (e.g., CIFAR-10)



class LeNet5Flexible(nn.Module):
    def __init__(self, dataset_name):
        """
        Initializes a flexible CNN model inspired by LeNet-5 for classification tasks.

        Args:
            dataset_name (str): The name of the dataset, used to configure the model.
        """
        super(LeNet5Flexible, self).__init__()
        
        input_channels = config[dataset_name]['input_channels']
        num_classes = config[dataset_name]['num_classes']
        
        self.criterion = nn.CrossEntropyLoss()
        self.task_type = 'classification'
        
        # LeNet-5 Inspired Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=2)  # LeNet-5 uses 6 filters
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # LeNet-5 uses 16 filters
        
        # MaxPooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers (will be dynamically initialized based on the input image size)
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First Conv + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second Conv + Pooling

        # Flatten the feature map
        x = x.view(x.size(0), -1)
        
        # Dynamically initialize fully connected layers
        if self.fc1 is None:
            num_features = x.size(1)
            self.fc1 = nn.Linear(num_features, 120)  # LeNet-5 uses 120 neurons
            self.fc2 = nn.Linear(120, 84)  # LeNet-5 uses 84 neurons
            self.fc3 = nn.Linear(84, self.num_classes)
            
            # Move fully connected layers to the same device as input
            self.fc1.to(x.device)
            self.fc2.to(x.device)
            self.fc3.to(x.device)

        x = F.relu(self.fc1(x))  # First Fully Connected Layer
        x = F.relu(self.fc2(x))  # Second Fully Connected Layer
        x = self.fc3(x)  # Output Layer
        
        return x



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet9(nn.Module):
    def __init__(self, dataset_name):
        super(ResNet9, self).__init__()
        
        input_channels = config[dataset_name]['input_channels']
        num_classes = config[dataset_name]['num_classes']
        
        self.criterion = nn.CrossEntropyLoss()
        self.task_type = 'classification'
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = BasicBlock(64, 128, stride=2)
        self.layer2 = BasicBlock(128, 256, stride=2)
        self.layer3 = BasicBlock(256, 512, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    


#########################################################################################
# Transformer model
#########################################################################################
# class TransformerModelFlexible(nn.Module):
#     def __init__(self, dataset_name, num_heads=2, num_layers=2, hidden_dim=64):
#         """
#         Initializes a flexible Transformer model for classification or regression tasks.

#         Args:
#             dataset_name (str): The name of the dataset, used to configure the model.
#         """
#         super(TransformerModelFlexible, self).__init__()

#         input_dim = config[dataset_name]['input_dim']
#         output_dim = config[dataset_name]['num_classes']

#         self.criterion = nn.CrossEntropyLoss() if config[dataset_name]['task_type'] == 'classification' else nn.MSELoss()
#         self.task_type = config[dataset_name]['task_type']

#         # Transformer Components
#         self.embedding = nn.Linear(input_dim, hidden_dim)
#         self.pos_encoder = nn.Parameter(torch.zeros(1, 100, hidden_dim))  # Positional encoding
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # self.fc = nn.Linear(hidden_dim, output_dim)
        
#         if self.task_type == 'regression':
#             self.fc = nn.Linear(hidden_dim, 1)  # Single output for regression
#         elif self.task_type == 'classification':
#             if output_dim is None:
#                 raise ValueError("output_dim must be specified for classification tasks.")
#             self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer for classification
#         else:
#             raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")

#     def forward(self, x):
#         x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
#         x = self.transformer(x)
#         x = x.mean(dim=1)  # Global average pooling
#         x = self.fc(x)
        
#         return x


# # transformer model with 'correct' input dimension
# class TransformerModelFlexible(nn.Module):
#     def __init__(self, dataset_name, num_heads=2, num_layers=2, hidden_dim=64):
#         super(TransformerModelFlexible, self).__init__()

#         input_dim = 1  # Since your time series has 1 feature
#         # sequence_length = 30  # Fixed sequence length for the time series data
#         output_dim = config[dataset_name]['num_classes']

#         self.criterion = nn.CrossEntropyLoss() if config[dataset_name]['task_type'] == 'classification' else nn.MSELoss()
#         self.task_type = config[dataset_name]['task_type']
#         self.sequence_length = config[dataset_name]['input_dim']

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

# Example usage:
# model = TransformerModel(input_dim=10, num_heads=2, num_layers=2, hidden_dim=64, output_dim=1)



#########################################################################################
# Long Short-Term Memory (LSTM) model
#########################################################################################
class MultiLayerLSTM(nn.Module):
    def __init__(self, dataset_name, hidden_dim=64, num_layers=1):
        super(MultiLayerLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        input_dim = 1  # Since your time series has 1 feature
        output_dim = config[dataset_name]['num_classes']

        self.criterion = nn.CrossEntropyLoss() if config[dataset_name]['task_type'] == 'classification' else nn.MSELoss()
        self.task_type = config[dataset_name]['task_type']

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        if self.task_type == 'regression':
            self.fc = nn.Linear(hidden_dim, 1)  # Single output for regression
        elif self.task_type == 'classification':
            if output_dim is None:
                raise ValueError("output_dim must be specified for classification tasks.")
            self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer for classification
        else:
            raise ValueError("Invalid task_type. Choose either 'regression' or 'classification'.")

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out




#########################################################################################
# Scalable networks
#########################################################################################
class ScalableMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ScalableMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Example usage:
# Start small and scale up
# model_small = ScalableMLP(input_dim=10, hidden_dim=32, num_layers=2, output_dim=1)  # A small MLP
# model_large = ScalableMLP(input_dim=10, hidden_dim=256, num_layers=10, output_dim=1)  # A larger MLP

class ScalableCNN(nn.Module):
    def __init__(self, input_channels, num_classes, base_filters=16, num_conv_layers=2):
        super(ScalableCNN, self).__init__()
        layers = []
        in_channels = input_channels
        for i in range(num_conv_layers):
            out_channels = base_filters * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.fc1 = nn.Linear(out_channels * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
# Start small and scale up
# model_small = ScalableCNN(input_channels=3, num_classes=10, base_filters=8, num_conv_layers=2)  # A small CNN
# model_large = ScalableCNN(input_channels=3, num_classes=10, base_filters=64, num_conv_layers=8)  # A larger CNN
