import torch
import torch.nn as nn


#########################################################################################
# Linear regression model
#########################################################################################
class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single output for regression

    def forward(self, x):
        return self.linear(x)

# Example usage:
# model = LinearRegressor(input_dim=10)  # Adjust input_dim as necessary



#########################################################################################
# Multilayer Perceptron (MLP) model
#########################################################################################
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Example usage:
# model = MLP(input_dim=10, hidden_dim=64, output_dim=1)  # Adjust dimensions as necessary



#########################################################################################
# Convolutional Neural Network (CNN) model
#########################################################################################
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Assuming input size is 32x32
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
# model = SimpleCNN(input_channels=3, num_classes=10)  # For example, 3 channels (RGB) and 10 classes (e.g., CIFAR-10)



#########################################################################################
# Transformer model
#########################################################################################
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, hidden_dim))  # Positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Example usage:
# model = TransformerModel(input_dim=10, num_heads=2, num_layers=2, hidden_dim=64, output_dim=1)



#########################################################################################
# Long Short-Term Memory (LSTM) model
#########################################################################################
class MultiLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MultiLayerLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Example usage:
# model = MultiLayerLSTM(input_dim=10, hidden_dim=64, num_layers=2, output_dim=1)  # Adjust dimensions as necessary








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
