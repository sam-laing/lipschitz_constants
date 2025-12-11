import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', seperate_biases=True):
        super(MLP, self).__init__()
        self.seperate_biases = seperate_biases

        if seperate_biases:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        else:
            # fc1: (input_dim + 1, hid2en_dim)
            # The key insight: effective fan_in is input_dim+1 due to augmented input
            fc1_full = torch.empty(input_dim + 1, hidden_dim)
            gain = nn.init.calculate_gain(activation)
            # Use input_dim+1 as effective fan_in for proper variance
            std1 = gain * math.sqrt(1.0 / (input_dim + 1))
            nn.init.normal_(fc1_full, mean=0.0, std=std1)
            self.fc1 = nn.Parameter(fc1_full)

            # fc2: (hidden_dim + 1, output_dim)
            fc2_full = torch.empty(hidden_dim + 1, output_dim)
            std2 = gain * math.sqrt(1.0 / (hidden_dim + 1))
            nn.init.normal_(fc2_full, mean=0.0, std=std2)
            self.fc2 = nn.Parameter(fc2_full)

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

    def forward(self, x):
        if self.seperate_biases:
            return self.fc2(self.activation(self.fc1(x)))
        else:
            if x.dim() == 1:
                x = x.unsqueeze(0)

            batch_size = x.shape[0]

            # Augment input with ones for bias
            x_aug = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
            h = self.activation(x_aug @ self.fc1)

            # Augment hidden layer with ones for bias
            h_aug = torch.cat([h, torch.ones(batch_size, 1, device=x.device)], dim=1)
            out = h_aug @ self.fc2

            if out.shape[0] == 1:
                return out.squeeze(0)
            return out