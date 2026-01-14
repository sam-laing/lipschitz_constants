import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 activation='relu', seperate_biases=True, no_bias=False):
        super().__init__()
        self.seperate_biases = seperate_biases

        # Activation mapping
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

        if seperate_biases:
            # Standard PyTorch Linear layers
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        else:
            # ---------------- Bias-absorbed layers ----------------
            # Layer 1: input_dim -> hidden_dim
            fan_in = input_dim
            W1 = torch.empty(input_dim + 1, hidden_dim)
            bound1 = math.sqrt(6.0 / fan_in)  # Kaiming uniform bound
            nn.init.uniform_(W1[:-1], -bound1, bound1)  # only weights
            W1[-1].zero_()  # bias column
            self.fc1 = nn.Parameter(W1)

            # Layer 2: hidden_dim -> output_dim
            fan_in = hidden_dim
            W2 = torch.empty(hidden_dim + 1, output_dim)
            bound2 = math.sqrt(6.0 / fan_in)
            nn.init.uniform_(W2[:-1], -bound2, bound2)
            W2[-1].zero_()
            self.fc2 = nn.Parameter(W2)

        if no_bias:
            #just ignore above and init two matrix multiplications without bias
            assert seperate_biases, "no_bias option only works with seperate_biases=True"
            
        



    def forward(self, x):
        if self.seperate_biases:
            return self.fc2(self.activation(self.fc1(x)))
        else:
            if x.dim() == 1:
                x = x.unsqueeze(0)

            ones = torch.ones(x.size(0), 1, device=x.device)

            # Layer 1
            x_aug = torch.cat([x, ones], dim=1)
            h = self.activation(x_aug @ self.fc1)

            # Layer 2
            h_aug = torch.cat([h, ones], dim=1)
            out = h_aug @ self.fc2

            return out.squeeze(0) if out.size(0) == 1 else out
