import torch
import torch.nn as nn
import math


class OrthogonalInitializer:
    @staticmethod
    def create_weight(shape, init_type='normal', activation='relu', rank=None):
        """
        shape: (in_features, out_features)
        returns tensor of shape (in_features, out_features)
        """
        in_features, out_features = shape
        gain = nn.init.calculate_gain(activation)

        if init_type == 'normal':
            weight = torch.empty(in_features, out_features)
            nn.init.kaiming_normal_(weight, nonlinearity=activation)
            return weight

        elif init_type == 'full_orthogonal':
            # Orthogonal matrix
            W = torch.empty(out_features, in_features)
            nn.init.orthogonal_(W)

            #scale to (hopefully) preserve variance 
            scale = gain / math.sqrt(in_features)
            with torch.no_grad():
                W.mul_(scale)

            return W.T

        elif init_type == "better_conditioned":
            # really just want to have normal type of init (kaiming)
            pass

        else:
            raise ValueError(f"Unsupported init_type: {init_type}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation='relu',
        separate_biases=True,
        use_bias=True,
        weight_init='normal',
        ortho_rank=None,
    ):
        super().__init__()

        self.weight_init = weight_init
        self.ortho_rank = ortho_rank

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'linear': nn.Identity(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=use_bias)

        self._init_layers(activation)

    def _init_layers(self, activation):
        if self.weight_init == 'normal':
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity=activation)
        else:
            W1 = OrthogonalInitializer.create_weight(
                shape=(self.fc1.in_features, self.fc1.out_features),
                init_type=self.weight_init,
                activation=activation,
                rank=self.ortho_rank,
            )
            self.fc1.weight.data.copy_(W1.T)

        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        if self.weight_init == 'normal':
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
        else:
            W2 = OrthogonalInitializer.create_weight(
                shape=(self.fc2.in_features, self.fc2.out_features),
                init_type=self.weight_init,
                activation='linear',
                rank=self.ortho_rank,
            )
            self.fc2.weight.data.copy_(W2.T)

        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
