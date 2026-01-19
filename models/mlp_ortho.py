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

        # full orthogonal (column-orthogonal + Kaiming scaling) 
        elif init_type == 'full_orthogonal':
            # nn.init.orthogonal_ works on (out, in)
            W = torch.empty(out_features, in_features)
            nn.init.orthogonal_(W)

            # fan-in scaling
            scale = gain / math.sqrt(in_features)
            with torch.no_grad():
                W.mul_(scale)

            return W.T

        # factorized orthogonal UV^T with specified rank
        elif init_type == 'factorized_orthogonal':
            if rank is None:
                rank = min(in_features, out_features)
            rank = min(rank, in_features, out_features)

            U = torch.empty(out_features, rank)
            V = torch.empty(in_features, rank)

            nn.init.orthogonal_(U)
            nn.init.orthogonal_(V)

            weight = U @ V.T

            #fan-in scaling
            scale = gain / math.sqrt(in_features)
            with torch.no_grad():
                weight.mul_(scale)

            return weight.T

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

        self.separate_biases = separate_biases
        self.use_bias = use_bias
        self.weight_init = weight_init
        self.ortho_rank = ortho_rank

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

        if separate_biases:
            self.fc1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)
            self.fc2 = nn.Linear(hidden_dim, output_dim, bias=use_bias)
            self._init_separate_layers(activation)
        else:
            fc1_in = input_dim + 1 if use_bias else input_dim
            fc2_in = hidden_dim + 1 if use_bias else hidden_dim

            self.fc1 = nn.Parameter(
                OrthogonalInitializer.create_weight(
                    shape=(fc1_in, hidden_dim),
                    init_type=weight_init,
                    activation=activation,
                    rank=ortho_rank,
                )
            )

            self.fc2 = nn.Parameter(
                OrthogonalInitializer.create_weight(
                    shape=(fc2_in, output_dim),
                    init_type=weight_init,
                    activation='linear',  # output layer
                    rank=ortho_rank,
                )
            )

    def _init_separate_layers(self, activation):
        # first layer
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
            nn.init.constant_(self.fc1.bias, 0.01)  #ReLU friendly init

        # second output layer
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
        if self.separate_biases:
            return self.fc2(self.activation(self.fc1(x)))

        # manual bias path (unchanged logic)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if self.use_bias:
            ones = torch.ones(x.size(0), 1, device=x.device)
            x = torch.cat([x, ones], dim=1)

        h = self.activation(x @ self.fc1)

        if self.use_bias:
            ones = torch.ones(h.size(0), 1, device=h.device)
            h = torch.cat([h, ones], dim=1)

        out = h @ self.fc2
        return out.squeeze(0) if out.size(0) == 1 else out
