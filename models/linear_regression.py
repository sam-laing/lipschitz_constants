import torch
import torch.nn as nn
import math


class OrthogonalInitializer:
    @staticmethod
    def create_weight(shape, init_type='normal', rank=None):
        """
        shape: (in_features, out_features)
        returns tensor of shape (in_features, out_features)
        """
        in_features, out_features = shape

        if init_type == 'normal':
            weight = torch.empty(in_features, out_features)
            nn.init.kaiming_normal_(weight, nonlinearity='linear')
            return weight

        # full orthogonal (column-orthogonal + scaling) 
        elif init_type == 'full_orthogonal':
            # nn.init.orthogonal_ works on (out, in)
            W = torch.empty(out_features, in_features)
            nn.init.orthogonal_(W)

            # fan-in scaling
            scale = 1.0 / math.sqrt(in_features)
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

            # fan-in scaling
            scale = 1.0 / math.sqrt(in_features)
            with torch.no_grad():
                weight.mul_(scale)

            return weight.T

        else:
            raise ValueError(f"Unsupported init_type: {init_type}")


class LinearRegression(nn.Module):
    """
    Simple linear regression model: y = Wx + b
    No hidden layers, no activations.
    Supports orthogonal vs normal initialization.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        use_bias=True,
        weight_init='normal',
        ortho_rank=None,
    ):
        super().__init__()

        self.use_bias = use_bias
        self.weight_init = weight_init
        self.ortho_rank = ortho_rank

        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        self._init_layer()

    def _init_layer(self):
        """Initialize weights based on weight_init type"""
        if self.weight_init == 'normal':
            nn.init.kaiming_normal_(self.fc.weight, nonlinearity='linear')
        else:
            W = OrthogonalInitializer.create_weight(
                shape=(self.fc.in_features, self.fc.out_features),
                init_type=self.weight_init,
                rank=self.ortho_rank,
            )
            self.fc.weight.data.copy_(W.T)

        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """Simple linear forward pass"""
        return self.fc(x)
