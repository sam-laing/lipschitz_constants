import torch
import torch.nn as nn


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

        elif init_type == 'full_orthogonal':
            # Orthogonal regression init: ||Wx||^2 ≈ ||x||^2
            W = torch.empty(out_features, in_features)
            nn.init.orthogonal_(W)
            return W.T

        elif init_type == 'factorized_orthogonal':
            raise ValueError(
                "factorized_orthogonal is not supported for regression; "
                "it introduces rank deficiency."
            )

        else:
            raise ValueError(f"Unsupported init_type: {init_type}")


class LinearRegression(nn.Module):
    """
    Simple linear regression model: y = Wx + b
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

        self.weight_init = weight_init
        self.ortho_rank = ortho_rank

        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        self._init_layer()

    def _init_layer(self):
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
        return self.fc(x)
