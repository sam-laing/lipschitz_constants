""" 
import torch
import torch.nn as nn
import math

class OrthogonalInitializer:
    @staticmethod
    def create_weight(shape, init_type='normal', activation='relu', rank=None):
        m, n = shape
        gain = nn.init.calculate_gain(activation)
        
        if init_type == 'normal':
            weight = torch.empty(m, n)
            std = gain / math.sqrt(m)
            nn.init.normal_(weight, mean=0.0, std=std)
            
        elif init_type == 'full_orthogonal':
            if m >= n:
                W = torch.randn(m, n)
                W, _ = torch.linalg.qr(W, mode='reduced')
            else:
                W = torch.randn(n, m)
                W, _ = torch.linalg.qr(W, mode='reduced')
                W = W.T
            
            current_std = torch.std(W)
            target_std = gain / math.sqrt(min(m, n))
            weight = W * (target_std / current_std)
            
        elif init_type == 'factorized_orthogonal':
            if rank is None:
                rank = min(m, n)
            rank = min(rank, m, n)
            
            U = torch.randn(m, rank)
            V = torch.randn(n, rank)
            
            U, _ = torch.linalg.qr(U, mode='reduced')
            V, _ = torch.linalg.qr(V, mode='reduced')
            
            weight = U @ V.T
            
            current_std = torch.std(weight)
            target_std = gain / math.sqrt(m)
            weight = weight * (target_std / current_std)
            
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")
        
        return weight


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', 
                 separate_biases=True, weight_init='normal', ortho_rank=None):
        super(MLP, self).__init__()
        self.separate_biases = separate_biases
        self.weight_init = weight_init
        self.ortho_rank = ortho_rank
        
        if separate_biases:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self._init_separate_layers(activation)
        else:
            fc1_full = OrthogonalInitializer.create_weight(
                shape=(input_dim + 1, hidden_dim),
                init_type=weight_init,
                activation=activation,
                rank=ortho_rank
            )
            self.fc1 = nn.Parameter(fc1_full)

            fc2_full = OrthogonalInitializer.create_weight(
                shape=(hidden_dim + 1, output_dim),
                init_type=weight_init,
                activation=activation,
                rank=ortho_rank
            )
            self.fc2 = nn.Parameter(fc2_full)
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]
    
    def _init_separate_layers(self, activation):
        if self.weight_init == 'normal':
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity=activation)
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity=activation)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
            
        elif self.weight_init in ['full_orthogonal', 'factorized_orthogonal']:
            W1 = OrthogonalInitializer.create_weight(
                shape=(self.fc1.in_features, self.fc1.out_features),
                init_type=self.weight_init,
                activation=activation,
                rank=self.ortho_rank
            )
            self.fc1.weight.data = W1.T
            
            W2 = OrthogonalInitializer.create_weight(
                shape=(self.fc2.in_features, self.fc2.out_features),
                init_type=self.weight_init,
                activation=activation,
                rank=self.ortho_rank
            )
            self.fc2.weight.data = W2.T
            
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        if self.separate_biases:
            return self.fc2(self.activation(self.fc1(x)))
        else:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            batch_size = x.shape[0]
            
            x_aug = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
            h = self.activation(x_aug @ self.fc1)
            
            h_aug = torch.cat([h, torch.ones(batch_size, 1, device=x.device)], dim=1)
            out = h_aug @ self.fc2
            
            if out.shape[0] == 1:
                return out.squeeze(0)
            return out


# Simple test function
def test_mlp():
    torch.manual_seed(42)
    
    # Test with combined weights and factorized orthogonal
    mlp = MLP(
        input_dim=10,
        hidden_dim=20,
        output_dim=5,
        activation='relu',
        separate_biases=False,
        weight_init='factorized_orthogonal',
        ortho_rank=8
    )
    
    x = torch.randn(32, 10)
    y = mlp(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean: {y.mean():.4f}")
    print(f"Output std: {y.std():.4f}")


if __name__ == "__main__":
    test_mlp()

""" 

import torch
import torch.nn as nn
import math


class OrthogonalInitializer:
    @staticmethod
    def create_weight(shape, init_type='normal', activation='relu', rank=None):
        m, n = shape
        gain = nn.init.calculate_gain(activation)

        if init_type == 'normal':
            weight = torch.empty(m, n)
            std = gain / math.sqrt(m)
            nn.init.normal_(weight, mean=0.0, std=std)

        elif init_type == 'full_orthogonal':
            if m >= n:
                W = torch.randn(m, n)
                W, _ = torch.linalg.qr(W, mode='reduced')
            else:
                W = torch.randn(n, m)
                W, _ = torch.linalg.qr(W, mode='reduced')
                W = W.T

            current_std = torch.std(W)
            target_std = gain / math.sqrt(min(m, n))
            weight = W * (target_std / current_std)

        elif init_type == 'factorized_orthogonal':
            if rank is None:
                rank = min(m, n)
            rank = min(rank, m, n)

            U = torch.randn(m, rank)
            V = torch.randn(n, rank)

            U, _ = torch.linalg.qr(U, mode='reduced')
            V, _ = torch.linalg.qr(V, mode='reduced')

            weight = U @ V.T

            current_std = torch.std(weight)
            target_std = gain / math.sqrt(m)
            weight = weight * (target_std / current_std)

        else:
            raise ValueError(f"Unsupported init_type: {init_type}")

        return weight


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
                    activation=activation,
                    rank=ortho_rank,
                )
            )

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation = activations[activation]

    def _init_separate_layers(self, activation):
        if self.weight_init == 'normal':
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity=activation)
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity=activation)

        elif self.weight_init in ['full_orthogonal', 'factorized_orthogonal']:
            W1 = OrthogonalInitializer.create_weight(
                shape=(self.fc1.in_features, self.fc1.out_features),
                init_type=self.weight_init,
                activation=activation,
                rank=self.ortho_rank,
            )
            self.fc1.weight.data = W1.T

            W2 = OrthogonalInitializer.create_weight(
                shape=(self.fc2.in_features, self.fc2.out_features),
                init_type=self.weight_init,
                activation=activation,
                rank=self.ortho_rank,
            )
            self.fc2.weight.data = W2.T

        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        if self.separate_biases:
            return self.fc2(self.activation(self.fc1(x)))

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


# Simple test function
def test_mlp():
    torch.manual_seed(42)

    mlp = MLP(
        input_dim=10,
        hidden_dim=20,
        output_dim=5,
        activation='relu',
        separate_biases=False,
        use_bias=False,  # ← pure W2 act(W1 x)
        weight_init='factorized_orthogonal',
        ortho_rank=None,
    )

    x = torch.randn(32, 10)
    y = mlp(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean: {y.mean():.4f}")
    print(f"Output std: {y.std():.4f}")


if __name__ == "__main__":
    test_mlp()
