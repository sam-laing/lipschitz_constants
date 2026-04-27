"""2-layer MLP with orthogonal initialization strategies for Muon optimizer.

Three initialization modes:
  1. "orthogonal"       — pure scaled orthogonal Q
  2. "perturbed"        — W* + radius * Q (orthogonal perturbation around a target)
  3. "kaiming_orthog"   — Kaiming-scaled orthogonal (preserves variance like kaiming but with orthogonal structure)

The key insight for Muon: since Muon projects gradients onto the Stiefel manifold
(orthonormal matrices) via Newton-Schulz, starting near that manifold means the
optimizer doesn't waste early steps reshaping the singular value spectrum.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Literal
from enum import Enum


class InitMode(str, Enum):
    ORTHOGONAL = "orthogonal"
    PERTURBED = "perturbed"
    KAIMING_ORTHOG = "kaiming_orthog"


def _orthogonal_matrix(rows: int, cols: int, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """Generate an orthogonal (or semi-orthogonal) matrix via QR decomposition.
    
    For non-square matrices, returns Q from the QR decomposition of a
    random Gaussian matrix, giving a uniformly-distributed element of
    the Stiefel manifold V_{min(m,n)}(R^{max(m,n)}).
    """
    mat = torch.randn(rows, cols, device=device, dtype=dtype)
    if rows >= cols:
        Q, R = torch.linalg.qr(mat)
        # Make QR unique by ensuring diag(R) > 0
        sign = torch.sign(torch.diag(R))
        sign[sign == 0] = 1
        Q = Q * sign.unsqueeze(0)
    else:
        Q, R = torch.linalg.qr(mat.T)
        sign = torch.sign(torch.diag(R))
        sign[sign == 0] = 1
        Q = (Q * sign.unsqueeze(0)).T
    return Q


def orthogonal_init_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Pure orthogonal init: W = gain * Q.
    
    For 2D weight matrices this is straightforward. All singular values 
    will be exactly `gain`.
    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError("Orthogonal init requires ndim >= 2")
        
        rows, cols = tensor.shape[0], tensor.shape[1]
        Q = _orthogonal_matrix(rows, cols, device=tensor.device, dtype=tensor.dtype)
        tensor.copy_(gain * Q)
    return tensor


def kaiming_orthogonal_init_(tensor: torch.Tensor, nonlinearity: str = 'relu') -> torch.Tensor:
    """Kaiming-scaled orthogonal init: W = sqrt(2 / fan_in) * Q.
    
    This gives you the variance-preserving property of Kaiming init
    (correct forward-pass signal magnitude for ReLU/SiLU networks)
    but with perfectly conditioned singular values (condition number = 1).
    
    Standard Kaiming draws iid Gaussian entries with std = sqrt(2/fan_in),
    which gives singular values following a Marchenko-Pastur distribution.
    Here we replace the random matrix with an orthogonal one scaled to
    match the same operator norm / Frobenius norm.
    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError("Kaiming orthogonal init requires ndim >= 2")
        
        fan_in = tensor.shape[1]
        
        # Gain lookup (matching torch.nn.init.calculate_gain)
        gains = {
            'linear': 1.0,
            'relu': math.sqrt(2.0),
            'silu': math.sqrt(2.0),
            'tanh': 5.0 / 3.0,
            'sigmoid': 1.0,
        }
        gain = gains.get(nonlinearity, 1.0)
        
        # Kaiming std = gain / sqrt(fan_in)
        scale = gain / math.sqrt(fan_in)
        
        rows, cols = tensor.shape[0], tensor.shape[1]
        Q = _orthogonal_matrix(rows, cols, device=tensor.device, dtype=tensor.dtype)
        tensor.copy_(scale * Q)
    return tensor


def perturbed_orthogonal_init_(
    tensor: torch.Tensor,
    W_star: Optional[torch.Tensor] = None,
    radius: float = 0.01,
    gain: float = 1.0,
    w_star_mode: Literal['zeros', 'kaiming', 'custom'] = 'kaiming'
) -> torch.Tensor:
    """Perturbed init: W = W* + radius * Q for orthogonal Q.
    
    This lets you start near a known-good solution W* while adding a
    controlled orthogonal perturbation.
    
    Ideas for W*:
      - 'zeros':   W* = 0, so W = radius * Q. Useful for output projections.
      - 'kaiming': W* = kaiming_uniform init. Adds orthogonal structure on top.
      - 'custom':  Pass your own W_star tensor.
    
    The radius controls how far you wander from W*.
    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError("Perturbed orthogonal init requires ndim >= 2")
        
        rows, cols = tensor.shape[0], tensor.shape[1]
        
        # Build W*
        if w_star_mode == 'custom':
            if W_star is None:
                raise ValueError("Must provide W_star tensor when w_star_mode='custom'")
            assert W_star.shape == tensor.shape, f"W_star shape {W_star.shape} != tensor shape {tensor.shape}"
            w_star_val = W_star.clone()
        elif w_star_mode == 'zeros':
            w_star_val = torch.zeros_like(tensor)
        elif w_star_mode == 'kaiming':
            w_star_val = tensor.clone()
            fan_in = cols
            std = math.sqrt(2.0) / math.sqrt(fan_in)
            w_star_val.normal_(0, std)
        else:
            raise ValueError(f"Unknown w_star_mode: {w_star_mode}")
        
        Q = _orthogonal_matrix(rows, cols, device=tensor.device, dtype=tensor.dtype)
        result = w_star_val + radius * gain * Q
        tensor.copy_(result)
    return tensor


class MLP(nn.Module):
    """2-layer MLP with configurable orthogonal initialization.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        activation: Activation function ('relu', 'tanh', 'sigmoid', 'silu')
        include_bias: Whether to include bias terms
        init_mode: Initialization mode ('orthogonal', 'kaiming_orthog', 'perturbed', 'kaiming')
        init_gain: Gain for orthogonal mode
        init_nonlinearity: Nonlinearity for kaiming gain calculation
        perturb_radius: Radius for perturbed mode
        perturb_w_star_mode: W* strategy for perturbed mode ('zeros', 'kaiming', 'custom')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = 'relu',
        include_bias: bool = True,
        init_mode: str = 'kaiming_orthog',
        init_gain: float = 0.02,
        init_nonlinearity: str = 'relu',
        perturb_radius: float = 0.01,
        perturb_w_star_mode: str = 'kaiming',
    ):
        super().__init__()
        
        # Activation function
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'silu': nn.SiLU(),
            'linear': nn.Identity(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]
        
        # Store config
        self.init_mode = init_mode
        self.init_gain = init_gain
        self.init_nonlinearity = init_nonlinearity
        self.perturb_radius = perturb_radius
        self.perturb_w_star_mode = perturb_w_star_mode
        
        # Layers
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=include_bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=include_bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights according to init_mode."""
        if self.init_mode == 'kaiming':
            # Standard PyTorch Kaiming init
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity=self.init_nonlinearity)
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
        
        elif self.init_mode == 'orthogonal':
            orthogonal_init_(self.fc1.weight, gain=self.init_gain)
            orthogonal_init_(self.fc2.weight, gain=self.init_gain / math.sqrt(2))
        
        elif self.init_mode == 'kaiming_orthog':
            kaiming_orthogonal_init_(self.fc1.weight, nonlinearity=self.init_nonlinearity)
            kaiming_orthogonal_init_(self.fc2.weight, nonlinearity='linear')
            # Scale output layer for residual-like behavior
            self.fc2.weight.data *= 1.0 / math.sqrt(2)
        
        elif self.init_mode == 'perturbed':
            perturbed_orthogonal_init_(
                self.fc1.weight,
                W_star=None,
                radius=self.perturb_radius,
                gain=self.init_gain,
                w_star_mode=self.perturb_w_star_mode,
            )
            perturbed_orthogonal_init_(
                self.fc2.weight,
                W_star=None,
                radius=self.perturb_radius,
                gain=self.init_gain / math.sqrt(2),
                w_star_mode=self.perturb_w_star_mode,
            )
        
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
        
        # Initialize biases to zero
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 2-layer MLP.
        
        Args:
            x: Input tensor of shape (..., input_dim)
        
        Returns:
            Output tensor of shape (..., output_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
