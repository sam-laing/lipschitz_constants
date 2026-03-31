# MLP with Orthogonal Initialization (mlp_ortho_v2)

A 2-layer MLP with configurable orthogonal initialization strategies, designed to work optimally with the Muon optimizer.

## Key Features

### Three Initialization Modes

1. **`orthogonal`** — Pure scaled orthogonal Q
   - W = gain * Q, where Q is an orthogonal matrix
   - All singular values exactly equal to `gain`
   - Best for starting with maximally diverse weight structure

2. **`kaiming_orthog`** — Kaiming-scaled orthogonal  
   - W = (gain / √fan_in) * Q
   - Preserves variance properties of Kaiming initialization
   - Perfectly conditioned singular values (condition number = 1)
   - **Recommended** for most cases, especially with ReLU/SiLU networks

3. **`perturbed`** — W* + radius * Q
   - Orthogonal perturbation around a target matrix W*
   - Start near a known solution while exploring orthogonal directions
   - Configurable  `w_star_mode`: 'zeros', 'kaiming', or 'custom'

4. **`kaiming`** — Standard PyTorch Kaiming initialization
   - For comparison/baseline

## Why Orthogonal Initialization for Muon?

The Muon optimizer projects gradients onto the Stiefel manifold (orthonormal matrices) via Newton-Schulz iterations. Starting near this manifold means:
- The optimizer doesn't waste early steps reshaping the singular value spectrum
- Faster convergence to well-conditioned weight matrices
- Better numerical stability during training

## Usage

### Basic Setup

```python
from models.mlp_ortho_v2 import MLP

# Create model with Kaiming-orthogonal initialization (recommended)
model = MLP(
    input_dim=3072,
    hidden_dim=4608,
    output_dim=10,
    activation='relu',
    include_bias=True,
    init_mode='kaiming_orthog',
)
```

### Configuration in YAML

Edit your config file (e.g., `config/config.yaml`):

```yaml
model: "mlp_ortho_v2"
hidden_dim: 1.5          # Hidden dim as proportion of input

# Initialization settings
include_bias: true
activation: "relu"       # Options: relu, tanh, sigmoid, silu, linear

# Initialization mode: "kaiming", "orthogonal", "kaiming_orthog", "perturbed"
init_mode: "kaiming_orthog"
init_gain: 0.02
init_nonlinearity: "relu"

# For "perturbed" mode only:
perturb_radius: 0.01            # Magnitude of orthogonal perturbation
perturb_w_star_mode: "kaiming"  # How to build W*: "zeros", "kaiming", "custom"
```

### Running with Train Script

```bash
# Using kaiming_orthog initialization (recommended)
python train.py --config config/config_mlp_ortho_v2_kaiming.yaml

# Using pure orthogonal initialization
python train.py --config config/config_mlp_ortho_v2_orthogonal.yaml

# Using perturbed initialization
python train.py --config config/config_mlp_ortho_v2_perturbed.yaml

# Custom config
python train.py --config config/my_config.yaml --job_idx 0
```

### Running with Test Script

```bash
python test.py --config config/config_mlp_ortho_v2_kaiming.yaml --interactive
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `init_mode` | str | 'kaiming_orthog' | Initialization strategy |
| `init_gain` | float | 0.02 | Scaling factor for orthogonal modes |
| `init_nonlinearity` | str | 'relu' | Nonlinearity for Kaiming gain calculation |
| `include_bias` | bool | True | Whether to use bias terms |
| `activation` | str | 'relu' | Hidden activation function |
| `perturb_radius` | float | 0.01 | Perturbation magnitude for perturbed mode |
| `perturb_w_star_mode` | str | 'kaiming' | W* strategy: 'zeros', 'kaiming', 'custom' |

## Example Configurations

Three pre-configured examples are provided:

1. **config/config_mlp_ortho_v2_kaiming.yaml** — Kaiming-orthogonal (recommended)
2. **config/config_mlp_ortho_v2_orthogonal.yaml** — Pure orthogonal
3. **config/config_mlp_ortho_v2_perturbed.yaml** — Perturbed orthogonal

## Implementation Details

### Orthogonal Matrix Generation

Uses QR decomposition on random Gaussian matrices:
- For m ≥ n: standard QR
- For m < n: QR on transpose
- Sign-corrected for uniqueness

### Kaiming-Orthogonal Scaling

Combines Kaiming variance preservation with orthogonal structure:
- Standard Kaiming: σ = √(2/fan_in)
- Orthogonal matrix Q has all singular values = 1
- Final weight: W = (gain/√fan_in) * Q

### Weight Initialization

- **Hidden layer (fc1)**: Full initialization according to init_mode
- **Output layer (fc2)**: Scaled by 1/√2 for residual-like behavior
- **Biases**: Always initialized to zero

## Compatibility

- Works seamlessly with existing `train.py` and `test.py` scripts
- Compatible with Muon, SGD, and other optimizers
- Supports Weights & Biases logging
- Works with both CIFAR-10 and CIFAR-10-5K datasets

## Files Modified/Created

- **Created**: `models/mlp_ortho_v2.py` — New model implementation
- **Modified**: `models/build_model.py` — Added support for new model
- **Created**: `config/config_mlp_ortho_v2_kaiming.yaml` — Example configs
- **Created**: `config/config_mlp_ortho_v2_orthogonal.yaml`
- **Created**: `config/config_mlp_ortho_v2_perturbed.yaml`
- **Created**: `README_ORTHO.md` — This file

## Internal Functions

### `_orthogonal_matrix(rows, cols, device, dtype)`
Generates semi-orthogonal matrix via QR decomposition.

### `orthogonal_init_(tensor, gain)`
Pure orthogonal initialization: W = gain * Q.

### `kaiming_orthogonal_init_(tensor, nonlinearity)`
Kaiming-scaled orthogonal: W = (gain/√fan_in) * Q.

### `perturbed_orthogonal_init_(tensor, W_star, radius, gain, w_star_mode)`
Perturbed orthogonal: W = W* + radius * Q.

## Notes

- The implementation preserves dimension handling for >2D tensors (though MLPs only use 2D weights)
- Bias initialization is always zeros, regardless of weight init mode
- Output layer scaling by 1/√2 mimics residual architectures even in non-residual MLPs
- For Muon optimization, `orthogonalize=True` and appropriate `ns_steps` in config recommended

## References

The orthogonal initialization strategies are adapted from transformer literature and optimized for compatibility with manifold-based optimizers like Muon.
