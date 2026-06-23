import torch

_INPUT_DIMS = {
    'cifar10_5k': 32 * 32 * 3,
    'cifar10':    32 * 32 * 3,
    'mnist_5k':   28 * 28,
    'fmnist_5k':  28 * 28,
}
_OUTPUT_DIMS = {
    'cifar10': 10,
}

def _input_dim(cfg):
    return _INPUT_DIMS.get(cfg.dataset, 32 * 32 * 3)

def _output_dim(cfg):
    return _OUTPUT_DIMS.get(cfg.dataset, 5)


def build_model(cfg):
    if cfg.model == 'mlp':
        from .mlp import MLP
        input_dim = _input_dim(cfg)
        hidden_dim = int(input_dim * cfg.hidden_dim)
        output_dim = _output_dim(cfg)

        return MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=cfg.activation if hasattr(cfg, 'activation') else 'relu',
            seperate_biases=cfg.seperate_biases,
        )

    elif cfg.model == 'mlp_ortho':
        from .mlp_ortho import MLP
        input_dim = _input_dim(cfg)
        hidden_dim = int(input_dim * cfg.hidden_dim)
        output_dim = _output_dim(cfg)
        if cfg.dataset == 'cifar10':
            output_dim = 10

        ortho_rank = getattr(cfg, 'ortho_rank', None)
        return MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_bias=cfg.use_bias if hasattr(cfg, 'use_bias') else True,
            activation=cfg.activation if hasattr(cfg, 'activation') else 'relu',
            separate_biases=cfg.seperate_biases,
            weight_init=cfg.weight_init,
            ortho_rank=ortho_rank
        )
    
    elif cfg.model == 'linear_regression':
        from .linear_regression import LinearRegression
        input_dim = _input_dim(cfg)
        output_dim = _output_dim(cfg)

        ortho_rank = getattr(cfg, 'ortho_rank', None)
        return LinearRegression(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=cfg.use_bias if hasattr(cfg, 'use_bias') else True,
            weight_init=cfg.weight_init,
            ortho_rank=ortho_rank
        )
    elif cfg.model == "resnet18":
        from torchvision.models import resnet18
        output_dim = 5
        if cfg.dataset == 'cifar10':
            output_dim = 10
        return resnet18(num_classes=output_dim)
    
    elif cfg.model == 'mlp_ortho_v2':
        from .mlp_ortho_v2 import MLP
        input_dim = _input_dim(cfg)
        hidden_dim = int(input_dim * cfg.hidden_dim)
        output_dim = _output_dim(cfg)

        return MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=cfg.activation if hasattr(cfg, 'activation') else 'relu',
            include_bias=cfg.use_bias if hasattr(cfg, 'use_bias') else True,
            init_mode=cfg.init_mode if hasattr(cfg, 'init_mode') else 'kaiming_orthog',
            init_gain=cfg.init_gain if hasattr(cfg, 'init_gain') else 0.02,
            init_nonlinearity=cfg.init_nonlinearity if hasattr(cfg, 'init_nonlinearity') else 'relu',
            perturb_radius=cfg.perturb_radius if hasattr(cfg, 'perturb_radius') else 0.01,
            perturb_w_star_mode=cfg.perturb_w_star_mode if hasattr(cfg, 'perturb_w_star_mode') else 'kaiming',
        )
    
    elif cfg.model == "cnn":
        from .cnn import SimpleCNN
        output_dim = 5
        if cfg.dataset == 'cifar10':
            output_dim = 10
        return SimpleCNN(num_classes=output_dim)

    else:
        raise ValueError(f"Unsupported model type: {cfg.model}")