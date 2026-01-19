import torch 


def build_model(cfg):
    if cfg.model == 'mlp':
        from .mlp import MLP
        # will probably add datasets so these numbers won't be hardcoded later
        input_dim = 32 * 32 * 3  
        hidden_dim = int(input_dim * cfg.hidden_dim)  
        output_dim = 5 
        if cfg.dataset == 'cifar10':
            output_dim = 10

        return MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=cfg.activation if hasattr(cfg, 'activation') else 'relu',
            seperate_biases=cfg.seperate_biases,
        )
    
    elif cfg.model == 'mlp_ortho':
        from .mlp_ortho import MLP
        input_dim = 32 * 32 * 3  
        hidden_dim = int(input_dim * cfg.hidden_dim)  
        output_dim = 5 
        if cfg.dataset == 'cifar10':
            output_dim = 10

        ortho_rank = getattr(cfg, 'ortho_rank', None)
        return MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=cfg.activation if hasattr(cfg, 'activation') else 'relu',
            separate_biases=cfg.seperate_biases,
            weight_init=cfg.weight_init,
            ortho_rank=ortho_rank
        )
    
    elif cfg.model == 'linear_regression':
        from .linear_regression import LinearRegression
        input_dim = 32 * 32 * 3  
        output_dim = 5
        if cfg.dataset == 'cifar10':
            output_dim = 10

        ortho_rank = getattr(cfg, 'ortho_rank', None)
        return LinearRegression(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=True,
            weight_init=cfg.weight_init,
            ortho_rank=ortho_rank
        )
    
    elif cfg.model == "cnn":
        from .cnn import SimpleCNN
        output_dim = 5
        if cfg.dataset == 'cifar10':
            output_dim = 10
        return SimpleCNN(num_classes=output_dim)

    else:
        raise ValueError(f"Unsupported model type: {cfg.model}")