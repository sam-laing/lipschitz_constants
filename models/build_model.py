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
            activation='relu',
            seperate_biases=cfg.seperate_biases
        )
    
    elif cfg.model == "cnn":
        from .cnn import SimpleCNN
        output_dim = 5
        if cfg.dataset == 'cifar10':
            output_dim = 10
        return SimpleCNN(num_classes=output_dim)

    else:
        raise ValueError(f"Unsupported model type: {cfg.model}")