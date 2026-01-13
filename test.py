import torch 
import torch.nn as nn 

from data import cifar10_5k_make_loaders   
from engine import Engine 
from utils import (
    load_config, 
    maybe_init_wandb, 
    config_to_ns,
    log_training_metrics,
    log_validation_metrics,
    log_test_summary
)
from models import build_model
import wandb
import argparse
from types import SimpleNamespace


if __name__ == "__main__":

    #set the seed
    torch.manual_seed(433)

    print("-------")
    from dataclasses import dataclass
    @dataclass  
    class Config:
        model: str = 'mlp_ortho'
        dataset: str = 'cifar10_5k'
        hidden_dim: float = 3.0
        batch_size: int = None 
        seperate_biases: bool = False
        num_workers: int = 2
        weight_init: str = 'factorized_orthogonal'
        ortho_rank: int = None
        seed: int = 1000
        iters: int = 2
        wandb_project_name: str = ''
        track_lipschitz: bool = False
        track_hessian: bool = False
    """ 
    #ok just generate a random x and see the output's loss for a number of different num_classes options

    for seed in [11,12,13,14,15,16,1768,18888]:
        torch.manual_seed(seed)
        cfg = Config(model="mlp", seperate_biases=True)
        cfg_toggle = Config(model="mlp", seperate_biases=False)

        x = torch.randn(2048, 32*32*3)
        y = torch.randint(0, 5, (2048,))
        from models.mlp import MLP


        model = build_model(cfg)
        model_toggle = build_model(cfg_toggle)
        criterion = nn.CrossEntropyLoss()

        outputs = model(x)
        loss = criterion(outputs, y)
        print(f"Output shape (separate biases): {outputs.shape}, Loss: {loss.item():.4f}")
        outputs_toggle = model_toggle(x)
        loss_toggle = criterion(outputs_toggle, y)
        print(f"Output shape (combined weights): {outputs_toggle.shape}, Loss: {loss_toggle.item():.4f}")

        print("-------")


    """ 
    for seed in [42, 100, 2024, 10000, 222, 333, 44, 55]:
        torch.manual_seed(seed) 
        cfg_normal = Config(weight_init='normal', seperate_biases=False)

        cfg_fact_ortho = Config(weight_init='factorized_orthogonal')
        cfg_fact_ortho_sep_bias = Config(weight_init='factorized_orthogonal', seperate_biases=True)
        cfg_normal_sep_bias = Config(weight_init='normal', seperate_biases=True)
        model = build_model(cfg_normal)
        model_fact_ortho = build_model(cfg_fact_ortho)
        model_fact_ortho_sep_bias = build_model(cfg_fact_ortho_sep_bias)
        model_normal_sep_bias = build_model(cfg_normal_sep_bias)

        cfg = Config(model ="mlp")
        cfg_sep_bias = Config(model ="mlp", seperate_biases=True)
        train_loader, val_loader, test_loader = cifar10_5k_make_loaders(cfg)    #_normal)
        

        # just check out the initlization loss for both models
        def compute_init_loss(model, data_loader):
            model.eval()
            total_loss = 0.0
            criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                for x, y in data_loader:
                    
                    outputs = model(x)

                    loss = criterion(outputs, y)
                    total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / len(data_loader.dataset)
            return avg_loss


        init_loss_normal = compute_init_loss(model, train_loader)
        init_loss_fact_ortho = compute_init_loss(model_fact_ortho, train_loader)
        #init_loss_fact_ortho_sep_bias = compute_init_loss(model_fact_ortho_sep_bias, train_loader)
        init_loss_normal_sep_bias = compute_init_loss(model_normal_sep_bias, train_loader)

        print(f"Initial loss with normal init: {init_loss_normal:.4f}")
        print(f"Initial loss with factorized orthogonal init: {init_loss_fact_ortho:.4f}")
        #print(f"Initial loss with factorized orthogonal init and separate biases: {init_loss_fact_ortho_sep_bias:.4f}")
        print(f"Initial loss with normal init and separate biases: {init_loss_normal_sep_bias:.4f}")
