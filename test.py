import torch 
import torch.nn as nn 
import os

from data import cifar10_5k_make_loaders, cifar10_make_loaders
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
path = "/fast/slaing/converged_weights/exp2/"

def main(store_weights=False):
    config_path = "./config/config_inter.yaml"
    cfg_dict = load_config(config_path, job_idx=0)               
    cfg = config_to_ns(cfg_dict)                     
    
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    use_wandb = bool(getattr(cfg, "wandb_project_name", None))
    if use_wandb:
        maybe_init_wandb(cfg, job_idx=0)

    # Choose loader based on dataset
    if cfg.dataset == 'cifar10_5k':
        train_loader, val_loader, test_loader = cifar10_5k_make_loaders(cfg)
    elif cfg.dataset == 'cifar10':
        train_loader, val_loader, test_loader = cifar10_make_loaders(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    
    model = build_model(cfg)
    engine = Engine(model=model, cfg=cfg)

    # Log initial loss before any training
    print("Computing initial loss...")
    init_metrics = engine.eval(train_loader)
    print(f"Initial train loss: {init_metrics['loss']:.6f}, accuracy: {init_metrics['accuracy']:.4f}")
    
    if use_wandb:
        wandb.log({
            "train/loss_init": init_metrics['loss'],
            "train/accuracy_init": init_metrics['accuracy']
        }, step=0)

    for i in range(cfg.iters):
        print(f"iter {i+1}/{cfg.iters}")
        
        for x, y in train_loader:
            train_metrics = engine.step(x, y)
        
        # Print
        print("Training metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value}")
        if i % 50 == 0:
            #save the weight to path 
            save_path = path + f"opt_{cfg.optimizer}_lr{cfg.lr}_{i+1}_{cfg.iters}_seed{cfg.seed}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model weights to {save_path}")
            
        
        if use_wandb:
            log_training_metrics(
                train_metrics, 
                step=engine.iteration,
                log_lipschitz=cfg.track_lipschitz, 
                log_hessian=cfg.track_hessian
            )

        # Validation
        val_metrics = engine.eval(val_loader)
        print(f"Validation metrics: {val_metrics}")
        
        if use_wandb:
            log_validation_metrics(val_metrics, step=engine.iteration)
    
    # Test set eval
    test_metrics = engine.eval(test_loader)
    print(f"Test metrics: {test_metrics}")
    
    if use_wandb:
        log_test_summary(test_metrics, final_step=engine.iteration)
        wandb.finish()

    if store_weights:
        save_path = path + f"opt_{cfg.optimizer}_lr{cfg.lr}_{cfg.iters}_seed{cfg.seed}_model{cfg.model}_dataset{cfg.dataset}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model weights to {save_path}")


if __name__ == "__main__":
    main(store_weights=True)
