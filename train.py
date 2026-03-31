import torch 
import torch.nn as nn 

from data import cifar10_5k_make_loaders, cifar10_make_loaders
from engine import Engine 
from utils import (
    load_config, 
    maybe_init_wandb, 
    config_to_ns,
    log_training_metrics,
    log_validation_metrics,
    log_test_summary,
    save_model_weights
)
from models import build_model
import wandb
import argparse
from types import SimpleNamespace
#from tml_tests import evaluate_model

def main(config_path: str, job_idx: int = 0):
    cfg_dict = load_config(config_path, job_idx=job_idx)               
    cfg = config_to_ns(cfg_dict)                     
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    use_wandb = bool(getattr(cfg, "wandb_project_name", None))
    if use_wandb:
        maybe_init_wandb(cfg, job_idx=job_idx)

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
        
        # For cifar10 with mini-batches, accumulate metrics across batches
        if cfg.dataset == 'cifar10' and cfg.batch_size != "full":
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for x, y in train_loader:
                train_metrics = engine.step(x, y)
                epoch_loss += train_metrics.get('loss', 0.0)
                epoch_accuracy += train_metrics.get('accuracy', 0.0)
                num_batches += 1
            
            # Compute epoch averages
            train_metrics = {
                'loss': epoch_loss / num_batches,
                'accuracy': epoch_accuracy / num_batches
            }
            # Use epoch number as step for mini-batch logging
            log_step = i + 1
        else:
            # For full batch or cifar10_5k, single step per epoch
            for x, y in train_loader:
                train_metrics = engine.step(x, y)
            log_step = engine.iteration
        
        # Print
        print("Training metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value}")
        
        # Log to W&B
        if use_wandb:
            log_training_metrics(
                train_metrics, 
                step=log_step,
                log_lipschitz=cfg.track_lipschitz, 
                log_hessian=cfg.track_hessian
            )

        # Validation
        val_metrics = engine.eval(val_loader)
        print(f"Validation metrics: {val_metrics}")
        
        if use_wandb:
            log_validation_metrics(val_metrics, step=log_step)
    
    # Test set eval
    test_metrics = engine.eval(test_loader)
    print(f"Test metrics: {test_metrics}")
    """ 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tml_dict = evaluate_model(model, test_loader, cfg, device )
    print(tml_dict) 
    """ 
    if use_wandb:
        # Use epoch count for final step in mini-batch case
        final_step = cfg.iters if (cfg.dataset == 'cifar10' and cfg.batch_size != "full") else engine.iteration
        log_test_summary(test_metrics, final_step=final_step)
        wandb.finish()
    
    # Save model weights
    save_model_weights(model, cfg)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./config/config.yaml")
    p.add_argument("--job_idx", type=int, default=0)
    args = p.parse_args()

    main(args.config, args.job_idx)