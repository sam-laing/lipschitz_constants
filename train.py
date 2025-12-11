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

    train_loader, val_loader, test_loader = cifar10_5k_make_loaders(cfg)
    model = build_model(cfg)
    engine = Engine(model=model, cfg=cfg)

    for i in range(cfg.iters):
        print(f"iter {i+1}/{cfg.iters}")
        
        for x, y in train_loader:
            train_metrics = engine.step(x, y)
        
        # Print
        print("Training metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value}")
        
        # Log to W&B
        if use_wandb:
            log_training_metrics(
                train_metrics, 
                step=engine.iteration,
                log_lipschitz=cfg.track_lipschitz
            )

        # Validation
        val_metrics = engine.eval(val_loader)
        print(f"Validation metrics: {val_metrics}")
        
        if use_wandb:
            log_validation_metrics(val_metrics, step=engine.iteration)
    
    # Test set eval
    test_metrics = engine.eval(test_loader)
    print(f"Test metrics: {test_metrics}")
    """ 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tml_dict = evaluate_model(model, test_loader, cfg, device )
    print(tml_dict) 
    """ 
    if use_wandb:
        log_test_summary(test_metrics, final_step=engine.iteration)
        wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./config/config.yaml")
    p.add_argument("--job_idx", type=int, default=0)
    args = p.parse_args()

    main(args.config, args.job_idx)