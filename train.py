import torch 
import torch.nn as nn 
import time

from data import cifar10_5k_make_loaders   
from engine import Engine 
from utils import load_config, maybe_init_wandb, config_to_ns
from models import build_model
import wandb
import argparse
from types import SimpleNamespace

def main(config_path: str, job_idx: int = 0):
    cfg_dict = load_config(config_path, job_idx=job_idx)               
    cfg = config_to_ns(cfg_dict)                     
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    # Initialize W&B
    use_wandb = bool(getattr(cfg, "wandb_project_name", None))
    maybe_init_wandb(cfg, job_idx=job_idx)

    train_loader, val_loader, test_loader = cifar10_5k_make_loaders(cfg)
    model = build_model(cfg)
    engine = Engine(model=model, cfg=cfg)

    # Training loop
    for i in range(cfg.iters):
        print(f"iter {i+1}/{cfg.iters}")
        
        for x, y in train_loader:
            train_metrics = engine.step(x, y)
        
        # Print
        print("Training metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value}")
        

        val_metrics = engine.eval(val_loader)
        print(f"Validation metrics: {val_metrics}")
        
        if use_wandb:
            wandb.log({
                "train/loss": float(train_metrics["loss"]),
            }, step=engine.iteration)

            wandb.log({
                "val/loss": float(val_metrics["loss"]),
                "val/accuracy": float(val_metrics["accuracy"]),
            }, step=engine.iteration)
    
    #test set eval
    test_metrics = engine.eval(test_loader)
    print(f"Test metrics: {test_metrics}")
    
    if use_wandb:
        # Log as summary only (shows in run table, no chart mixing scales)
        wandb.run.summary["test_loss"] = float(test_metrics["loss"])
        wandb.run.summary["test_accuracy"] = float(test_metrics["accuracy"])
        wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./config/config.yaml")
    p.add_argument("--job_idx", type=int, default=0)
    args = p.parse_args()

    main(args.config, args.job_idx)