import torch 
import torch.nn as nn 

from data import cifar10_5k_make_loaders   
from engine import Engine 
from utils import load_config, init_wandb, config_to_ns
from models import build_model

import argparse
from types import SimpleNamespace

from data import cifar10_5k_make_loaders

def main(config_path: str, job_idx: int = 0):
    cfg_dict = load_config(config_path)               
    cfg = config_to_ns(cfg_dict)                     
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    

    train_loader, val_loader, test_loader = cifar10_5k_make_loaders(cfg)

    model = build_model(cfg)


    engine = Engine(model=model, cfg=cfg)

    for x,y in train_loader:
        metrics = engine.step(x,y)
        #pretty print the metrics using pprint
        print("Training metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        break
    
    val_metrics = engine.eval(val_loader)
    print("Validation metrics:", val_metrics)
    





    
    



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./config/config.yaml")
    p.add_argument("--job_idx", type=int, default=0)
    args = p.parse_args()


    main(args.config, args.job_idx)






