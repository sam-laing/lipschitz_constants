import os
import yaml
from types import SimpleNamespace
from typing import Any, Dict, Optional
from collections import namedtuple
from itertools import product
import wandb

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config(path, job_idx=None):
  """
  Parse a yaml file and return the correspondent config as a dict.
  If the config files has multiple entries, returns the one corresponding to job_idx.
  """
  
  with open(path, 'r') as file:
    config_dict = yaml.safe_load(file)

  if job_idx is None:
    cfg = config_dict
  else:
    keys = list(config_dict.keys())
    values = [val if isinstance(val, list) else [val] for val in config_dict.values()]
    combinations = list(product(*values))

    if job_idx >= len(combinations):
      raise ValueError("job_idx exceeds the total number of hyperparam combinations.")

    combination = combinations[job_idx]
    cfg = {keys[i]: combination[i] for i in range(len(keys))}

  return cfg

def config_to_ns(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Convert dict to SimpleNamespace for attribute access."""
    return SimpleNamespace(**cfg)


def maybe_init_wandb(cfg: SimpleNamespace, job_idx: int = 0):
    """
    initialize Weights & Biases logging.

    set basically everything in the config as a tag
    
    """
    use_wandb = cfg.wandb_project_name is not None
    if use_wandb:
      os.environ["WANDB__SERVICE_WAIT"] = "600"
      os.environ["WANDB_SILENT"] = "true"

      wandb_run_name = f"{cfg.optimizer},"
      if not cfg.orthogonalize:
         wandb_run_name += f"ns={cfg.ns_steps},"

      wandb_run_name += f"lr={cfg.lr}, hd={cfg.hidden_dim}, seed={cfg.seed}"

      wandb.init(
          project=cfg.wandb_project_name,
          config=vars(cfg),
          name=wandb_run_name,
          tags = [
              f"job_{job_idx}",
              f"optim_{cfg.optimizer}",
              f"sched_{cfg.scheduler}",
              f"lr_{cfg.lr}", 
              f"momentum_{cfg.momentum}",
              f"weight_decay_{cfg.weight_decay}",
              f"seed_{cfg.seed}", 
              f"beta1_{cfg.beta1}",
              f"beta2_{cfg.beta2}",
              f"batch_size_{cfg.batch_size}",   
          ]
      )

def log(cfg, metrics, step):
    """
    Log metrics to Weights & Biases.
    """
    wandb.log(metrics, step=step)



