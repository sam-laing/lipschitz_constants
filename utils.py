import os
import yaml
from types import SimpleNamespace
from typing import Any, Dict, Optional
from itertools import product
import wandb
import pyhessian  


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
    Initialize Weights & Biases logging.
    Set basically everything in the config as a tag.
    """
    use_wandb = cfg.wandb_project_name is not None
    if use_wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "600"
        os.environ["WANDB_SILENT"] = "true"

        wandb_run_name = f"{cfg.optimizer},"
        if cfg.orthogonalize and cfg.optimizer == "muon":
            wandb_run_name += "ortho,"
        elif not cfg.orthogonalize and cfg.optimizer == "muon":
            wandb_run_name += f"ns={cfg.ns_steps},"
        if cfg.dual_decay:
            wandb_run_name += "dual_decay,"


        if cfg.momentum != 0.0:
            wandb_run_name += f"mom={cfg.momentum},"

        wandb_run_name += f"lr={cfg.lr}, hd={cfg.hidden_dim}, sep={cfg.seperate_biases}, seed={cfg.seed}"

        wandb.init(
            project=cfg.wandb_project_name,
            config=vars(cfg),
            name=wandb_run_name,
            tags=[
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
                f"dual_decay_{cfg.dual_decay}" if hasattr(cfg, "dual_decay") else "dual_decay_False", 
                f"sep_biases_{cfg.seperate_biases}" if hasattr(cfg, "seperate_biases") else "sep_biases_True"
            ]
        )


def log_training_metrics(metrics_dict: Dict[str, Any], step: int, log_lipschitz: bool = False, log_hessian: bool = False):
    """
    Log training metrics to W&B with layer-first hierarchy:
    
    Structure:
      train/loss
      {layer_name}/lipschitz/{norm_type}
      {layer_name}/grad_norm/{norm_type}
    
    This makes each layer a top-level collapsible section.
    """
    logs = {}
    
    # Always log loss under train section
    if "loss" in metrics_dict:
        logs["train/loss"] = float(metrics_dict["loss"])
    
    # Log Lipschitz metrics with LAYER as top-level section
    if log_lipschitz:
        lipschitz_layerwise = metrics_dict.get("lipschitz_layerwise", {})
        grad_layerwise = metrics_dict.get("grad_layerwise", {})
        
        # Reorganize: layer -> metric_type -> norm_type
        # This creates: fc1_weight/lipschitz/L2, fc1_weight/grad_norm/L2, etc.
        
        for norm_type, layer_dict in lipschitz_layerwise.items():
            if layer_dict is None or not isinstance(layer_dict, dict):
                continue
                
            for layer_name, value in layer_dict.items():
                if value is not None and value != float('inf'):
                    # Layer as top-level section
                    clean_layer = layer_name.replace(".", "_")
                    logs[f"{clean_layer}/lipschitz/{norm_type}"] = float(value)
        
        # Log gradient norms
        for norm_type, layer_dict in grad_layerwise.items():
            if layer_dict is None or not isinstance(layer_dict, dict):
                continue
                
            for layer_name, value in layer_dict.items():
                if value is not None:
                    clean_layer = layer_name.replace(".", "_")
                    logs[f"{clean_layer}/grad_norm/{norm_type}"] = float(value)

    if log_hessian:
        """   
        this is the function     def hessian_stats(self, x, y, k=10):
        self.model.zero_grad(set_to_none=True)
        x,y = x.to(self.device), y.to(self.device)
        loss = self.criterion(self.model(x), y)

        params = [p for p in self.model.parameters() if p.requires_grad]
        dim = sum(p.numel() for p in params if p.requires_grad)
        def hvp_fn(v):
            return self.hessian_vector_product(loss, params, v)

        eigenvals = self.lancosz(hvp_fn, dim, k=k)
        eigenvals = eigenvals.detach().cpu()

        lambda_max = eigenvals[0].item()
        lambda_min = eigenvals[-1].item()
        cond_num = lambda_max / (lambda_min + 1e-8)

        return {
            "hessian_top_eigenvalues": eigenvals.tolist(),
            "hessian_lambda_max": lambda_max,
            "hessian_lambda_min": lambda_min,
            "hessian_condition_number": cond_num
        }
        so log the 4 pieces of information from this function in a panel 

        
        """
        hessian_top_eigenvalues = metrics_dict.get("hessian_top_eigenvalues", [])
        if hessian_top_eigenvalues:
            for idx, eigenval in enumerate(hessian_top_eigenvalues):
                logs[f"hessian/eigenvalue_{idx}"] = float(eigenval)
        if "hessian_lambda_max" in metrics_dict:
            logs["hessian/lambda_max"] = float(metrics_dict["hessian_lambda_max"])
        if "hessian_lambda_min" in metrics_dict:
            logs["hessian/lambda_min"] = float(metrics_dict["hessian_lambda_min"])
        if "hessian_condition_number" in metrics_dict:
            logs["hessian/condition_number"] = float(metrics_dict["hessian_condition_number"])
        

        
 

    
    if logs:
        wandb.log(logs, step=step)


def log_validation_metrics(metrics_dict: Dict[str, Any], step: int):
    """Log validation metrics to W&B."""
    logs = {
        "val/loss": float(metrics_dict["loss"]),
        "val/accuracy": float(metrics_dict["accuracy"]),
    }
    wandb.log(logs, step=step)


def log_test_summary(metrics_dict: Dict[str, Any], final_step: int):
    """
    Log final test metrics to W&B.
    Logs as both regular metrics (for charts) and summary (for tables).
    
    Args:
        metrics_dict: Dict with 'loss' and 'accuracy'
        final_step: Final training iteration (for x-axis positioning)
    """
    # Log as regular metrics (creates charts/bar charts)
    wandb.log({
        "test/loss": float(metrics_dict["loss"]),
        "test/accuracy": float(metrics_dict["accuracy"]),
    }, step=final_step)
    
    # Also log to summary for easy comparison in workspace table
    wandb.run.summary["test_loss"] = float(metrics_dict["loss"])
    wandb.run.summary["test_accuracy"] = float(metrics_dict["accuracy"])