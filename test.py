import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from data import cifar10_5k_make_loaders, cifar10_make_loaders
from engine import Engine
from utils import (
    load_config,
    maybe_init_wandb,
    config_to_ns,
    log_training_metrics,
    log_validation_metrics,
    log_test_summary,
    save_model_weights,
)
from models import build_model
import wandb


def _prompt_str(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw != "" else default


def _prompt_int(label: str, default: Optional[int]) -> Optional[int]:
    default_str = "" if default is None else str(default)
    raw = input(f"{label} [{default_str}]: ").strip()
    if raw == "":
        return default
    return int(raw)


def _prompt_float(label: str, default: Optional[float]) -> Optional[float]:
    default_str = "" if default is None else str(default)
    raw = input(f"{label} [{default_str}]: ").strip()
    if raw == "":
        return default
    return float(raw)


def _prompt_bool(label: str, default: bool) -> bool:
    default_str = "y" if default else "n"
    raw = input(f"{label} [y/n, default {default_str}]: ").strip().lower()
    if raw == "":
        return default
    if raw in {"y", "yes", "true", "1"}:
        return True
    if raw in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"Invalid boolean input: {raw}")


def _safe_get(cfg: Any, name: str, fallback: Any) -> Any:
    return getattr(cfg, name, fallback)


def _apply_overrides(cfg: Any, overrides: Dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def run_pipeline(cfg: Any, max_train_batches: Optional[int] = None,
                 max_val_batches: Optional[int] = None,
                 max_test_batches: Optional[int] = None) -> None:
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    use_wandb = bool(getattr(cfg, "wandb_project_name", None))
    if use_wandb:
        maybe_init_wandb(cfg, job_idx=getattr(cfg, "job_idx", 0))

    if cfg.dataset == "cifar10_5k":
        train_loader, val_loader, test_loader = cifar10_5k_make_loaders(cfg)
    elif cfg.dataset == "cifar10":
        train_loader, val_loader, test_loader = cifar10_make_loaders(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    model = build_model(cfg)
    engine = Engine(model=model, cfg=cfg)

    print("Computing initial loss...")
    init_metrics = _eval_with_limit(engine, train_loader, max_batches=max_train_batches)
    print(f"Initial train loss: {init_metrics['loss']:.6f}, accuracy: {init_metrics['accuracy']:.4f}")

    if use_wandb:
        wandb.log({
            "train/loss_init": init_metrics["loss"],
            "train/accuracy_init": init_metrics["accuracy"],
        }, step=0)

    for i in range(cfg.iters):
        print(f"iter {i + 1}/{cfg.iters}")
        train_metrics = _train_with_limit(engine, train_loader, max_batches=max_train_batches)

        print("Training metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value}")

        if use_wandb:
            log_training_metrics(
                train_metrics,
                step=engine.iteration,
                log_lipschitz=cfg.track_lipschitz,
                log_hessian=cfg.track_hessian,
            )

        val_metrics = _eval_with_limit(engine, val_loader, max_batches=max_val_batches)
        print(f"Validation metrics: {val_metrics}")

        if use_wandb:
            log_validation_metrics(val_metrics, step=engine.iteration)

    test_metrics = _eval_with_limit(engine, test_loader, max_batches=max_test_batches)
    print(f"Test metrics: {test_metrics}")

    if use_wandb:
        log_test_summary(test_metrics, final_step=engine.iteration)
        wandb.finish()
    
    # Save model weights
    save_model_weights(model, cfg)


def _train_with_limit(engine: Engine, loader, max_batches: Optional[int] = None) -> Dict[str, Any]:
    last_metrics = None
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        last_metrics = engine.step(x, y)
    if last_metrics is None:
        raise ValueError("No training batches were processed.")
    return last_metrics


@torch.no_grad()
def _eval_with_limit(engine: Engine, loader, max_batches: Optional[int] = None) -> Dict[str, Any]:
    engine.model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(engine.device), y.to(engine.device)

        if engine.cfg.loss == "mse":
            y_loss = torch.nn.functional.one_hot(y, num_classes=engine.output_dim).float()
        else:
            y_loss = y

        output = engine.model(x)
        loss = engine.criterion(output, y_loss)
        total_loss += loss.item() * x.size(0)
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == y).sum().item()
        total_seen += x.size(0)

    if total_seen == 0:
        raise ValueError("No evaluation batches were processed.")

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
    }


def interactive_run(config_path: str, job_idx: int = 0) -> None:
    cfg_dict = load_config(config_path, job_idx=job_idx)
    cfg = config_to_ns(cfg_dict)
    setattr(cfg, "job_idx", job_idx)

    print("\nInteractive pipeline setup")
    overrides = {
        "model": _prompt_str("Model", _safe_get(cfg, "model", "mlp")),
        "dataset": _prompt_str("Dataset", _safe_get(cfg, "dataset", "cifar10_5k")),
        "iters": _prompt_int("Iters", _safe_get(cfg, "iters", 1)),
        "batch_size": _prompt_int("Batch size", _safe_get(cfg, "batch_size", None)),
        "num_workers": _prompt_int("Num workers", _safe_get(cfg, "num_workers", 2)),
        "hidden_dim": _prompt_float("Hidden dim", _safe_get(cfg, "hidden_dim", None)),
        "track_lipschitz": _prompt_bool("Track Lipschitz", _safe_get(cfg, "track_lipschitz", False)),
        "track_hessian": _prompt_bool("Track Hessian", _safe_get(cfg, "track_hessian", False)),
        "wandb_project_name": _prompt_str("W&B project (blank disables)", _safe_get(cfg, "wandb_project_name", "")),
    }

    if overrides["wandb_project_name"] == "":
        overrides["wandb_project_name"] = None

    cfg = _apply_overrides(cfg, overrides)

    max_train_batches = _prompt_int("Max train batches (blank=full)", None)
    max_val_batches = _prompt_int("Max val batches (blank=full)", None)
    max_test_batches = _prompt_int("Max test batches (blank=full)", None)

    run_pipeline(cfg, max_train_batches=max_train_batches,
                 max_val_batches=max_val_batches,
                 max_test_batches=max_test_batches)


def main(config_path: str, job_idx: int, interactive: bool) -> None:
    if interactive:
        interactive_run(config_path, job_idx)
        return

    cfg_dict = load_config(config_path, job_idx=job_idx)
    cfg = config_to_ns(cfg_dict)
    setattr(cfg, "job_idx", job_idx)
    run_pipeline(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive pipeline runner")
    parser.add_argument("--config", default="./config/config_inter.yaml")
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    main(args.config, args.job_idx, args.interactive)
