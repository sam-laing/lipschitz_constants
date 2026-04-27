"""
Local sweep: compares a grid of fixed LRs vs exact line search for GD and Muon.
Results saved to sweep_results.json after each run (safe to interrupt and resume).

Usage:
    conda run -n 310nets python3 sweep_local.py
"""
import json
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/content/lipschitz_constants"
sys.path.insert(0, _HERE)

import torch

from data import cifar10_5k_make_loaders
from engine import Engine
from models import build_model

# ── sweep config ──────────────────────────────────────────────────────────────
SEED      = 42
ITERS     = 100
DATA_ROOT = None  # set to e.g. "/content/cifar10_5k" if data isn't in the default location
LRS     = [5e-4, 1e-3, 2e-3, 5e-3, 3e-3, 1e-2, "line_search"]
OPTIMIZERS = ["sgd", "muon"]
OUT     = os.path.join(_HERE, "sweep_results.json")
# ─────────────────────────────────────────────────────────────────────────────


def base_cfg(optimizer: str, lr) -> SimpleNamespace:
    return SimpleNamespace(
        # data
        dataset="cifar10_5k",
        data_root=DATA_ROOT,
        num_workers=0,
        batch_size="full",
        seed=SEED,
        # model
        model="mlp_ortho",
        hidden_dim=1.5,
        output_dim=5,
        activation="relu",
        weight_init="normal",
        use_bias=True,
        seperate_biases=True,
        ortho_rank=None,
        # optimizer
        optimizer=optimizer,
        lr=lr,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        dual_decay=True,
        adjust_lr=False,
        # muon-specific
        orthogonalize=True,
        ns_steps=5,
        # adamw (used for biases with muon)
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        # line search
        line_search_bracket=4.0,
        # loss
        loss="cross_entropy",
        # tracking (off for speed)
        track_lipschitz=False,
        track_hessian=False,
        scheduler=None,
        wandb_project_name=None,
        iters=ITERS,
    )


def run_single(optimizer: str, lr) -> dict:
    cfg = base_cfg(optimizer, lr)
    lr_label = str(lr)

    print(f"\n{'='*60}")
    print(f"  optimizer={optimizer}  lr={lr_label}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    train_loader, val_loader, test_loader = cifar10_5k_make_loaders(cfg)

    torch.manual_seed(SEED)
    model = build_model(cfg)
    engine = Engine(model=model, cfg=cfg)

    result = {
        "optimizer": optimizer,
        "lr": lr_label,
        "iters": ITERS,
        "seed": SEED,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "found_lr": [],      # populated only when lr="line_search"
        "iter_time_s": [],
    }

    for i in range(ITERS):
        t0 = time.perf_counter()

        for x, y in train_loader:
            metrics = engine.step(x, y)

        val_metrics = engine.eval(val_loader)
        elapsed = time.perf_counter() - t0

        result["train_loss"].append(metrics["loss"])
        result["train_acc"].append(metrics["accuracy"])
        result["val_loss"].append(val_metrics["loss"])
        result["val_acc"].append(val_metrics["accuracy"])
        result["iter_time_s"].append(round(elapsed, 3))
        if "lr_line_search" in metrics:
            result["found_lr"].append(metrics["lr_line_search"])

        eta = elapsed * (ITERS - i - 1)
        print(
            f"  [{i+1:3d}/{ITERS}] train_loss={metrics['loss']:.4f}"
            f"  val_loss={val_metrics['loss']:.4f}"
            f"  acc={val_metrics['accuracy']:.3f}"
            + (f"  lr*={metrics['lr_line_search']:.4f}" if "lr_line_search" in metrics else "")
            + f"  ETA {eta:.0f}s"
        )

    test_metrics = engine.eval(test_loader)
    result["test_loss"] = test_metrics["loss"]
    result["test_acc"] = test_metrics["accuracy"]
    print(f"  → test_loss={test_metrics['loss']:.4f}  test_acc={test_metrics['accuracy']:.3f}")

    return result


def load_existing() -> list:
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return []


def already_done(results: list, optimizer: str, lr) -> bool:
    lr_label = str(lr)
    return any(r["optimizer"] == optimizer and r["lr"] == lr_label for r in results)


def main():
    results = load_existing()
    if results:
        print(f"Resuming — {len(results)} runs already in {OUT}")

    for optimizer in OPTIMIZERS:
        for lr in LRS:
            if already_done(results, optimizer, lr):
                print(f"  skipping {optimizer} lr={lr} (already done)")
                continue
            result = run_single(optimizer, lr)
            results.append(result)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  saved → {OUT}")

    print(f"\nDone. {len(results)} runs in {OUT}")


if __name__ == "__main__":
    main()

try:
    from google.colab import drive
    drive.mount('/content/drive')
    import shutil
    shutil.copy(OUT, '/content/drive/MyDrive/sweep_results.json')
    print("Saved to Google Drive: MyDrive/sweep_results.json")
except ImportError:
    pass
