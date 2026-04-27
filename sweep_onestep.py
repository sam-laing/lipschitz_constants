"""
One-step Muon convergence experiment — init type sweep (mlp_ortho_v2).

For matrix quadratics f(W) = trace(W^T H W), ortho init + Muon gives one-step
convergence because polar(HW) = W when W is orthogonal.

Design: {kaiming, kaiming_orthog, orthogonal, perturbed} x {sgd, muon} x {linear, relu}
All runs use line search so found_lr traces local curvature each iteration.

Key observables:
  - val loss after iteration 1 (how much does ortho buy on the first step?)
  - found_lr trace (should be near-constant and near 1.0 for ortho+muon+linear)

Paste into Colab after pulling the repo to /content/lipschitz_constants.
"""

import json, os, sys, time
from types import SimpleNamespace

_HERE = "/content/lipschitz_constants"
sys.path.insert(0, _HERE)

import torch
from data import cifar10_5k_make_loaders
from engine import Engine
from models import build_model

# ── config ────────────────────────────────────────────────────────────────────
SEED        = 42
ITERS       = 50
DATA_ROOT   = None
ACTIVATIONS = ["linear", "relu"]
INITS       = ["kaiming", "kaiming_orthog", "orthogonal", "perturbed"]
OPTIMIZERS  = ["sgd", "muon"]
# scale used for orthogonal and perturbed modes — set to match kaiming scale
# for this network: sqrt(2) / sqrt(hidden_dim) ~ sqrt(2) / sqrt(4608) ~ 0.021
INIT_GAIN   = 0.02
PERTURB_R   = 0.1
OUT         = "/content/sweep_onestep.json"
DRIVE_SAVE  = "/content/drive/MyDrive/sweep_onestep.json"
# ─────────────────────────────────────────────────────────────────────────────


def make_cfg(optimizer, init_mode, activation):
    return SimpleNamespace(
        dataset="cifar10_5k", data_root=DATA_ROOT,
        num_workers=0, batch_size="full", seed=SEED,
        model="mlp_ortho_v2", hidden_dim=1.5, output_dim=5,
        activation=activation,
        use_bias=False, seperate_biases=False,
        init_mode=init_mode,
        init_gain=INIT_GAIN,
        init_nonlinearity=activation,   # matches activation so kaiming gain is correct
        perturb_radius=PERTURB_R,
        perturb_w_star_mode="kaiming",
        optimizer=optimizer, lr="line_search",
        momentum=0.0, nesterov=False, weight_decay=0.0,
        dual_decay=True, adjust_lr=False,
        orthogonalize=True, ns_steps=5,
        beta1=0.9, beta2=0.999, eps=1e-8,
        line_search_bracket=4.0,
        loss="cross_entropy",
        track_lipschitz=False, track_hessian=False,
        scheduler=None, wandb_project_name=None,
        iters=ITERS,
    )


def run_single(optimizer, init_mode, activation):
    cfg = make_cfg(optimizer, init_mode, activation)
    label = f"{optimizer} / {init_mode} / {activation}"
    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    torch.manual_seed(SEED)
    train_loader, val_loader, _ = cifar10_5k_make_loaders(cfg)
    torch.manual_seed(SEED)
    model = build_model(cfg)
    engine = Engine(model=model, cfg=cfg)

    rec = dict(
        optimizer=optimizer, init_mode=init_mode, activation=activation,
        train_loss=[], val_loss=[], train_acc=[], val_acc=[],
        found_lr=[], iter_time_s=[],
    )

    for i in range(ITERS):
        t0 = time.perf_counter()
        for x, y in train_loader:
            m = engine.step(x, y)
        v = engine.eval(val_loader)
        elapsed = time.perf_counter() - t0

        rec["train_loss"].append(m["loss"])
        rec["val_loss"].append(v["loss"])
        rec["train_acc"].append(m.get("accuracy"))
        rec["val_acc"].append(v.get("accuracy"))
        rec["found_lr"].append(m.get("lr_line_search"))
        rec["iter_time_s"].append(round(elapsed, 3))

        lr_str = f"  lr*={m['lr_line_search']:.4f}" if "lr_line_search" in m else ""
        print(f"  [{i+1:3d}/{ITERS}] train={m['loss']:.4f}  val={v['loss']:.4f}{lr_str}  ({elapsed:.1f}s)")

    return rec


def load_existing():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return []


def already_done(results, optimizer, init_mode, activation):
    return any(
        r["optimizer"] == optimizer
        and r["init_mode"] == init_mode
        and r["activation"] == activation
        for r in results
    )


# ── sweep ─────────────────────────────────────────────────────────────────────
results = load_existing()
if results:
    print(f"Resuming — {len(results)} runs already saved")

for activation in ACTIVATIONS:
    for init_mode in INITS:
        for optimizer in OPTIMIZERS:
            if already_done(results, optimizer, init_mode, activation):
                print(f"  skip: {optimizer} / {init_mode} / {activation}")
                continue
            rec = run_single(optimizer, init_mode, activation)
            results.append(rec)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  saved → {OUT}")

print(f"\nAll done. {len(results)} runs → {OUT}")

# ── save to Drive ─────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive")
    import shutil
    shutil.copy(OUT, DRIVE_SAVE)
    print(f"Saved to Drive: {DRIVE_SAVE}")
except Exception:
    pass


# ── plots ─────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

COLORS = {
    "kaiming":        "#888888",
    "kaiming_orthog": "#5b8dd9",
    "orthogonal":     "#e07b54",
    "perturbed":      "#6abf69",
}
LS = {"sgd": "--", "muon": "-"}
LW = {"sgd": 1.5,  "muon": 2.2}

by_key = {(r["optimizer"], r["init_mode"], r["activation"]): r for r in results}

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Init sweep — one-step Muon convergence analog (mlp_ortho_v2)", fontsize=13)

for col, act in enumerate(ACTIVATIONS):
    ax_loss = axes[0, col]
    ax_lr   = axes[1, col]

    ax_loss.set_title(f"activation = {act}", fontsize=11)
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("val loss")
    ax_lr.set_xlabel("iteration")
    ax_lr.set_ylabel("line-search LR found")
    ax_lr.axhline(1.0, color="gray", ls=":", lw=1, label="lr = 1 (one-step prediction)")

    for opt in OPTIMIZERS:
        for init in INITS:
            rec = by_key.get((opt, init, act))
            if rec is None:
                continue
            xs  = range(1, len(rec["val_loss"]) + 1)
            sty = dict(color=COLORS[init], ls=LS[opt], lw=LW[opt])
            ax_loss.plot(xs, rec["val_loss"], label=f"{opt} / {init}", **sty)
            ax_lr.plot(xs,   rec["found_lr"], label=f"{opt} / {init}", **sty)

    ax_loss.legend(fontsize=7, ncol=2)
    ax_lr.legend(fontsize=7, ncol=2)

    # inset: first 5 iterations — where the one-step claim lives
    axins = ax_loss.inset_axes([0.38, 0.38, 0.58, 0.58])
    for opt in OPTIMIZERS:
        for init in INITS:
            rec = by_key.get((opt, init, act))
            if rec is None:
                continue
            axins.plot(range(1, 6), rec["val_loss"][:5],
                       color=COLORS[init], ls=LS[opt], lw=LW[opt])
    axins.set_xticks([1, 2, 3, 4, 5])
    axins.set_title("iters 1–5", fontsize=7)
    ax_loss.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.5)

plt.tight_layout()
plt.savefig("/content/sweep_onestep.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → /content/sweep_onestep.png")
