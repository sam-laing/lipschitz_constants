import json
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 0.8,
})

# Shades per optimizer: [worst, 2nd-best, best]
# "best in middle" → best gets the mid-intensity shade (index 1)
GD_SHADES   = ["#aec7e8", "#1f77b4", "#08306b"]  # light, mid-blue, dark-blue
MUON_SHADES = ["#b5cfb5", "#2ca02c", "#004d00"]  # light, mid-green, dark-green


def load(path):
    with open(path) as f:
        return json.load(f)


EXCLUDE_LRS = {1e-2}  # LRs to skip when picking top 3

def top3_fixed(runs, loss_key):
    """Pick 3 best fixed-LR runs by final loss. Returns [worst, 2nd, best]."""
    fixed = [r for r in runs if r["lr"] != "line_search" and r[loss_key]
             and float(r["lr"]) not in EXCLUDE_LRS]
    ranked = sorted(fixed, key=lambda r: r[loss_key][-1])  # best first
    top = ranked[:3]
    # return worst→best so shades go light→mid→dark with best in middle
    # order: [worst, best, 2nd] → zips with shades [light, mid, dark]
    top_sorted_by_lr = sorted(top, key=lambda r: float(r["lr"]))
    return top_sorted_by_lr


def draw(ax, fixed_runs, shades, ls_run, loss_key, title):
    for run, color in zip(fixed_runs, shades):
        lr = float(run["lr"])
        ax.plot(run[loss_key], color=color, lw=1.8, label=f"lr = {lr:.0e}")

    if ls_run:
        ax.plot(ls_run[loss_key], color="black", lw=2.2,
                linestyle="--", label="Line search", zorder=5)

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_title(title)
    ax.grid(alpha=0.25, ls="--", which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, framealpha=0.9)


def plot_sweep(json_path, loss_key="train_loss", save_path=None):
    data = load(json_path)

    sgd_runs  = [r for r in data if r["optimizer"] == "sgd"]
    muon_runs = [r for r in data if r["optimizer"] == "muon"]

    sgd_ls   = next((r for r in sgd_runs  if r["lr"] == "line_search"), None)
    muon_ls  = next((r for r in muon_runs if r["lr"] == "line_search"), None)

    sgd_top  = top3_fixed(sgd_runs,  loss_key)
    muon_top = top3_fixed(muon_runs, loss_key)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    draw(ax_l, sgd_top,  GD_SHADES,   sgd_ls,  loss_key, "Gradient Descent")
    draw(ax_r, muon_top, MUON_SHADES, muon_ls, loss_key, "Muon")

    ax_l.set_ylabel("Train loss")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sweep_results.json"
    plot_sweep(path, save_path="sweep_plot.pdf")
