import json
import os
import matplotlib.pyplot as plt

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = "/content/lipschitz_constants"

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

NORMAL_SHADES = ["#b5cfb5", "#2ca02c", "#004d00"]
ORTHO_SHADES  = ["#b5cfb5", "#2ca02c", "#004d00"]

EXCLUDE_LRS = {1e-2}


def load(path):
    with open(path) as f:
        return json.load(f)


def top3_fixed(runs, loss_key):
    fixed = [r for r in runs
             if r["lr"] != "line_search"
             and r[loss_key]
             and float(r["lr"]) not in EXCLUDE_LRS]
    ranked = sorted(fixed, key=lambda r: r[loss_key][-1])
    top = ranked[:3]
    return sorted(top, key=lambda r: float(r["lr"]))


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


def plot_sweep_init(json_path, loss_key="train_loss", save_path=None):
    data = load(json_path)

    normal_runs = [r for r in data if r["weight_init"] == "normal"]
    ortho_runs  = [r for r in data if r["weight_init"] == "full_orthogonal"]

    normal_ls = next((r for r in normal_runs if r["lr"] == "line_search"), None)
    ortho_ls  = next((r for r in ortho_runs  if r["lr"] == "line_search"), None)

    normal_top = top3_fixed(normal_runs, loss_key)
    ortho_top  = top3_fixed(ortho_runs,  loss_key)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    draw(ax_l, normal_top, NORMAL_SHADES, normal_ls, loss_key, "Normal (Kaiming) init")
    draw(ax_r, ortho_top,  ORTHO_SHADES,  ortho_ls,  loss_key, "Orthogonal init")

    ax_l.set_ylabel("Train loss")

    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved → {save_path}")
    plt.show()


def plot_found_lr(json_path, save_path=None):
    data = load(json_path)

    normal_ls = next((r for r in data
                      if r["weight_init"] == "normal" and r["lr"] == "line_search"), None)
    ortho_ls  = next((r for r in data
                      if r["weight_init"] == "full_orthogonal" and r["lr"] == "line_search"), None)

    fig, ax = plt.subplots(figsize=(7, 4))

    if normal_ls and normal_ls["found_lr"]:
        ax.plot(normal_ls["found_lr"], color="#d63fa0", lw=2.0, label="Normal (Kaiming)")
    if ortho_ls and ortho_ls["found_lr"]:
        ax.plot(ortho_ls["found_lr"],  color="#7b2d8b", lw=2.0, label="Orthogonal")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Line-search LR")
    ax.set_title("Greedy line-search LR: normal vs orthogonal init")
    ax.grid(alpha=0.25, ls="--", which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    OUT = os.path.join(_HERE, "sweep_init_results.json")
    IMG = os.path.join(_HERE, "sweep_init_plot.png")
    plot_sweep_init(OUT, loss_key="train_loss", save_path=IMG)
    plot_found_lr(OUT, save_path=os.path.join(_HERE, "sweep_init_found_lr.png"))