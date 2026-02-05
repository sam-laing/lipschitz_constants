import torch 
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


def get_svd(weight_matrix):
    """Perform SVD analysis on the given weight matrix."""
    U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    return U, S, Vh

def parse_iteration_from_filename(filename):
    """Extract iteration number from filename like opt_muon_lr0.003_51_500_seed81.pt"""
    parts = filename.replace('.pt', '').split('_')
    for i, part in enumerate(parts):
        if part.isdigit() and i+1 < len(parts) and parts[i+1].isdigit():
            return int(part)
    return None

def get_matching_weight_files(weight_dir, seed=81):
    """Find matching muon and sgd weight files at same iterations."""
    files = os.listdir(weight_dir)
    
    muon_files = {}
    sgd_files = {}
    
    for f in files:
        if f'seed{seed}' not in f:
            continue
        if f.startswith('opt_muon_lr'):
            iteration = parse_iteration_from_filename(f)
            if iteration is not None:
                muon_files[iteration] = f
        elif f.startswith('opt_sgd_lr'):
            iteration = parse_iteration_from_filename(f)
            if iteration is not None:
                sgd_files[iteration] = f
    
    common_iters = sorted(set(muon_files.keys()) & set(sgd_files.keys()))
    return [(it, muon_files[it], sgd_files[it]) for it in common_iters]

def load_weights_from_file(filepath, device):
    """Load weight matrices from a checkpoint file."""
    state_dict = torch.load(filepath, map_location=device)
    W1 = state_dict['fc1.weight']
    W2 = state_dict['fc2.weight']
    return W1, W2

def collect_all_singular_values(weight_dir, seed, device):
    """Collect singular values for all iterations."""
    matching_files = get_matching_weight_files(weight_dir, seed=seed)
    
    data = {
        'iterations': [],
        'S1_muon': [], 'S2_muon': [],
        'S1_sgd': [], 'S2_sgd': [],
    }
    
    for iteration, muon_file, sgd_file in matching_files:
        muon_path = os.path.join(weight_dir, muon_file)
        sgd_path = os.path.join(weight_dir, sgd_file)
        
        W1_muon, W2_muon = load_weights_from_file(muon_path, device)
        W1_sgd, W2_sgd = load_weights_from_file(sgd_path, device)
        
        _, S1_muon, _ = get_svd(W1_muon)
        _, S2_muon, _ = get_svd(W2_muon)
        _, S1_sgd, _ = get_svd(W1_sgd)
        _, S2_sgd, _ = get_svd(W2_sgd)
        
        data['iterations'].append(iteration)
        data['S1_muon'].append(S1_muon.cpu().numpy())
        data['S2_muon'].append(S2_muon.cpu().numpy())
        data['S1_sgd'].append(S1_sgd.cpu().numpy())
        data['S2_sgd'].append(S2_sgd.cpu().numpy())
    
    return data

def compute_divergence_metrics(data):
    """Compute various divergence metrics between muon and sgd over iterations."""
    metrics = {
        'iterations': data['iterations'],
        # Spectral norms (max singular value)
        'spectral_norm_muon_W1': [S[0] for S in data['S1_muon']],
        'spectral_norm_muon_W2': [S[0] for S in data['S2_muon']],
        'spectral_norm_sgd_W1': [S[0] for S in data['S1_sgd']],
        'spectral_norm_sgd_W2': [S[0] for S in data['S2_sgd']],
        # Condition numbers
        'cond_muon_W1': [S[0]/S[-1] for S in data['S1_muon']],
        'cond_muon_W2': [S[0]/S[-1] for S in data['S2_muon']],
        'cond_sgd_W1': [S[0]/S[-1] for S in data['S1_sgd']],
        'cond_sgd_W2': [S[0]/S[-1] for S in data['S2_sgd']],
        # Wasserstein distance between distributions
        'wasserstein_W1': [],
        'wasserstein_W2': [],
        # L2 distance between sorted SVs
        'l2_dist_W1': [],
        'l2_dist_W2': [],
        # Network Lipschitz constants
        'lipschitz_muon': [],
        'lipschitz_sgd': [],
    }
    
    for i in range(len(data['iterations'])):
        # Wasserstein distance
        metrics['wasserstein_W1'].append(
            wasserstein_distance(data['S1_muon'][i], data['S1_sgd'][i])
        )
        metrics['wasserstein_W2'].append(
            wasserstein_distance(data['S2_muon'][i], data['S2_sgd'][i])
        )
        # L2 distance
        metrics['l2_dist_W1'].append(
            np.linalg.norm(data['S1_muon'][i] - data['S1_sgd'][i])
        )
        metrics['l2_dist_W2'].append(
            np.linalg.norm(data['S2_muon'][i] - data['S2_sgd'][i])
        )
        # Lipschitz
        metrics['lipschitz_muon'].append(
            data['S1_muon'][i][0] * data['S2_muon'][i][0]
        )
        metrics['lipschitz_sgd'].append(
            data['S1_sgd'][i][0] * data['S2_sgd'][i][0]
        )
    
    return metrics

def plot_divergence_summary(metrics, output_dir="./plots"):
    """Plot a summary of how distributions diverge over iterations."""
    os.makedirs(output_dir, exist_ok=True)
    
    iters = metrics['iterations']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Muon vs SGD: Singular Value Evolution Over Training', fontsize=16)
    
    # 1. Spectral norms over iterations
    ax = axes[0, 0]
    ax.plot(iters, metrics['spectral_norm_sgd_W1'], 'b-o', label='SGD W1', markersize=5)
    ax.plot(iters, metrics['spectral_norm_muon_W1'], 'r-s', label='Muon W1', markersize=5)
    ax.plot(iters, metrics['spectral_norm_sgd_W2'], 'b--^', label='SGD W2', markersize=5)
    ax.plot(iters, metrics['spectral_norm_muon_W2'], 'r--d', label='Muon W2', markersize=5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Spectral Norm (σ_max)')
    ax.set_title('Spectral Norm Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Condition numbers over iterations
    ax = axes[0, 1]
    ax.plot(iters, metrics['cond_sgd_W1'], 'b-o', label='SGD W1', markersize=5)
    ax.plot(iters, metrics['cond_muon_W1'], 'r-s', label='Muon W1', markersize=5)
    ax.plot(iters, metrics['cond_sgd_W2'], 'b--^', label='SGD W2', markersize=5)
    ax.plot(iters, metrics['cond_muon_W2'], 'r--d', label='Muon W2', markersize=5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Condition Number (σ_max/σ_min)')
    ax.set_title('Condition Number Over Training')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Network Lipschitz constant
    ax = axes[0, 2]
    ax.plot(iters, metrics['lipschitz_sgd'], 'b-o', label='SGD', markersize=6)
    ax.plot(iters, metrics['lipschitz_muon'], 'r-s', label='Muon', markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Network Lipschitz (σ₁(W1) × σ₁(W2))')
    ax.set_title('Network Lipschitz Constant')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Wasserstein distance (divergence between optimizers)
    ax = axes[1, 0]
    ax.plot(iters, metrics['wasserstein_W1'], 'g-o', label='W1', markersize=6)
    ax.plot(iters, metrics['wasserstein_W2'], 'm-s', label='W2', markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('Distribution Divergence (Muon vs SGD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. L2 distance between sorted SVs
    ax = axes[1, 1]
    ax.plot(iters, metrics['l2_dist_W1'], 'g-o', label='W1', markersize=6)
    ax.plot(iters, metrics['l2_dist_W2'], 'm-s', label='W2', markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('L2 Distance')
    ax.set_title('L2 Distance Between Singular Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Ratio of Lipschitz constants
    ax = axes[1, 2]
    ratio = [s/m for s, m in zip(metrics['lipschitz_sgd'], metrics['lipschitz_muon'])]
    ax.plot(iters, ratio, 'k-o', markersize=6)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Lipschitz Ratio (SGD / Muon)')
    ax.set_title('Lipschitz Ratio Over Training')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "divergence_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_sv_heatmaps(data, output_dir="./plots"):
    """Plot heatmaps showing singular value evolution."""
    os.makedirs(output_dir, exist_ok=True)
    
    iters = data['iterations']
    
    # Stack singular values into 2D arrays (iterations x sv_index)
    S1_muon_arr = np.stack(data['S1_muon'])
    S1_sgd_arr = np.stack(data['S1_sgd'])
    S2_muon_arr = np.stack(data['S2_muon'])
    S2_sgd_arr = np.stack(data['S2_sgd'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Singular Value Heatmaps Over Training', fontsize=16)
    
    # W1 Muon
    im = axes[0, 0].imshow(S1_muon_arr, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('W1 - Muon')
    axes[0, 0].set_xlabel('Singular Value Index')
    axes[0, 0].set_ylabel('Iteration')
    axes[0, 0].set_yticks(range(len(iters)))
    axes[0, 0].set_yticklabels(iters)
    plt.colorbar(im, ax=axes[0, 0])
    
    # W1 SGD
    im = axes[0, 1].imshow(S1_sgd_arr, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('W1 - SGD')
    axes[0, 1].set_xlabel('Singular Value Index')
    axes[0, 1].set_ylabel('Iteration')
    axes[0, 1].set_yticks(range(len(iters)))
    axes[0, 1].set_yticklabels(iters)
    plt.colorbar(im, ax=axes[0, 1])
    
    # W1 Difference
    diff1 = S1_sgd_arr - S1_muon_arr
    vmax = max(abs(diff1.min()), abs(diff1.max()))
    im = axes[0, 2].imshow(diff1, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('W1 - Difference (SGD - Muon)')
    axes[0, 2].set_xlabel('Singular Value Index')
    axes[0, 2].set_ylabel('Iteration')
    axes[0, 2].set_yticks(range(len(iters)))
    axes[0, 2].set_yticklabels(iters)
    plt.colorbar(im, ax=axes[0, 2])
    
    # W2 Muon
    im = axes[1, 0].imshow(S2_muon_arr, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('W2 - Muon')
    axes[1, 0].set_xlabel('Singular Value Index')
    axes[1, 0].set_ylabel('Iteration')
    axes[1, 0].set_yticks(range(len(iters)))
    axes[1, 0].set_yticklabels(iters)
    plt.colorbar(im, ax=axes[1, 0])
    
    # W2 SGD
    im = axes[1, 1].imshow(S2_sgd_arr, aspect='auto', cmap='viridis')
    axes[1, 1].set_title('W2 - SGD')
    axes[1, 1].set_xlabel('Singular Value Index')
    axes[1, 1].set_ylabel('Iteration')
    axes[1, 1].set_yticks(range(len(iters)))
    axes[1, 1].set_yticklabels(iters)
    plt.colorbar(im, ax=axes[1, 1])
    
    # W2 Difference
    diff2 = S2_sgd_arr - S2_muon_arr
    vmax = max(abs(diff2.min()), abs(diff2.max()))
    im = axes[1, 2].imshow(diff2, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 2].set_title('W2 - Difference (SGD - Muon)')
    axes[1, 2].set_xlabel('Singular Value Index')
    axes[1, 2].set_ylabel('Iteration')
    axes[1, 2].set_yticks(range(len(iters)))
    axes[1, 2].set_yticklabels(iters)
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "sv_heatmaps.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_sv_waterfall(data, output_dir="./plots"):
    """Plot waterfall/ridge plots showing SV distribution evolution."""
    os.makedirs(output_dir, exist_ok=True)
    
    iters = data['iterations']
    n_iters = len(iters)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Singular Value Distribution Evolution (Waterfall Plot)', fontsize=14)
    
    # W1
    ax = axes[0]
    colors_muon = plt.cm.Reds(np.linspace(0.3, 1, n_iters))
    colors_sgd = plt.cm.Blues(np.linspace(0.3, 1, n_iters))
    
    for i, it in enumerate(iters):
        offset = i * 0.5  # vertical offset for each iteration
        ax.plot(data['S1_muon'][i] + offset, color=colors_muon[i], linewidth=1.5, label=f'Muon {it}' if i == n_iters-1 else '')
        ax.plot(data['S1_sgd'][i] + offset, color=colors_sgd[i], linewidth=1.5, linestyle='--', label=f'SGD {it}' if i == n_iters-1 else '')
    
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value (stacked)')
    ax.set_title('W1 (fc1) - Evolution')
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', label='Muon (light→dark = early→late)'),
        Line2D([0], [0], color='blue', linestyle='--', label='SGD (light→dark = early→late)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # W2
    ax = axes[1]
    for i, it in enumerate(iters):
        offset = i * 0.3
        ax.plot(data['S2_muon'][i] + offset, color=colors_muon[i], linewidth=1.5)
        ax.plot(data['S2_sgd'][i] + offset, color=colors_sgd[i], linewidth=1.5, linestyle='--')
    
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value (stacked)')
    ax.set_title('W2 (fc2) - Evolution')
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "sv_waterfall.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_top_svs_over_time(data, top_k=10, output_dir="./plots"):
    """Plot the top-k singular values over iterations for both optimizers."""
    os.makedirs(output_dir, exist_ok=True)
    
    iters = data['iterations']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Top {top_k} Singular Values Over Training', fontsize=14)
    
    # W1 - Muon
    ax = axes[0, 0]
    for k in range(top_k):
        vals = [data['S1_muon'][i][k] for i in range(len(iters))]
        ax.plot(iters, vals, marker='o', markersize=4, label=f'σ_{k+1}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Singular Value')
    ax.set_title('W1 - Muon')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # W1 - SGD
    ax = axes[0, 1]
    for k in range(top_k):
        vals = [data['S1_sgd'][i][k] for i in range(len(iters))]
        ax.plot(iters, vals, marker='o', markersize=4, label=f'σ_{k+1}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Singular Value')
    ax.set_title('W1 - SGD')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # W2 - Muon
    ax = axes[1, 0]
    for k in range(min(top_k, len(data['S2_muon'][0]))):
        vals = [data['S2_muon'][i][k] for i in range(len(iters))]
        ax.plot(iters, vals, marker='o', markersize=4, label=f'σ_{k+1}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Singular Value')
    ax.set_title('W2 - Muon')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # W2 - SGD
    ax = axes[1, 1]
    for k in range(min(top_k, len(data['S2_sgd'][0]))):
        vals = [data['S2_sgd'][i][k] for i in range(len(iters))]
        ax.plot(iters, vals, marker='o', markersize=4, label=f'σ_{k+1}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Singular Value')
    ax.set_title('W2 - SGD')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"top_{top_k}_svs_over_time.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    weight_dir = "/fast/slaing/converged_weights/exp/"
    output_dir = "./plots/iteration_comparison"
    seed = 81
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Collecting singular values from all checkpoints...")
    data = collect_all_singular_values(weight_dir, seed, device)
    print(f"Found {len(data['iterations'])} iterations: {data['iterations']}")
    
    print("\nComputing divergence metrics...")
    metrics = compute_divergence_metrics(data)
    
    print("\nGenerating plots...")
    plot_divergence_summary(metrics, output_dir)
    plot_sv_heatmaps(data, output_dir)
    plot_sv_waterfall(data, output_dir)
    plot_top_svs_over_time(data, top_k=10, output_dir=output_dir)
    
    print(f"\nAll plots saved to {output_dir}")
