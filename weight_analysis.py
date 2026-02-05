import torch 
import os
import matplotlib.pyplot as plt
import numpy as np


def get_svd(weight_matrix):
    """Perform SVD analysis on the given weight matrix."""
    U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    return U, S, Vh

def get_histogram(S, num_bins=50):
    """Get histogram data for singular values."""
    hist, bin_edges = torch.histogram(S, bins=num_bins)
    return hist, bin_edges

def analyze_singular_values(S, layer_name, optimizer_name):
    """Print statistics about singular values."""
    S_np = S.cpu().numpy()
    print(f"\n{optimizer_name} - {layer_name}:")
    print(f"  Max singular value: {S_np[0]:.6f}")
    print(f"  Min singular value: {S_np[-1]:.6f}")
    print(f"  Condition number: {S_np[0]/S_np[-1]:.6f}")
    print(f"  Mean: {S_np.mean():.6f}")
    print(f"  Std: {S_np.std():.6f}")
    print(f"  Spectral norm (Lipschitz): {S_np[0]:.6f}")

def plot_comparison(S1_sgd, S2_sgd, S1_muon, S2_muon, output_dir="./plots"):
    """Plot singular value comparison between SGD and Muon."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Layer 1 - Linear plot
    axes[0, 0].plot(S1_sgd.cpu().numpy(), marker='o', label='SGD', alpha=0.7)
    axes[0, 0].plot(S1_muon.cpu().numpy(), marker='s', label='Muon', alpha=0.7)
    axes[0, 0].set_title('Layer 1 (fc1) - Singular Values')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Singular Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Layer 1 - Log plot
    axes[0, 1].plot(S1_sgd.cpu().numpy(), marker='o', label='SGD', alpha=0.7)
    axes[0, 1].plot(S1_muon.cpu().numpy(), marker='s', label='Muon', alpha=0.7)
    axes[0, 1].set_title('Layer 1 (fc1) - Singular Values (Log Scale)')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Singular Value')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Layer 2 - Linear plot
    axes[1, 0].plot(S2_sgd.cpu().numpy(), marker='o', label='SGD', alpha=0.7)
    axes[1, 0].plot(S2_muon.cpu().numpy(), marker='s', label='Muon', alpha=0.7)
    axes[1, 0].set_title('Layer 2 (fc2) - Singular Values')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Singular Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Layer 2 - Log plot
    axes[1, 1].plot(S2_sgd.cpu().numpy(), marker='o', label='SGD', alpha=0.7)
    axes[1, 1].plot(S2_muon.cpu().numpy(), marker='s', label='Muon', alpha=0.7)
    axes[1, 1].set_title('Layer 2 (fc2) - Singular Values (Log Scale)')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Singular Value')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "sgd_vs_muon_singular_values.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()

def plot_histograms(S1_sgd, S2_sgd, S1_muon, S2_muon, output_dir="./plots"):
    """Plot histograms of singular value distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Layer 1 histogram
    axes[0].hist(S1_sgd.cpu().numpy(), bins=50, alpha=0.5, label='SGD', density=True)
    axes[0].hist(S1_muon.cpu().numpy(), bins=50, alpha=0.5, label='Muon', density=True)
    axes[0].set_title('Layer 1 (fc1) - Singular Value Distribution')
    axes[0].set_xlabel('Singular Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Layer 2 histogram
    axes[1].hist(S2_sgd.cpu().numpy(), bins=50, alpha=0.5, label='SGD', density=True)
    axes[1].hist(S2_muon.cpu().numpy(), bins=50, alpha=0.5, label='Muon', density=True)
    axes[1].set_title('Layer 2 (fc2) - Singular Value Distribution')
    axes[1].set_xlabel('Singular Value')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "sgd_vs_muon_histograms.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Histogram plot saved to: {save_path}")
    plt.close()


def parse_iteration_from_filename(filename):
    """Extract iteration number from filename like opt_muon_lr0.003_51_500_seed81.pt"""
    parts = filename.replace('.pt', '').split('_')
    # Find the iteration part (number before total iterations)
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
    
    # Find common iterations
    common_iters = sorted(set(muon_files.keys()) & set(sgd_files.keys()))
    
    return [(it, muon_files[it], sgd_files[it]) for it in common_iters]

def plot_iteration_comparison(iteration, S1_sgd, S2_sgd, S1_muon, S2_muon, output_dir="./plots"):
    """Plot comparative histograms for a single iteration."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Iteration {iteration}: Muon vs SGD Singular Value Comparison', fontsize=14)
    
    # Layer 1 (W1) histogram - step style for cleaner comparison
    axes[0, 0].hist(S1_sgd.cpu().numpy(), bins=30, alpha=0.7, label='SGD', color='blue', density=True, histtype='step', linewidth=2)
    axes[0, 0].hist(S1_muon.cpu().numpy(), bins=30, alpha=0.7, label='Muon', color='red', density=True, histtype='step', linewidth=2)
    axes[0, 0].set_title('W1 (fc1) - Singular Value Distribution')
    axes[0, 0].set_xlabel('Singular Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Layer 1 (W1) sorted singular values (log scale)
    axes[0, 1].plot(S1_sgd.cpu().numpy(), marker='o', markersize=3, label='SGD', alpha=0.7)
    axes[0, 1].plot(S1_muon.cpu().numpy(), marker='s', markersize=3, label='Muon', alpha=0.7)
    axes[0, 1].set_title('W1 (fc1) - Singular Values (log scale)')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Singular Value')
    #axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Layer 2 (W2) histogram - step style for cleaner comparison
    axes[1, 0].hist(S2_sgd.cpu().numpy(), bins=30, alpha=0.7, label='SGD', color='blue', density=True, histtype='step', linewidth=2)
    axes[1, 0].hist(S2_muon.cpu().numpy(), bins=30, alpha=0.7, label='Muon', color='red', density=True, histtype='step', linewidth=2)
    axes[1, 0].set_title('W2 (fc2) - Singular Value Distribution')
    axes[1, 0].set_xlabel('Singular Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Layer 2 (W2) sorted singular values (log scale)
    axes[1, 1].plot(S2_sgd.cpu().numpy(), marker='o', markersize=3, label='SGD', alpha=0.7)
    axes[1, 1].plot(S2_muon.cpu().numpy(), marker='s', markersize=3, label='Muon', alpha=0.7)
    axes[1, 1].set_title('W2 (fc2) - Singular Values (log scale)')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Singular Value')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"iter_{iteration}_muon_vs_sgd.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def load_weights_from_file(filepath, device):
    """Load weight matrices from a checkpoint file."""
    state_dict = torch.load(filepath, map_location=device)
    W1 = state_dict['fc1.weight']
    W2 = state_dict['fc2.weight']
    return W1, W2

if __name__ == "__main__":
    weight_dir = "/fast/slaing/converged_weights/exp/"
    output_dir = "./plots/iteration_comparison"
    seed = 81
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find matching files
    matching_files = get_matching_weight_files(weight_dir, seed=seed)
    print(f"Found {len(matching_files)} matching iterations:")
    for it, muon_f, sgd_f in matching_files:
        print(f"  Iteration {it}: {muon_f} vs {sgd_f}")
    
    # Process each iteration
    for iteration, muon_file, sgd_file in matching_files:
        print(f"\n{'='*60}")
        print(f"Processing iteration {iteration}")
        print('='*60)
        
        # Load weights
        muon_path = os.path.join(weight_dir, muon_file)
        sgd_path = os.path.join(weight_dir, sgd_file)
        
        W1_muon, W2_muon = load_weights_from_file(muon_path, device)
        W1_sgd, W2_sgd = load_weights_from_file(sgd_path, device)
        
        # Compute SVD
        _, S1_muon, _ = get_svd(W1_muon)
        _, S2_muon, _ = get_svd(W2_muon)
        _, S1_sgd, _ = get_svd(W1_sgd)
        _, S2_sgd, _ = get_svd(W2_sgd)
        
        # Print statistics
        analyze_singular_values(S1_sgd, "W1 (fc1)", "SGD")
        analyze_singular_values(S1_muon, "W1 (fc1)", "Muon")
        analyze_singular_values(S2_sgd, "W2 (fc2)", "SGD")
        analyze_singular_values(S2_muon, "W2 (fc2)", "Muon")
        
        # Lipschitz constants
        lipschitz_sgd = S1_sgd[0].item() * S2_sgd[0].item()
        lipschitz_muon = S1_muon[0].item() * S2_muon[0].item()
        print(f"\nLipschitz - SGD: {lipschitz_sgd:.4f}, Muon: {lipschitz_muon:.4f}, Ratio: {lipschitz_sgd/lipschitz_muon:.4f}")
        
        # Plot comparison
        plot_iteration_comparison(iteration, S1_sgd, S2_sgd, S1_muon, S2_muon, output_dir)
    
    print(f"\nAnalysis complete! Plots saved to {output_dir}")