import os
import argparse
from typing import Dict, Any, Tuple, Optional, List
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
from tqdm import tqdm

from cifar10_5k import make_loaders, CIFAR10_5k

class VAE(nn.Module):
    def __init__(self, input_dim: int = 3072, hidden_dims: List[int] = [512, 256], latent_dim: int = 64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE Trainer with detailed loss tracking
class VAETrainer:
    def __init__(self, model, train_loader, device, lr=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x.view(-1, 3072), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = MSE + KLD
        return total_loss, MSE, KLD
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_kld = 0
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss, mse, kld = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()
        
        dataset_size = len(self.train_loader.dataset)
        return (total_loss / dataset_size, 
                total_mse / dataset_size, 
                total_kld / dataset_size)
    
    def train(self, epochs):
        losses = []
        mse_losses = []
        kld_losses = []
        
        for epoch in range(epochs):
            total_loss, mse_loss, kld_loss = self.train_epoch()
            losses.append(total_loss)
            mse_losses.append(mse_loss)
            kld_losses.append(kld_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, '
                  f'Total Loss: {total_loss:.4f}, '
                  f'MSE: {mse_loss:.4f}, '
                  f'KLD: {kld_loss:.4f}')
        
        return losses, mse_losses, kld_losses

# Dimensionality Reduction Function
def reduce_dimensionality(model, dataloader, device, latent_dim: int):
    """Extract latent representations from VAE"""
    model.eval()
    all_latent = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            mu, logvar = model.encode(data)
            z = mu
            all_latent.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.vstack(all_latent), np.concatenate(all_labels)

# Main Pipeline
def run_vae_dimensionality_reduction(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure output directory exists before any savefig/save calls
    cfg.output_dir = os.path.expanduser(cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Get the original splits to access the data
    train_loader, val_loader, test_loader = make_loaders(cfg)
    
    # Create combined dataset (train + test)
    print("Creating combined dataset for VAE training...")
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914423, 0.48771504, 0.45364332],
                                        std=[0.24486475, 0.2414065, 0.26222563]), 
    ])
    
    train_dataset = CIFAR10_5k(train=True, transform=transform)
    test_dataset = CIFAR10_5k(train=False, transform=transform)
    
    # Combine train and test datasets
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    print(f"Combined dataset size: {len(combined_dataset)}")
    
    # Initialize VAE
    print(f"Initializing VAE with latent dimension: {cfg.latent_dim}")
    vae = VAE(input_dim=3072, hidden_dims=[512, 256], latent_dim=cfg.latent_dim)
    
    # Train VAE on combined dataset
    print("Training VAE on combined dataset...")
    trainer = VAETrainer(vae, combined_loader, device, lr=cfg.learning_rate)
    losses, mse_losses, kld_losses = trainer.train(epochs=cfg.epochs)
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(losses, label='Total Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Training Progress - Total Loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(mse_losses, label='Reconstruction Loss (MSE)', alpha=0.8)
    plt.plot(kld_losses, label='Regularization Loss (KLD)', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Training Progress - Loss Components')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Extract latent representations for each original split
    print("Extracting latent representations for original splits...")
    
    # Create individual loaders for each original split
    train_loader_ind = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    val_loader_ind = DataLoader(val_loader.dataset, batch_size=len(val_loader.dataset), shuffle=False)
    test_loader_ind = DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset), shuffle=False)
    
    # Get latent representations
    train_latent, train_labels = reduce_dimensionality(vae, train_loader_ind, device, cfg.latent_dim)
    val_latent, val_labels = reduce_dimensionality(vae, val_loader_ind, device, cfg.latent_dim)
    test_latent, test_labels = reduce_dimensionality(vae, test_loader_ind, device, cfg.latent_dim)
    
    # Save the reduced datasets
    print(f"Saving reduced datasets to {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    np.savez(os.path.join(cfg.output_dir, 'train_reduced.npz'),
             features=train_latent, labels=train_labels)
    np.savez(os.path.join(cfg.output_dir, 'val_reduced.npz'),
             features=val_latent, labels=val_labels)
    np.savez(os.path.join(cfg.output_dir, 'test_reduced.npz'),
             features=test_latent, labels=test_labels)
    
    # Save VAE model
    torch.save(vae.state_dict(), os.path.join(cfg.output_dir, 'vae_model.pth'))
    
    # Print statistics
    print(f"\nDimensionality Reduction Complete!")
    print(f"Original dimension: 3072")
    print(f"Reduced dimension: {cfg.latent_dim}")
    print(f"VAE trained on: {len(combined_dataset)} samples")
    print(f"Train set (original): {train_latent.shape}")
    print(f"Validation set: {val_latent.shape}")
    print(f"Test set: {test_latent.shape}")
    
    # Print final loss values
    print(f"\nFinal Loss Values:")
    print(f"Total Loss: {losses[-1]:.4f}")
    print(f"Reconstruction Loss (MSE): {mse_losses[-1]:.4f}")
    print(f"Regularization Loss (KLD): {kld_losses[-1]:.4f}")
    
    return train_latent, val_latent, test_latent

# Test function to verify the reduced datasets
def test_reduced_datasets(output_dir):
    """Test function to verify the saved reduced datasets"""
    print("\nTesting reduced datasets...")
    
    try:
        train_data = np.load(os.path.join(output_dir, 'train_reduced.npz'))
        train_features, train_labels = train_data['features'], train_data['labels']
        print(f"Train set - Features: {train_features.shape}, Labels: {train_labels.shape}")
        
        val_data = np.load(os.path.join(output_dir, 'val_reduced.npz'))
        val_features, val_labels = val_data['features'], val_data['labels']
        print(f"Val set - Features: {val_features.shape}, Labels: {val_labels.shape}")
        
        test_data = np.load(os.path.join(output_dir, 'test_reduced.npz'))
        test_features, test_labels = test_data['features'], test_data['labels']
        print(f"Test set - Features: {test_features.shape}, Labels: {test_labels.shape}")
        
        print(f"\nData verification:")
        print(f"Unique labels in train: {np.unique(train_labels)}")
        print(f"Unique labels in val: {np.unique(val_labels)}")
        print(f"Unique labels in test: {np.unique(test_labels)}")
        
        print(f"All datasets loaded successfully and verified!")
        
    except Exception as e:
        print(f"Error testing reduced datasets: {e}")

# Configuration class
class Config:
    def __init__(self):
        self.dataset = 'cifar10_5k'
        self.batch_size = 128
        self.num_workers = 4
        self.seed = 42
        self.latent_dim = 128
        self.epochs = 50
        self.learning_rate = 1e-3
        self.output_dir = f'/fast/slaing/data/vision/small_cifar/vae_reduced_data_dim={self.latent_dim}'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE Dimensionality Reduction')
    parser.add_argument('--latent_dim', type=int, default=64, help='Target latent dimension')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--test_only', action='store_true', help='Only test existing reduced datasets')
    
    args = parser.parse_args()
    
    cfg = Config()
    cfg.latent_dim = args.latent_dim
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
        cfg.output_dir = f'/fast/slaing/data/vision/small_cifar/vae_reduced_data_dim={cfg.latent_dim}'
    
    if args.test_only:
        test_reduced_datasets(cfg.output_dir)
    else:
        run_vae_dimensionality_reduction(cfg)
        test_reduced_datasets(cfg.output_dir)