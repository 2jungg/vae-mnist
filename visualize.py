"""
Visualization script for trained VAE
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from vae import VAE


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = VAE(
        input_dim=784,
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def visualize_reconstruction(model, test_loader, device, num_images=10):
    """Compare original vs reconstructed images."""
    model.eval()
    
    # Get a batch of test images
    data, labels = next(iter(test_loader))
    data = data[:num_images].to(device)
    
    with torch.no_grad():
        recon, _, _ = model(data)
    
    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', loc='left', fontsize=12)
        
        # Reconstructed
        axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', loc='left', fontsize=12)
    
    plt.suptitle('Original vs Reconstructed MNIST Images', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/reconstruction.png', dpi=150)
    print('Saved: results/reconstruction.png')


def visualize_latent_space(model, test_loader, device, num_samples=5000):
    """Visualize the latent space using t-SNE."""
    model.eval()
    
    all_mu = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if sum(len(l) for l in all_labels) >= num_samples:
                break
    
    all_mu = np.concatenate(all_mu, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]
    
    print(f'Running t-SNE on {len(all_mu)} samples...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(all_mu)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=all_labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('VAE Latent Space (t-SNE Visualization)')
    plt.tight_layout()
    plt.savefig('results/latent_space_tsne.png', dpi=150)
    print('Saved: results/latent_space_tsne.png')


def visualize_latent_2d(model, test_loader, device, num_samples=5000):
    """If latent_dim >= 2, visualize first two dimensions directly."""
    model.eval()
    
    all_mu = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if sum(len(l) for l in all_labels) >= num_samples:
                break
    
    all_mu = np.concatenate(all_mu, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]
    
    # Plot first 2 dimensions
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], 
                         c=all_labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space (First 2 Dimensions)')
    plt.tight_layout()
    plt.savefig('results/latent_space_2d.png', dpi=150)
    print('Saved: results/latent_space_2d.png')


def generate_samples(model, device, num_samples=100):
    """Generate new samples from the latent space."""
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(num_samples, device)
    
    # Plot grid
    n = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            axes[i, j].imshow(samples[idx].cpu().view(28, 28), cmap='gray')
            axes[i, j].axis('off')
    
    plt.suptitle('Generated Samples from VAE', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/generated_samples.png', dpi=150)
    print('Saved: results/generated_samples.png')


def interpolate_latent(model, test_loader, device):
    """Interpolate between two digits in latent space."""
    model.eval()
    
    # Get two different digits
    data, labels = next(iter(test_loader))
    
    # Find index of digit 0 and digit 9
    idx0 = (labels == 0).nonzero(as_tuple=True)[0][0]
    idx9 = (labels == 9).nonzero(as_tuple=True)[0][0]
    
    img0 = data[idx0:idx0+1].to(device)
    img9 = data[idx9:idx9+1].to(device)
    
    with torch.no_grad():
        mu0, _ = model.encode(img0.view(-1, 784))
        mu9, _ = model.encode(img9.view(-1, 784))
        
        # Interpolate
        n_steps = 10
        interpolations = []
        for alpha in np.linspace(0, 1, n_steps):
            z = (1 - alpha) * mu0 + alpha * mu9
            recon = model.decode(z)
            interpolations.append(recon)
    
    # Plot
    fig, axes = plt.subplots(1, n_steps, figsize=(15, 2))
    for i, recon in enumerate(interpolations):
        axes[i].imshow(recon.cpu().view(28, 28), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{i/(n_steps-1):.1f}')
    
    plt.suptitle('Latent Space Interpolation (0 → 9)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/interpolation.png', dpi=150)
    print('Saved: results/interpolation.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model, config = load_model('models/vae_mnist.pt', device)
    print(f'Loaded model with config: {config}')
    
    # Load test data
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    
    print('\n' + '='*50)
    print('Generating visualizations...')
    print('='*50 + '\n')
    
    # Generate visualizations
    visualize_reconstruction(model, test_loader, device)
    visualize_latent_2d(model, test_loader, device)
    visualize_latent_space(model, test_loader, device)
    generate_samples(model, device)
    interpolate_latent(model, test_loader, device)
    
    print('\nAll visualizations complete!')


if __name__ == '__main__':
    main()
