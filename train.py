"""
Training script for VAE on MNIST
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from vae import VAE, vae_loss


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}: Average loss = {avg_loss:.4f}')
    return avg_loss


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss(recon_batch, data, mu, logvar).item()
    
    avg_loss = test_loss / len(test_loader.dataset)
    print(f'Test loss: {avg_loss:.4f}')
    return avg_loss


def main():
    # Configuration
    batch_size = 128
    epochs = 20
    latent_dim = 20
    hidden_dim = 400
    learning_rate = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # Model
    model = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f'\nModel architecture:')
    print(model)
    print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Training
    train_losses = []
    test_losses = []
    
    print('\n' + '='*50)
    print('Starting training...')
    print('='*50)
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch)
        test_loss = test(model, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'config': {
            'latent_dim': latent_dim,
            'hidden_dim': hidden_dim,
            'epochs': epochs
        }
    }, 'models/vae_mnist.pt')
    print('\nModel saved to models/vae_mnist.pt')
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training on MNIST')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_curve.png', dpi=150)
    print('Training curve saved to results/training_curve.png')
    

if __name__ == '__main__':
    main()
