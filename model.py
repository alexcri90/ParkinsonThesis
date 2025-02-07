"""
model.py

This file defines a semi-supervised VAE model for 3D DATSCAN images
and provides training and evaluation routines with progress bars and
checkpointing to enable pausing and resuming training.

Requirements:
- PyTorch
- tqdm
- nVidia GPU (e.g. 4070Ti) is recommended

Author: [Your Name]
Date: [Today's Date]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

###############################################################################
# Model Definition
###############################################################################

class SemiSupervisedVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=3):  # Directly use num_classes
        """
        Args:
            latent_dim (int): Dimension of the latent space.
            num_classes (int): Number of classes for classification (PD, SWEDD, Control).
        """
        super(SemiSupervisedVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # -------------------------
        # Encoder: 3D Convolutions
        # Input shape: (batch, 1, 128, 128, 128)
        # Output after conv5: (batch, 512, 4, 4, 4)
        # -------------------------
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)   # -> (32,64,64,64)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)  # -> (64,32,32,32)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1) # -> (128,16,16,16)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1) # -> (256,8,8,8)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1) # -> (512,4,4,4)

        # Flatten the output and compress to an intermediate vector.
        self.fc1 = nn.Linear(512 * 4 * 4 * 4, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # -------------------------
        # Decoder: Fully Connected + 3D Transposed Convolutions
        # -------------------------
        self.fc2 = nn.Linear(latent_dim, 1024)
        self.fc3 = nn.Linear(1024, 512 * 4 * 4 * 4)
        # The deconvolutions mirror the encoder:
        self.deconv1 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # -> (256,8,8,8)
        self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # -> (128,16,16,16)
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # -> (64,32,32,32)
        self.deconv4 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # -> (32,64,64,64)
        self.deconv5 = nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # -> (1,128,128,128)

        # -------------------------
        # Classification head
        # -------------------------
        self.classification_head = nn.Linear(latent_dim, num_classes) # Single classification head

    def encode(self, x):
        """Encodes the input into latent space parameters."""
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = h.view(h.size(0), -1)  # Flatten
        h = F.relu(self.fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Samples a latent vector using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decodes the latent vector back to the image space."""
        h = F.relu(self.fc2(z))
        h = F.relu(self.fc3(h))
        h = h.view(-1, 512, 4, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv4(h))
        recon = torch.sigmoid(self.deconv5(h))
        return recon

    def forward(self, x):
        """Performs a forward pass through the VAE.

        Returns:
            recon: Reconstructed image.
            mu: Mean of the latent Gaussian.
            logvar: Log variance of the latent Gaussian.
            class_pred: Prediction for patient group.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        class_pred = self.classification_head(z) # Single classification head
        return recon, mu, logvar, class_pred

    def get_latent(self, x):
        """Returns the latent vector (sampled) for a given input x."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

###############################################################################
# Loss Function
###############################################################################

def loss_function(recon, x, mu, logvar, class_pred, labels, beta=1.0, class_weight=1.0):
    """
    Computes the VAE loss.

    Args:
        recon: Reconstructed images.
        x: Original images.
        mu: Mean from the encoder.
        logvar: Log variance from the encoder.
        class_pred: Prediction for patient group.
        labels: Ground truth labels for patient group.
        beta (float): Weight for the KL divergence term.
        class_weight (float): Weight for the classification loss.

    Returns:
        total_loss: Combined loss.
        recon_loss: Reconstruction loss.
        kl_loss: KL divergence loss.
        class_loss: Classification loss.
    """
    # Reconstruction loss (using binary cross-entropy since the output is sigmoid-activated)
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Classification loss
    class_loss = F.cross_entropy(class_pred, labels, reduction='sum')

    total_loss = recon_loss + beta * kl_loss + class_weight * class_loss
    return total_loss, recon_loss, kl_loss, class_loss

###############################################################################
# Training, Evaluation, and Latent Extraction Functions
###############################################################################

def train_model(model, train_loader, optimizer, device, num_epochs, checkpoint_path,
                start_epoch=0, beta=1.0, class_weight=1.0):
    """
    Trains the model with progress bars and checkpointing.

    Args:
        model: The VAE model.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        device: Device to run training on.
        num_epochs (int): Number of epochs to train.
        checkpoint_path (str): File path to save checkpoints.
        start_epoch (int): Epoch number to resume from.
        beta (float): Weight for the KL divergence term.
        class_weight (float): Weight for the classification loss.

    Returns:
        history (dict): Dictionary containing epoch numbers and losses.
    """
    history = {'epoch': [], 'loss': [], 'recon_loss': [], 'kl_loss': [], 'class_loss': []}

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_class_loss = 0.0

        # Progress bar for the epoch
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, labels in train_loader:  # Expect batch to be (images, labels)
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                recon, mu, logvar, class_pred = model(images)
                loss, recon_loss, kl_loss, class_loss = loss_function(
                    recon, images, mu, logvar, class_pred, labels, beta, class_weight)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_recon_loss += recon_loss.item()
                running_kl_loss += kl_loss.item()
                running_class_loss += class_loss.item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.2f}",
                    "recon": f"{recon_loss.item():.2f}",
                    "KL": f"{kl_loss.item():.2f}",
                    "class": f"{class_loss.item():.2f}"
                })
                pbar.update(1)

        avg_loss = running_loss / len(train_loader.dataset)
        avg_recon_loss = running_recon_loss / len(train_loader.dataset)
        avg_kl_loss = running_kl_loss / len(train_loader.dataset)
        avg_class_loss = running_class_loss / len(train_loader.dataset)

        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['class_loss'].append(avg_class_loss)

        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Class: {avg_class_loss:.4f}")

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1, # Save next epoch number
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}\n")

    return history

def evaluate_model(model, val_loader, device, beta=1.0, class_weight=1.0):
    """
    Evaluates the model on a validation set.

    Args:
        model: The VAE model.
        val_loader: DataLoader for validation data.
        device: Device to run evaluation on.
        beta (float): Weight for the KL divergence term.
        class_weight (float): Weight for the classification loss.

    Returns:
        avg_loss (float): Average loss over the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            recon, mu, logvar, class_pred = model(images)
            loss, _, _, _ = loss_function(
                recon, images, mu, logvar, class_pred, labels, beta, class_weight)
            running_loss += loss.item()

    avg_loss = running_loss / len(val_loader.dataset)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def extract_latent_space(model, data_loader, device):
    """
    Extracts latent representations for the entire dataset.

    Args:
        model: The VAE model.
        data_loader: DataLoader for the dataset.
        device: Device to run extraction on.

    Returns:
        latent_space (torch.Tensor): Tensor containing latent vectors.
        labels (torch.Tensor): Tensor containing corresponding labels.
    """
    model.eval()
    latent_vectors = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting latent space", unit="batch"):
            images = images.to(device)
            z = model.get_latent(images)
            latent_vectors.append(z.cpu())
            all_labels.append(labels.cpu())

    latent_space = torch.cat(latent_vectors, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return latent_space, labels

###############################################################################
# Optional Testing Block
###############################################################################

if __name__ == '__main__':
    # Quick test to verify that the model runs on dummy data.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create a dummy input tensor: (batch_size, 1, 128, 128, 128)
    dummy_input = torch.randn(2, 1, 128, 128, 128).to(device)
    dummy_labels = torch.randint(0, 3, (2,)).to(device) # Dummy labels (0, 1, 2)

    # Instantiate the model.
    model = SemiSupervisedVAE(latent_dim=128, num_classes=3).to(device)

    # Forward pass
    recon, mu, logvar, class_pred = model(dummy_input)

    print("Output shapes:")
    print("  Reconstruction:", recon.shape)      # Expected: (2, 1, 128, 128, 128)
    print("  Mu:", mu.shape)                       # Expected: (2, 128)
    print("  Logvar:", logvar.shape)               # Expected: (2, 128)
    print("  Class predictions:", class_pred.shape) # Expected: (2, 3)