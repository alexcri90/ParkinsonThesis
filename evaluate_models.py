# Cell 15: Evaluating Trained Models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import gc

# Import project modules
from models.autoencoder import Autoencoder
from dataset import DATSCANDataset
from trainers.ae_trainer import AutoencoderTrainer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv("validated_file_paths.csv")
dataset = DATSCANDataset(df['file_path'].tolist(), df['label'].tolist())

# Create smaller batch size for evaluation to avoid memory issues
BATCH_SIZE = 2
NUM_WORKERS = 4

# Split dataset
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Create test loader
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Load model
model_name = "autoencoder"  # Change to the model you want to evaluate
model_dir = Path(f"trained_models/{model_name}")

# Load configuration
with open(model_dir / "config.json", "r") as f:
    config = json.load(f)

# Create model
model = Autoencoder(
    latent_dim=config["model"]["latent_dim"],
    name=model_name
).to(device)

# Load best model
checkpoint_path = model_dir / "checkpoints" / f"{model_name}_best.pt"
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
else:
    print(f"No checkpoint found at {checkpoint_path}")

# Set model to evaluation mode
model.eval()

# Create visualizations for reconstructions
def visualize_reconstructions(loader, num_samples=5):
    # Get random samples
    all_data = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in loader:
            all_data.append(data)
            all_labels.append(labels)
            if len(all_data) * loader.batch_size >= num_samples:
                break
    
    # Concatenate data
    data = torch.cat(all_data, dim=0)[:num_samples]
    
    # Move to device
    data = data.to(device)
    
    # Get reconstructions
    reconstructions, _ = model(data)
    
    # Plot
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    
    for i in range(num_samples):
        # Get middle slice
        slice_idx = data.size(2) // 2
        
        # Plot original
        axes[i, 0].imshow(data[i, 0, slice_idx].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f'Original {i+1}')
        axes[i, 0].axis('off')
        
        # Plot reconstruction
        axes[i, 1].imshow(reconstructions[i, 0, slice_idx].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title(f'Reconstruction {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Visualize reconstructions
fig = visualize_reconstructions(test_loader, num_samples=3)

# Extract and visualize latent space
def visualize_latent_space(loader, label_map={0: 'Control', 1: 'PD', 2: 'SWEDD'}):
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            _, latent = model(data)
            latent_vectors.append(latent.cpu().numpy())
            labels.extend(label.numpy())
    
    # Concatenate all latent vectors
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # Apply dimensionality reduction
    if latent_vectors.shape[1] > 50:
        # First reduce with PCA if latent space is very high-dimensional
        pca = PCA(n_components=50)
        latent_vectors = pca.fit_transform(latent_vectors)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': latent_2d[:, 0],
        'y': latent_2d[:, 1],
        'label': [label_map.get(l, str(l)) for l in labels]
    })
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='viridis', s=100, alpha=0.7)
    plt.title(f'{model_name} Latent Space Visualization (t-SNE)')
    plt.tight_layout()
    plt.show()
    
    return df

# Visualize latent space
latent_df = visualize_latent_space(test_loader)

# Extract and analyze latent vectors for clinical groups
def analyze_latent_space(loader):
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            _, latent = model(data)
            latent_vectors.append(latent.cpu().numpy())
            labels.extend(label.numpy())
    
    # Concatenate all latent vectors
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.array(labels)
    
    # Group by label
    label_map = {0: 'Control', 1: 'PD', 2: 'SWEDD'}
    groups = {}
    
    for label_id, label_name in label_map.items():
        mask = labels == label_id
        if np.any(mask):
            groups[label_name] = latent_vectors[mask]
    
    # Compute statistics
    stats = {}
    for group_name, vectors in groups.items():
        stats[group_name] = {
            'mean': np.mean(vectors, axis=0),
            'std': np.std(vectors, axis=0),
            'min': np.min(vectors, axis=0),
            'max': np.max(vectors, axis=0)
        }
    
    # Plot mean activation per dimension for each group
    num_dims = min(10, latent_vectors.shape[1])  # Show first 10 dimensions at most
    
    plt.figure(figsize=(12, 6))
    x = np.arange(num_dims)
    width = 0.2
    
    for i, (group_name, group_stats) in enumerate(stats.items()):
        plt.bar(x + i*width, group_stats['mean'][:num_dims], width, 
                label=group_name, alpha=0.7)
        
    plt.xlabel('Latent Dimension')
    plt.ylabel('Mean Activation')
    plt.title('Mean Latent Activations by Group')
    plt.xticks(x + width, [f'Dim {i}' for i in range(num_dims)])
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return stats, groups

# Analyze latent space
stats, groups = analyze_latent_space(test_loader)

# Cleanup
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
plt.tight_layout()
    plt.show()
    
    # Plot variance per dimension
    plt.figure(figsize=(12, 6))
    
    for i, (group_name, group_stats) in enumerate(stats.items()):
        plt.bar(x + i*width, group_stats['std'][:num_dims]**2, width, 
                label=group_name, alpha=0.7)
        
    plt.xlabel('Latent Dimension')
    plt.ylabel('Variance')
    plt.title('Latent Dimension Variance by Group')
    plt.xticks(x + width, [f'Dim {i}' for i in range(num_dims)])
    plt.legend()