# evaluate.py
import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from dataset import MemoryEfficientDATSCANDataset
from torch.utils.data import DataLoader
from autoencoder import Autoencoder, StriatalMSELoss
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import gc

def clear_memory():
    """Clear memory and cache to free up resources."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (allocated), "
              f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB (reserved)")

def visualize_reconstructions(model, dataloader, device, num_samples=3, output_dir=None):
    """Visualize original vs reconstructed volumes for a few samples."""
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            volumes = batch['volume'].to(device)
            reconstructions, _ = model(volumes)
            
            for idx in range(min(num_samples, volumes.shape[0])):
                fig = plt.figure(figsize=(15, 10))
                
                # Get the original and reconstructed volumes
                orig_vol = volumes[idx, 0].cpu().numpy()
                recon_vol = reconstructions[idx, 0].cpu().numpy()
                
                # Determine value range for consistent display
                vmin = min(orig_vol.min(), recon_vol.min())
                vmax = max(orig_vol.max(), recon_vol.max())
                
                # Plot in three orientations (axial, coronal, sagittal)
                views = ['Axial', 'Coronal', 'Sagittal']
                
                # Get middle slices
                axial_idx = orig_vol.shape[0] // 2
                coronal_idx = orig_vol.shape[1] // 2
                sagittal_idx = orig_vol.shape[2] // 2
                
                # Get slices
                orig_slices = [
                    orig_vol[axial_idx, :, :],      # Axial
                    orig_vol[:, coronal_idx, :],    # Coronal
                    orig_vol[:, :, sagittal_idx]    # Sagittal
                ]
                
                recon_slices = [
                    recon_vol[axial_idx, :, :],     # Axial
                    recon_vol[:, coronal_idx, :],   # Coronal
                    recon_vol[:, :, sagittal_idx]   # Sagittal
                ]
                
                # Compute difference maps
                diff_slices = [
                    np.abs(orig - recon) for orig, recon in zip(orig_slices, recon_slices)
                ]
                
                # Plot each orientation
                for i, (view, orig, recon, diff) in enumerate(zip(views, orig_slices, recon_slices, diff_slices)):
                    # Original
                    ax1 = plt.subplot(3, 3, i*3 + 1)
                    im1 = ax1.imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
                    ax1.set_title(f'Original {view}')
                    ax1.axis('off')
                    
                    # Reconstructed
                    ax2 = plt.subplot(3, 3, i*3 + 2)
                    im2 = ax2.imshow(recon, cmap='gray', vmin=vmin, vmax=vmax)
                    ax2.set_title(f'Reconstructed {view}')
                    ax2.axis('off')
                    
                    # Difference map
                    ax3 = plt.subplot(3, 3, i*3 + 3)
                    im3 = ax3.imshow(diff, cmap='hot', vmin=0, vmax=diff.max())
                    ax3.set_title(f'Difference {view}')
                    ax3.axis('off')
                    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                
                # Save or show the figure
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(os.path.join(output_dir, f'reconstruction_sample_{idx+1}.png'))
                    plt.close()
                else:
                    plt.show()
            
            # Only process the first batch
            break

def analyze_latent_space(model, dataloader, device, output_dir=None):
    """Analyze the latent space representations."""
    model.eval()
    
    all_latents = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding samples"):
            volumes = batch['volume'].to(device)
            labels = batch['label']
            paths = batch['path']
            
            # Encode to latent space
            latents = model.encode(volumes)
            
            # Store results
            all_latents.append(latents.cpu().numpy())
            all_labels.extend(labels)
            all_paths.extend(paths)
            
            # Clear memory
            del volumes, latents
            clear_memory()
    
    # Concatenate all latent vectors
    all_latents = np.vstack(all_latents)
    
    # Create dataframe for easy analysis
    latent_df = pd.DataFrame({
        'label': all_labels,
        'path': all_paths
    })
    
    # Add latent dimensions as columns
    for i in range(all_latents.shape[1]):
        latent_df[f'latent_{i}'] = all_latents[:, i]
    
    # Save latent data
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        latent_df.to_csv(os.path.join(output_dir, 'latent_vectors.csv'), index=False)
    
    # Visualize latent space with dimensionality reduction
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_latents)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_latents)-1))
    tsne_result = tsne.fit_transform(all_latents)
    
    # Create plots
    plt.figure(figsize=(20, 8))
    
    # PCA plot
    plt.subplot(1, 2, 1)
    scatter_pca = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=[label_to_color(l) for l in all_labels], alpha=0.7)
    plt.title('PCA of Latent Space')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Create legend
    unique_labels = list(set(all_labels))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color(l), 
                          markersize=10, label=l) for l in unique_labels]
    plt.legend(handles=handles, title='Group')
    
    # t-SNE plot
    plt.subplot(1, 2, 2)
    scatter_tsne = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=[label_to_color(l) for l in all_labels], alpha=0.7)
    plt.title('t-SNE of Latent Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color(l), 
                          markersize=10, label=l) for l in unique_labels]
    plt.legend(handles=handles, title='Group')
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'latent_space_visualization.png'))
        plt.close()
    else:
        plt.show()
    
    # Return the latent data for further analysis
    return latent_df

def label_to_color(label):
    """Convert a label to a color for visualization."""
    color_map = {
        'PD': 'red',
        'Control': 'blue',
        'SWEDD': 'green'
    }
    return color_map.get(label, 'gray')

def find_outliers(model, dataloader, device, output_dir=None):
    """Find outliers in the dataset based on reconstruction error."""
    model.eval()
    
    reconstruction_errors = {}
    striatal_errors = {}
    
    striatal_criterion = StriatalMSELoss().to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Finding outliers"):
            volumes = batch['volume'].to(device)
            paths = batch['path']
            
            # Forward pass
            reconstructions, _ = model(volumes)
            
            # Calculate reconstruction errors
            errors = F.mse_loss(reconstructions, volumes, reduction='none')
            errors = errors.view(errors.size(0), -1).mean(dim=1)
            
            # Calculate striatal-specific errors
            striatal_errs = striatal_criterion(reconstructions, volumes).view(-1)
            
            # Store errors
            for path, error, striatal_err in zip(paths, errors.cpu().numpy(), striatal_errs.cpu().numpy()):
                reconstruction_errors[path] = error.item()
                striatal_errors[path] = striatal_err.item()
            
            # Clear memory
            del volumes, reconstructions, errors
            clear_memory()
    
    # Convert to dataframe
    error_df = pd.DataFrame({
        'path': list(reconstruction_errors.keys()),
        'reconstruction_error': list(reconstruction_errors.values()),
        'striatal_error': list(striatal_errors.values())
    })
    
    # Extract labels from paths
    error_df['label'] = error_df['path'].apply(extract_label_from_path)
    
    # Sort by reconstruction error
    error_df = error_df.sort_values('reconstruction_error', ascending=False)
    
    # Identify outliers (e.g., errors > 3 standard deviations)
    mean_error = error_df['reconstruction_error'].mean()
    std_error = error_df['reconstruction_error'].std()
    error_df['is_outlier'] = error_df['reconstruction_error'] > (mean_error + 3 * std_error)
    
    # Save outlier data
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        error_df.to_csv(os.path.join(output_dir, 'reconstruction_errors.csv'), index=False)
    
    # Plot error distributions
    plt.figure(figsize=(20, 8))
    
    # Overall error distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=error_df, x='reconstruction_error', hue='label', element='step', kde=True)
    plt.axvline(x=mean_error + 3 * std_error, color='red', linestyle='--', 
                label=f'Outlier threshold ({mean_error + 3 * std_error:.4f})')
    plt.title('Distribution of Reconstruction Errors')
    plt.xlabel('MSE Reconstruction Error')
    plt.ylabel('Count')
    plt.legend()
    
    # Striatal error distribution
    plt.subplot(1, 2, 2)
    sns.histplot(data=error_df, x='striatal_error', hue='label', element='step', kde=True)
    plt.title('Distribution of Striatal Region Errors')
    plt.xlabel('MSE Striatal Error')
    plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'error_distributions.png'))
        plt.close()
    else:
        plt.show()
    
    return error_df

def extract_label_from_path(path):
    """Extract label from a file path."""
    if 'PPMI_Images_PD' in path:
        return 'PD'
    elif 'PPMI_Images_SWEDD' in path:
        return 'SWEDD'
    elif 'PPMI_Images_Cont' in path:
        return 'Control'
    else:
        return 'Unknown'

def main():
    """Main function to evaluate a trained model."""
    parser = argparse.ArgumentParser(description="Evaluate a trained autoencoder model")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data_csv', type=str, required=True,
                        help='Path to CSV file with file paths and labels')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for caching processed volumes (optional)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Load model
    latent_dim = checkpoint.get('latent_dim', 128)
    model = Autoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model with latent dimension {latent_dim}")
    
    # Load data
    df = pd.read_csv(args.data_csv)
    print(f"Loaded data CSV with {len(df)} entries")
    
    # Create dataset and dataloader
    dataset = MemoryEfficientDATSCANDataset(df, cache_dir=args.cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Created dataloader with batch size {args.batch_size}")
    
    # Create results subdirectories
    reconstructions_dir = os.path.join(args.output_dir, 'reconstructions')
    latent_dir = os.path.join(args.output_dir, 'latent_analysis')
    outliers_dir = os.path.join(args.output_dir, 'outliers')
    
    # Run evaluation
    print("Visualizing reconstructions...")
    visualize_reconstructions(model, dataloader, device, num_samples=5, output_dir=reconstructions_dir)
    
    print("Analyzing latent space...")
    latent_df = analyze_latent_space(model, dataloader, device, output_dir=latent_dir)
    
    print("Finding outliers...")
    error_df = find_outliers(model, dataloader, device, output_dir=outliers_dir)
    
    print(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()