"""
Exploratory Data Analysis module for DATSCAN images.
This module provides functionality to analyze and visualize DATSCAN images
and their associated metadata.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import Tensor
import pandas as pd

from data_loader import DATSCANMetadata, DATSCANDataLoader
from preprocessor import DATSCANPreprocessor, PreprocessingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DATSCANAnalyzer:
    """Class for analyzing DATSCAN images and metadata."""
    
    def __init__(self, 
                 save_dir: Union[str, Path],
                 preprocessor: Optional[DATSCANPreprocessor] = None):
        """
        Initialize analyzer.
        
        Args:
            save_dir: Directory to save analysis results
            preprocessor: Optional preprocessor for the images
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = preprocessor or DATSCANPreprocessor()
        
    def compute_image_statistics(self, 
                               images: List[Union[np.ndarray, Tensor]]) -> pd.DataFrame:
        """
        Compute basic statistics for a list of images.
        
        Args:
            images: List of image arrays or tensors
            
        Returns:
            DataFrame with image statistics
        """
        stats = []
        for i, img in enumerate(images):
            if isinstance(img, Tensor):
                img = img.cpu().numpy()
                
            stats.append({
                'image_idx': i,
                'mean': np.mean(img),
                'median': np.median(img),
                'std': np.std(img),
                'min': np.min(img),
                'max': np.max(img),
                'q1': np.percentile(img, 25),
                'q3': np.percentile(img, 75),
                'zero_fraction': np.mean(img == 0),
                'non_zero_voxels': np.sum(img != 0)
            })
            
        return pd.DataFrame(stats)
    
    def analyze_metadata(self, metadata_list: List[DATSCANMetadata]) -> pd.DataFrame:
        """
        Analyze metadata from DATSCAN images.
        
        Args:
            metadata_list: List of metadata objects
            
        Returns:
            DataFrame with metadata analysis
        """
        meta_data = []
        for meta in metadata_list:
            # Parse exam date
            try:
                exam_date = datetime.strptime(meta.exam_date.split('_')[0], '%Y-%m-%d')
            except:
                exam_date = None
                
            meta_data.append({
                'patient_id': meta.patient_id,
                'exam_date': exam_date,
                'exam_id': meta.exam_id,
                'image_shape': meta.image_shape,
                'pixel_spacing': meta.pixel_spacing
            })
            
        return pd.DataFrame(meta_data)
    
    def plot_intensity_histogram(self, 
                               images: List[Union[np.ndarray, Tensor]], 
                               n_samples: int = 5) -> None:
        """
        Plot intensity histograms for a subset of images.
        
        Args:
            images: List of images
            n_samples: Number of images to plot
        """
        plt.figure(figsize=(12, 6))
        for i in range(min(n_samples, len(images))):
            img = images[i]
            if isinstance(img, Tensor):
                img = img.cpu().numpy()
            
            plt.hist(img.ravel(), bins=50, alpha=0.5, label=f'Image {i}')
            
        plt.title('Intensity Distribution Across Images')
        plt.xlabel('Intensity Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(self.save_dir / 'intensity_histogram.png')
        plt.close()
        
    def plot_central_slices(self, 
                           image: Union[np.ndarray, Tensor],
                           title: str = "Central Slices") -> None:
        """
        Plot central slices of a 3D image in all three planes.
        
        Args:
            image: 3D image array or tensor
            title: Plot title
        """
        if isinstance(image, Tensor):
            image = image.cpu().numpy()
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get central slices
        x_center = image.shape[0] // 2
        y_center = image.shape[1] // 2
        z_center = image.shape[2] // 2
        
        # Plot axial, sagittal, and coronal views
        axes[0].imshow(image[x_center, :, :], cmap='gray')
        axes[0].set_title('Axial')
        
        axes[1].imshow(image[:, y_center, :], cmap='gray')
        axes[1].set_title('Sagittal')
        
        axes[2].imshow(image[:, :, z_center], cmap='gray')
        axes[2].set_title('Coronal')
        
        plt.suptitle(title)
        plt.savefig(self.save_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()
        
    def plot_exam_timeline(self, metadata_df: pd.DataFrame) -> None:
        """
        Plot timeline of exams.
        
        Args:
            metadata_df: DataFrame with metadata analysis
        """
        if 'exam_date' not in metadata_df.columns:
            logger.warning("No exam dates available for timeline plot")
            return
            
        plt.figure(figsize=(12, 6))
        metadata_df['exam_date'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribution of Exams Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Exams')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'exam_timeline.png')
        plt.close()
        
    def generate_summary_report(self, 
                              image_stats: pd.DataFrame, 
                              metadata_df: pd.DataFrame) -> str:
        """
        Generate a text summary of the analysis.
        
        Args:
            image_stats: DataFrame with image statistics
            metadata_df: DataFrame with metadata analysis
            
        Returns:
            Summary report as string
        """
        report = ["DATSCAN Analysis Summary Report", "=" * 30 + "\n"]
        
        # Image statistics summary
        report.append("Image Statistics:")
        report.append("-" * 20)
        report.append(f"Number of images: {len(image_stats)}")
        report.append(f"Mean intensity (avg across images): {image_stats['mean'].mean():.2f}")
        report.append(f"Std intensity (avg across images): {image_stats['std'].mean():.2f}")
        report.append(f"Average non-zero voxels: {image_stats['non_zero_voxels'].mean():.2f}")
        report.append("")
        
        # Metadata summary
        report.append("Metadata Summary:")
        report.append("-" * 20)
        report.append(f"Number of unique patients: {metadata_df['patient_id'].nunique()}")
        if 'exam_date' in metadata_df.columns:
            date_range = metadata_df['exam_date'].dropna()
            if not date_range.empty:
                report.append(f"Date range: {date_range.min().date()} to {date_range.max().date()}")
        
        return "\n".join(report)
    
    def run_full_analysis(self, 
                         images: List[Union[np.ndarray, Tensor]], 
                         metadata_list: List[DATSCANMetadata]) -> None:
        """
        Run complete analysis pipeline and save results.
        
        Args:
            images: List of images
            metadata_list: List of metadata objects
        """
        logger.info("Starting full analysis...")
        
        # Compute statistics
        image_stats = self.compute_image_statistics(images)
        metadata_df = self.analyze_metadata(metadata_list)
        
        # Generate plots
        self.plot_intensity_histogram(images)
        self.plot_exam_timeline(metadata_df)
        
        # Plot central slices for a few sample images
        for i in range(min(3, len(images))):
            self.plot_central_slices(images[i], f"Sample Image {i+1}")
        
        # Generate and save report
        report = self.generate_summary_report(image_stats, metadata_df)
        with open(self.save_dir / 'analysis_report.txt', 'w') as f:
            f.write(report)
            
        # Save statistics to CSV
        image_stats.to_csv(self.save_dir / 'image_statistics.csv', index=False)
        metadata_df.to_csv(self.save_dir / 'metadata_analysis.csv', index=False)
        
        logger.info(f"Analysis complete. Results saved to {self.save_dir}")

def main():
    """Example usage of the analyzer."""
    # Initialize data loader and load images
    base_path = Path("Images_Test/dicom")
    loader = DATSCANDataLoader(base_path)
    images, metadata = loader.load_all_data()
    
    # Initialize analyzer
    analyzer = DATSCANAnalyzer(save_dir="analysis_results")
    
    # Run analysis
    analyzer.run_full_analysis(images, list(metadata.values()))
    
if __name__ == "__main__":
    main()