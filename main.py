"""
Main script for DATSCAN image analysis pipeline.
This script integrates data loading, preprocessing, and analysis modules.
"""

import logging
from pathlib import Path

from data_loader import DATSCANDataLoader
from preprocessor import DATSCANPreprocessor, PreprocessingParams
from eda import DATSCANAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Run the complete analysis pipeline:
    1. Load DATSCAN images
    2. Preprocess the images
    3. Perform exploratory data analysis
    """
    # Set up paths
    base_path = Path("Images_Test/dicom")
    results_path = Path("analysis_results")
    
    logger.info("Starting analysis pipeline...")
    
    # Step 1: Load the data
    logger.info("Loading DATSCAN images...")
    loader = DATSCANDataLoader(base_path)
    raw_images, metadata = loader.load_all_data()
    logger.info(f"Loaded {len(raw_images)} images")
    
    # Step 2: Preprocess the images
    logger.info("Preprocessing images...")
    preprocessing_params = PreprocessingParams(
        normalization_method='zscore',  # or 'minmax' or 'percentile'
        mask_threshold=0.1
    )
    preprocessor = DATSCANPreprocessor(preprocessing_params)
    
    # Process all images
    processed_images = preprocessor.preprocess_batch(raw_images, list(metadata.values()))
    logger.info("Preprocessing complete")
    
    # Step 3: Run analysis on both raw and processed images
    logger.info("Running analysis...")
    
    # Analyze raw images
    raw_analyzer = DATSCANAnalyzer(save_dir=results_path / "raw_analysis")
    raw_analyzer.run_full_analysis(raw_images, list(metadata.values()))
    logger.info("Raw image analysis complete")
    
    # Analyze processed images
    processed_analyzer = DATSCANAnalyzer(save_dir=results_path / "processed_analysis")
    processed_analyzer.run_full_analysis(processed_images, list(metadata.values()))
    logger.info("Processed image analysis complete")
    
    # Log completion
    logger.info(f"""
    Analysis pipeline complete! Results are saved in:
    - Raw image analysis: {results_path / 'raw_analysis'}
    - Processed image analysis: {results_path / 'processed_analysis'}
    """)

if __name__ == "__main__":
    main()