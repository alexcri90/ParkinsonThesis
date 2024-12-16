# DATSCAN Analysis Project

## Project Structure
This project consists of several Python modules and a main Jupyter notebook:
- `preprocessing.py`: Contains data loading and preprocessing functionality
- `model.py`: Contains the basic VAE model implementation
- `preprocessing_semisupervised.py`: Enhanced preprocessing with metadata handling
- `model_semisupervised.py`: Semi-supervised VAE implementation
- `notebook.py`: Main Jupyter notebook for execution

## Notebook Structure
The notebook is organized into functional sections:
1. **Setup** (Cells 0-1)
   - Dependencies installation
   - Basic imports

2. **Data Loading** (Cells 2-6)
   - CSV processing
   - DICOM file path collection
   - Dataset organization

3. **Exploratory Data Analysis (EDA)** (Cells 7-24)
   - Image statistics
   - Group comparisons
   - Volume visualization
   - Optional but recommended for first-time analysis

4. **Preprocessing Setup** (Cells 25-26)
   - Preprocessing module initialization
   - Dataloader creation

5. **Basic VAE** (Cells 27-32)
   - Model training
   - Latent space analysis
   - Results visualization

6. **Semi-supervised Extension** (Cells 33-40)
   - Metadata analysis
   - Enhanced model setup
   - Training and evaluation

## Usage Scenarios

### 1. First-Time User
Required sequence:
1. Run Setup (Cells 0-1)
2. Run Data Loading (Cells 2-6)
3. Run EDA (Cells 7-24) - Recommended for data understanding
4. For Basic VAE:
   - Run Cells 25-26
   - Run Cells 27-28 for model setup
   - Run Cell 29 for training
5. For Semi-supervised VAE:
   - Ensure Cells 25-26 are run
   - Run Cells 33-35 for metadata setup
   - Run Cell 36 for model training

### 2. Partially Trained Basic VAE
Required sequence:
1. Run Setup (Cells 0-1)
2. Run Data Loading (Cells 2-6)
3. Run Preprocessing Setup (Cells 25-26)
4. Run Cell 29b for model loading
5. Run Cell 29 with remaining epochs
6. Run Analysis (Cells 30-32)

### 3. Completed Basic VAE
Required sequence:
1. Run Setup (Cells 0-1)
2. Run Data Loading (Cells 2-6)
3. Run Preprocessing Setup (Cells 25-26)
4. Run Cell 29b for model loading
5. Run Analysis (Cells 30-32)

### 4. Partially Trained Semi-supervised VAE
Required sequence:
1. Run Setup (Cells 0-1)
2. Run Data Loading (Cells 2-6)
3. Run Preprocessing Setup (Cells 25-26)
4. Run Metadata Setup (Cells 33-35)
5. Run Cell 36 initialization
6. Load checkpoint and continue training
7. Run Analysis (Cells 37-40)

### 5. Completed Semi-supervised VAE
Required sequence:
1. Run Setup (Cells 0-1)
2. Run Data Loading (Cells 2-6)
3. Run Preprocessing Setup (Cells 25-26)
4. Run Metadata Setup (Cells 33-35)
5. Run Cell 36 initialization
6. Run Analysis (Cells 37-40)

## Important Notes

### Memory Management
- The project is optimized for GPUs with 12GB+ memory (e.g., 4070Ti)
- Each major section includes memory cleanup
- If memory issues occur, reduce batch sizes in Cells 26 and 35

### Checkpoint Files
- `datscan_vae_improved.pt`: Basic VAE model checkpoint
- `datscan_semisupervised_vae.pt`: Semi-supervised VAE checkpoint
- Checkpoints are saved every 10 epochs

### Optional Components
- EDA section (Cells 7-24) can be skipped if data is already understood
- Visualization cells can be run independently after model training
- Analysis sections can be run multiple times without retraining

### Best Practices
1. Always clear kernel when switching between basic and semi-supervised implementations
2. Monitor GPU memory usage during training
3. Save intermediate results during long training sessions
4. Use provided memory cleanup cells between major operations

### Error Recovery
If encountering errors:
1. Clear kernel
2. Start from the minimal required sequence for your scenario
3. Ensure all prerequisite cells are executed
4. Check GPU memory usage and adjust batch sizes if needed