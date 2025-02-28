# docker_setup.py
import os
import argparse
import subprocess
import shutil
import sys

def create_dockerfile():
    """Create Dockerfile for DATSCAN project."""
    dockerfile_content = """FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set working directory
WORKDIR /workspace

# Install additional dependencies
RUN pip install --no-cache-dir \
    scikit-learn \
    scikit-image \
    SimpleITK \
    nibabel \
    nilearn \
    albumentations \
    seaborn \
    pandas \
    matplotlib \
    tqdm \
    pydicom \
    scipy \
    umap-learn

# Copy project files
COPY . /workspace/

# Set up entrypoint
ENTRYPOINT ["/bin/bash"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    print("Created Dockerfile")

def create_docker_compose():
    """Create docker-compose.yml for DATSCAN project."""
    compose_content = """version: '3'
services:
  datscan-dev:
    build: .
    image: datscan-dev
    container_name: datscan-dev
    volumes:
      - .:/workspace
      - ./data:/workspace/data
      - ./trained_models:/workspace/trained_models
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "8888:8888"  # For JupyterLab
    tty: true
    stdin_open: true
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose_content)
    
    print("Created docker-compose.yml")

def create_setup_script():
    """Create setup.sh script for Docker environment."""
    script_content = """#!/bin/bash
# Setup script for DATSCAN project in Docker environment

# Create necessary directories
mkdir -p data/processed_cache
mkdir -p trained_models/autoencoder/checkpoints
mkdir -p evaluation_results

# Check NVIDIA Docker installation
if command -v nvidia-smi &> /dev/null
then
    echo "NVIDIA GPU detected"
    nvidia-smi
else
    echo "WARNING: NVIDIA GPU not detected or drivers not installed"
fi

# Launch JupyterLab if requested
if [ "$1" = "--jupyter" ]; then
    echo "Starting JupyterLab server..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
fi
"""
    
    with open('setup.sh', 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod('setup.sh', 0o755)
    
    print("Created setup.sh script")

def create_run_scripts():
    """Create helper scripts for running training and evaluation."""
    train_script = """#!/bin/bash
# Run training in Docker container

# Default configuration
CONFIG_FILE="rtx_4070ti.json"

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --config)
        CONFIG_FILE="$2"
        shift
        shift
        ;;
        *)
        EXTRA_ARGS="$EXTRA_ARGS $1"
        shift
        ;;
    esac
done

# Run the training script
python train.py --config $CONFIG_FILE $EXTRA_ARGS
"""
    
    eval_script = """#!/bin/bash
# Run evaluation in Docker container

# Parse command line arguments
MODEL_PATH=""
DATA_CSV=""
OUTPUT_DIR="evaluation_results"

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --model)
        MODEL_PATH="$2"
        shift
        shift
        ;;
        --data)
        DATA_CSV="$2"
        shift
        shift
        ;;
        --output)
        OUTPUT_DIR="$2"
        shift
        shift
        ;;
        *)
        EXTRA_ARGS="$EXTRA_ARGS $1"
        shift
        ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ] || [ -z "$DATA_CSV" ]; then
    echo "Error: Required arguments missing"
    echo "Usage: ./run_evaluation.sh --model MODEL_PATH --data DATA_CSV [--output OUTPUT_DIR]"
    exit 1
fi

# Run the evaluation script
python evaluate.py --model_path $MODEL_PATH --data_csv $DATA_CSV --output_dir $OUTPUT_DIR $EXTRA_ARGS
"""
    
    with open('run_training.sh', 'w') as f:
        f.write(train_script)
    
    with open('run_evaluation.sh', 'w') as f:
        f.write(eval_script)
    
    # Make scripts executable
    os.chmod('run_training.sh', 0o755)
    os.chmod('run_evaluation.sh', 0o755)
    
    print("Created run_training.sh and run_evaluation.sh scripts")

def create_config_files():
    """Create configuration files for different hardware setups."""
    # Default config for RTX 4070 Ti
    rtx_4070ti_config = {
        "data": {
            "data_csv": "validated_file_paths.csv",
            "cache_dir": "processed_cache",
            "train_ratio": 0.8,
            "batch_size": 16,
            "num_workers": 6,
            "pin_memory": True
        },
        "model": {
            "name": "autoencoder",
            "latent_dim": 128,
            "striatal_weight": 5.0
        },
        "training": {
            "epochs": 200,
            "learning_rate": 3e-5,
            "weight_decay": 1e-5,
            "gradient_clip": 1.0,
            "gradient_accumulation": 2,
            "patience": 15,
            "scheduler_metric": "striatal",
            "output_dir": "./trained_models/autoencoder",
            "resume": False
        }
    }
    
    # Config for low memory systems
    low_memory_config = {
        "data": {
            "data_csv": "validated_file_paths.csv",
            "cache_dir": "processed_cache",
            "train_ratio": 0.8,
            "batch_size": 4,
            "num_workers": 2,
            "pin_memory": True
        },
        "model": {
            "name": "autoencoder",
            "latent_dim": 64,
            "striatal_weight": 5.0
        },
        "training": {
            "epochs": 200,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "gradient_clip": 1.0,
            "gradient_accumulation": 8,
            "patience": 15,
            "scheduler_metric": "striatal",
            "output_dir": "./trained_models/autoencoder",
            "resume": False
        }
    }
    
    # Save configs as JSON files
    import json
    
    os.makedirs('configs', exist_ok=True)
    
    with open('configs/rtx_4070ti.json', 'w') as f:
        json.dump(rtx_4070ti_config, f, indent=4)
    
    with open('configs/low_memory.json', 'w') as f:
        json.dump(low_memory_config, f, indent=4)
    
    print("Created configuration files in configs/ directory")

def check_docker_installation():
    """Check if Docker and NVIDIA Docker are installed."""
    try:
        # Check Docker installation
        docker_version = subprocess.check_output(['docker', '--version']).decode().strip()
        print(f"Docker installed: {docker_version}")
        
        # Check Docker Compose installation
        compose_version = subprocess.check_output(['docker-compose', '--version']).decode().strip()
        print(f"Docker Compose installed: {compose_version}")
        
        # Check NVIDIA Docker
        try:
            nvidia_docker = subprocess.check_output(['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.0-base', 'nvidia-smi']).decode()
            print("NVIDIA Docker is working correctly")
        except subprocess.CalledProcessError:
            print("WARNING: NVIDIA Docker is not working correctly. Please install nvidia-container-toolkit.")
            print("Instructions: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Docker or Docker Compose is not installed or not in PATH")
        print("Please install Docker and Docker Compose first")
        print("Docker installation: https://docs.docker.com/engine/install/")
        print("Docker Compose installation: https://docs.docker.com/compose/install/")
        return False
    
    return True

def setup_docker_environment(args):
    """Set up Docker environment for DATSCAN project."""
    print("Setting up Docker environment for DATSCAN project...")
    
    # Create project structure
    project_dir = args.output_dir
    os.makedirs(project_dir, exist_ok=True)
    
    # Switch to project directory
    original_dir = os.getcwd()
    os.chdir(project_dir)
    
    # Create Docker files
    create_dockerfile()
    create_docker_compose()
    create_setup_script()
    create_run_scripts()
    create_config_files()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Copy Python scripts
    script_files = ['train.py', 'evaluate.py', 'dataset.py', 'autoencoder.py', 'config.py', 'clear_memory.py']
    
    for script in script_files:
        src_path = os.path.join(original_dir, script)
        dst_path = os.path.join(project_dir, script)
        
        # Check if source file exists
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {script} to project directory")
        else:
            print(f"Warning: {script} not found in current directory")
    
    # Create README.md
    readme_content = """# DATSCAN Autoencoder Project

## Docker Setup

This project uses NVIDIA's PyTorch Docker image for easy setup and reproducibility.

### Prerequisites

- Docker
- NVIDIA Docker (nvidia-container-toolkit)
- NVIDIA GPU driver

### Getting Started

1. Build the Docker image:
   ```
   docker-compose build
   ```

2. Run the container:
   ```
   docker-compose up -d
   ```

3. Execute setup script:
   ```
   docker-compose exec datscan-dev ./setup.sh
   ```

4. Train the model:
   ```
   docker-compose exec datscan-dev ./run_training.sh
   ```

5. Evaluate the model:
   ```
   docker-compose exec datscan-dev ./run_evaluation.sh --model trained_models/autoencoder/checkpoints/best_val.pt --data validated_file_paths.csv
   ```

### Running JupyterLab

To start JupyterLab server:
```
docker-compose exec datscan-dev ./setup.sh --jupyter
```

Then open the provided URL in your browser.

## Configuration

Configuration files are stored in the `configs/` directory:
- `rtx_4070ti.json`: Optimized for RTX 4070 Ti GPU
- `low_memory.json`: Optimized for systems with limited RAM

## Directory Structure

- `data/`: Dataset files and cache
- `trained_models/`: Saved model checkpoints
- `evaluation_results/`: Model evaluation outputs
- `configs/`: Configuration files
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"\nDocker environment setup completed in {project_dir}")
    print("\nNext steps:")
    print("1. Copy your validated_file_paths.csv to the data/ directory")
    print("2. Build the Docker image: docker-compose build")
    print("3. Run the container: docker-compose up -d")
    print("4. Execute setup script: docker-compose exec datscan-dev ./setup.sh")
    print("5. Train the model: docker-compose exec datscan-dev ./run_training.sh")
    
    # Return to original directory
    os.chdir(original_dir)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up Docker environment for DATSCAN project")
    parser.add_argument('--output-dir', type=str, default='./datscan-docker',
                        help='Output directory for Docker setup (default: ./datscan-docker)')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check Docker installation without creating files')
    
    args = parser.parse_args()
    
    # Check Docker installation
    if not check_docker_installation():
        sys.exit(1)
    
    # Exit if check-only
    if args.check_only:
        print("Docker installation check completed")
        sys.exit(0)
    
    # Set up Docker environment
    setup_docker_environment(args)

if __name__ == "__main__":
    main()