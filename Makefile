# Makefile for DATSCAN Unsupervised Learning Project

.PHONY: clean install train evaluate test

# Default Python interpreter
PYTHON = python

# Installation
install:
	$(PYTHON) -m pip install -e .

# Train autoencoder with default settings
train:
	$(PYTHON) train.py --model autoencoder --batch_size 16 --epochs 200 --lr 3e-5

# Train with specific configuration file
train-config:
	$(PYTHON) train.py --config configs/autoencoder_config.json

# Evaluate trained model
evaluate:
	$(PYTHON) evaluate.py --model_dir trained_models/autoencoder --find_outliers --analyze_latent

# Clean temporary files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf models/__pycache__
	rm -rf trainers/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Create project directory structure
init:
	mkdir -p models
	mkdir -p trainers
	mkdir -p configs
	mkdir -p trained_models
	touch models/__init__.py
	touch trainers/__init__.py
	touch __init__.py

# Test model loading (with dummy data)
test:
	$(PYTHON) -c "from models.autoencoder import Autoencoder; model = Autoencoder(); print(f'Model created with {model.count_parameters():,} parameters')"