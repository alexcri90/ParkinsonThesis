# Use NVIDIA's PyTorch container as base
FROM nvcr.io/nvidia/pytorch:23.12-py3

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
    tqdm \
    pydicom \
    umap-learn

# Set environment variables for better GPU utilization
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy project files
COPY . /workspace/

# Command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]