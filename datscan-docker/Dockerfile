FROM nvcr.io/nvidia/pytorch:24.12-py3

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

# Make sure the container keeps running
CMD ["bash"]