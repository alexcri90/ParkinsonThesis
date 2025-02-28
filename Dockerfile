FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set working directory
WORKDIR /workspace

# Install additional Python packages needed for the project
RUN pip install --no-cache-dir scikit-learn scikit-image SimpleITK nibabel nilearn \
    albumentations seaborn pandas numpy matplotlib tqdm pydicom scipy \
    umap-learn ipywidgets psutil

# Install additional VS Code dependencies
RUN pip install --no-cache-dir pylint black ipykernel jupyter_client

# Set environment variables for better GPU performance
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=8
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Expose port for Jupyter
EXPOSE 8888

# Command to run Jupyter when container starts (will be overridden by docker-compose)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]