FROM nvcr.io/nvidia/pytorch:24.12-py3

# Install additional dependencies your project needs
RUN pip install --no-cache-dir \
    pytest \
    pylint \
    black \
    jupyterlab

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["bash"]