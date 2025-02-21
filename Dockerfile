# Use the official NVIDIA NGC PyTorch container with Python 3.12 (24.12 release)
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set the working directory inside the container
WORKDIR /workspace

# Copy your repository contents into the container
COPY . /workspace

# (Optional) Install additional dependencies if needed:
# RUN pip install -r requirements.txt

# Expose port 8888 (for Jupyter Lab or other web apps)
EXPOSE 8888

# Default command - adjust as necessary for your development needs
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
