FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We copy requirements first to leverage caching
COPY ai_service/requirements.txt .

# Use the system python (which is the conda env in this image)
# We don't need a venv because we want to use the pre-installed PyTorch
# However, we must be careful not to overwrite the system torch with pip
# We modify requirements to exclude torch/torchvision/torchaudio/mmcv if they confuse things
# But pip is smart enough to skip if satisfied.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ai_service/ ./ai_service/

# Set environment variables for Blackwell Shim (just in case)
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV FORCE_CUDA=1
ENV MMCV_WITH_OPS=1

# Expose the port
EXPOSE 8001

# Command to run the service
CMD ["uvicorn", "ai_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
