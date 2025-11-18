# Use NVIDIA CUDA base image compatible with PyTorch and Tesla T4
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash appuser
WORKDIR /home/appuser/app

# Create and activate virtual environment
RUN python3.10 -m venv /home/appuser/venv
ENV PATH="/home/appuser/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install dependencies
# This is done in a separate step to leverage Docker layer caching
COPY requirements.txt .

# Install PyTorch and related packages first to ensure CUDA compatibility
# Ensure torch version is compatible with CUDA 12.4 from the base image
RUN pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies, ignoring numpy for now to avoid conflicts
RUN pip install -r requirements.txt --no-deps

# Copy the rest of the application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /home/appuser/app
USER appuser

# Expose port and run the application
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

