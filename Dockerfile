# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Copy Montserrat font from repository and register it
COPY fonts /tmp/fonts
RUN mkdir -p /usr/share/fonts/truetype/montserrat && \
    cp /tmp/fonts/*.ttf /usr/share/fonts/truetype/montserrat/ && \
    fc-cache -f -v && \
    rm -rf /tmp/fonts

# Create a non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

# Set up virtual environment
ENV VIRTUAL_ENV=/home/appuser/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /home/appuser/app

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch and related packages first to ensure CUDA compatibility
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
RUN pip install torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

