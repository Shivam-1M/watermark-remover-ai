# =============================================================================
# Dockerfile for Video Watermark Removal Application
# =============================================================================
# Base: NVIDIA CUDA 11.8 with cuDNN 8 on Ubuntu 22.04
# This image provides GPU pass-through support for PyTorch inference via
# the NVIDIA Container Toolkit (nvidia-docker2).
# =============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
# Prevent interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive
# Ensure Python output is sent straight to the container logs
ENV PYTHONUNBUFFERED=1
# Set the working directory inside the container
ENV APP_HOME=/app

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# Install Python 3.10, pip, ffmpeg (for video processing), and essential libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python for convenience
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
WORKDIR ${APP_HOME}

# Copy and install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Create the uploads directory for video processing artifacts.
# This directory is also mapped as a Docker volume for persistence.
RUN mkdir -p /app/uploads

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Expose the FastAPI default port
EXPOSE 8000

# Start the Uvicorn ASGI server.
# Listens on 0.0.0.0 inside the container so Docker can map it to the host.
# For production, consider adding --workers N for multi-process serving.
# TODO(security): In production, bind to 127.0.0.1 behind a reverse proxy.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
