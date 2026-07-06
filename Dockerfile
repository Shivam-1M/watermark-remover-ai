# =============================================================================
# Dockerfile for Video Watermark Removal Application
# =============================================================================
# Base: python:3.10-slim (Debian Bookworm)
# Using the standard slim Python image since we have no NVIDIA GPU.
# PyTorch will run in CPU-only mode.
# =============================================================================

FROM python:3.10-slim

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# ffmpeg: for video splitting / reassembly
# libgl1 + libglib2.0: required by OpenCV headless
# git: needed to clone ProPainter sources during build
# wget: needed by the weight-download script
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
WORKDIR ${APP_HOME}

# Copy and install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# ProPainter — clone the model source code
# ---------------------------------------------------------------------------
# We clone ProPainter into /app/propainter so its model modules are
# importable as `from propainter.model.propainter import InpaintGenerator`.
# The clone is pinned to the exact commit of the v0.1.0 release for
# reproducibility. Update the commit hash when upgrading.
RUN git clone https://github.com/sczhou/ProPainter.git /app/propainter && \
    cd /app/propainter && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# ProPainter weights — download at build time
# ---------------------------------------------------------------------------
# Baking the weights into the image avoids any startup latency when the
# container runs. The three required weight files are:
#   - ProPainter.pth               (~100 MB) — main inpainting model
#   - recurrent_flow_completion.pth (~35 MB)  — flow completion network
#   - raft-things.pth              (~21 MB)  — RAFT optical flow backbone
#
# Total image size increase: ~160 MB of weights.
RUN mkdir -p /app/weights && \
    wget -q -O /app/weights/ProPainter.pth \
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth" && \
    wget -q -O /app/weights/recurrent_flow_completion.pth \
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth" && \
    wget -q -O /app/weights/raft-things.pth \
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth"

# Copy the rest of the application source code
COPY . .

# Create the uploads directory for video processing artifacts
RUN mkdir -p /app/uploads

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
EXPOSE 8000

# TODO(security): In production, bind to 127.0.0.1 behind a reverse proxy.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
