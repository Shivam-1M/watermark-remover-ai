# =============================================================================
# Dockerfile for Video Watermark Removal Application
# =============================================================================
# Base: python:3.10-slim (Debian Bookworm)
# Lightweight image — no deep learning frameworks, no model weights.
# Inpainting is handled entirely by OpenCV (C++ backend, ~5ms per frame).
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
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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

# Create the uploads directory for video processing artifacts
RUN mkdir -p /app/uploads

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
EXPOSE 8000

# TODO(security): In production, bind to 127.0.0.1 behind a reverse proxy.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
