# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build dependencies
# Using slim to keep the image small; pytorch/torchvision installed via pip
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies for Pillow and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
# (.dockerignore excludes venv, old_weights, glove, training scripts, etc.)
COPY . .

# HF Spaces runs as a non-root user — ensure cache dirs are writable
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Pre-download CLIP model weights into the image at build time
# This avoids a slow cold-start on first request in HF Spaces
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M'); \
    CLIPProcessor.from_pretrained('wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M'); \
    print('CLIP model pre-cached.')"

# Expose HF Spaces default port
EXPOSE 7860

# Use gunicorn for production serving
# --workers 1 because CLIP model is loaded in-process (memory constrained)
# --timeout 120 for the vocab embedding computation on first boot
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--worker-class", "sync", "app:app"]
