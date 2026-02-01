# Dockerfile for Depth Model API (Azure Container Apps)
# - Remote build friendly (az acr build)
# - Supports scale-to-zero cold starts
# - Model checkpoint is mounted via Azure Files (NOT baked, NOT downloaded)

FROM python:3.10-slim

WORKDIR /app

# System deps (OpenCV headless + curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libgomp1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.api.txt ./requirements.api.txt
RUN pip uninstall -y depth_pro depth-pro || true
RUN pip install --no-cache-dir -r requirements.api.txt

# Copy app code only (no model artifacts)
COPY depth_service.py /app/
COPY openapi.json /app/
COPY src/ /app/src/
COPY config.yaml /app/


# Create mount point for Azure Files
# (the Azure Files volume will be mounted here at runtime)
RUN mkdir -p /app/checkpoints

EXPOSE 5000

# Health check (only checks HTTP server readiness)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:5000/health || exit 1

# Start Gunicorn directly
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "depth_service:app"]
