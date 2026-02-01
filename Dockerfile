FROM python:3.10-slim

WORKDIR /app

# System deps (OpenCV headless + curl for healthcheck + git for installing Apple Depth Pro)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libgomp1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.api.txt ./requirements.api.txt

# Remove any PyPI name-collision packages just in case
RUN pip uninstall -y depth_pro depth-pro || true

# Install dependencies (includes Apple Depth Pro via git+https://...)
RUN pip install --no-cache-dir -r requirements.api.txt

# Fail the build if Apple Depth Pro isn't importable
RUN python -c "import depth_pro; print('depth_pro OK:', depth_pro.__file__); assert hasattr(depth_pro, 'DepthProConfig')"

# Copy app code only (no model artifacts)
COPY depth_service.py /app/
COPY openapi.json /app/
COPY src/ /app/src/

# Azure Files mount point (mounted at runtime)
RUN mkdir -p /app/checkpoints

EXPOSE 5000

# Optional healthcheck (kept simple; ACA doesn't require Docker HEALTHCHECK)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-5000}/health || exit 1

# Start Gunicorn: bind to ACA PORT if provided
CMD ["sh", "-c", "gunicorn -w 1 -k gthread -t 300 --threads 4 -b 0.0.0.0:${PORT:-5000} depth_service:app"]
