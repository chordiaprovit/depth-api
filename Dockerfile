FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libgomp1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.api.txt ./requirements.api.txt

# Remove any PyPI name-collision packages just in case
RUN pip uninstall -y depth_pro depth-pro || true

# Install dependencies
RUN pip install --no-cache-dir -r requirements.api.txt

# Force-install Apple Depth Pro LAST so it wins
RUN pip install --no-cache-dir --upgrade --force-reinstall git+https://github.com/apple/ml-depth-pro.git

# Verify Apple Depth Pro is correct (fail build if wrong)
RUN python -c "import depth_pro; print('depth_pro path:', getattr(depth_pro,'__file__',None)); print('has DepthProConfig:', hasattr(depth_pro,'DepthProConfig')); assert hasattr(depth_pro,'DepthProConfig')"

COPY depth_service.py /app/
COPY openapi.json /app/
COPY src/ /app/src/

RUN mkdir -p /app/checkpoints

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-5000}/health || exit 1

CMD ["sh", "-c", "gunicorn -w 1 -k gthread -t 300 --threads 4 -b 0.0.0.0:${PORT:-5000} depth_service:app"]
