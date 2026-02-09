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

RUN rm -rf /usr/local/lib/python3.10/site-packages/depth_pro* || true
RUN pip install --no-cache-dir -r requirements.api.txt
RUN git clone --depth 1 https://github.com/apple/ml-depth-pro.git /tmp/ml-depth-pro \
 && pip install --no-cache-dir --upgrade --force-reinstall /tmp/ml-depth-pro \
 && rm -rf /tmp/ml-depth-pro

RUN python -c "import depth_pro; print('depth_pro path:', depth_pro.__file__); print('dir has DepthProConfig:', 'DepthProConfig' in dir(depth_pro)); assert 'DepthProConfig' in dir(depth_pro)"

COPY depth_service.py /app/
COPY openapi.json /app/
COPY src/ /app/src/

RUN mkdir -p /app/checkpoints

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-5000}/health || exit 1

CMD ["sh", "-c", "gunicorn -w 1 -k gthread -t 300 --threads 4 -b 0.0.0.0:${PORT:-5000} depth_service:app"]
