from __future__ import annotations

import base64
import json
import logging
import os
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint

# Make sure /app is on path so `import src...` works when src is copied to /app/src
sys.path.append(os.path.dirname(__file__))

from src.config.config import get_depth_model, load_config  # noqa: E402

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Env-driven config (ACA-friendly) ----
MODEL_PATH = os.getenv("MODEL_PATH", "/app/checkpoints/depth_pro.pt")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")  # optional; may not exist
API_KEY = os.getenv("API_KEY", "").strip()  # should be set from ACA secret

# Globals (lazy initialized)
_depth_model = None
_config = None


# ---------------------------
# Swagger UI (Flask)
# ---------------------------
SWAGGER_URL = "/docs"
API_URL = "/openapi.json"

swagger_bp = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={"app_name": "Depth Estimation API"},
)
app.register_blueprint(swagger_bp, url_prefix=SWAGGER_URL)


@app.get("/openapi.json")
def openapi():
    """Serve OpenAPI spec for Swagger UI."""
    spec_path = os.path.join(os.path.dirname(__file__), "openapi.json")
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return (
            jsonify(
                {
                    "error": "openapi.json not found",
                    "hint": "Copy openapi.json next to depth_service.py in the image.",
                }
            ),
            500,
        )
    except Exception as e:
        return jsonify({"error": "Failed to load openapi.json", "detail": str(e)}), 500


# ---------------------------
# Auth helpers
# ---------------------------
def _is_auth_required() -> bool:
    return bool(API_KEY)


def _check_api_key() -> bool:
    """Validate incoming request against API_KEY (if set)."""
    if not API_KEY:
        return True  # auth disabled

    # Prefer x-api-key
    got = request.headers.get("x-api-key", "").strip()
    if got and got == API_KEY:
        return True

    # Support Authorization: Bearer <key>
    auth = request.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return token == API_KEY

    return False


# ---------------------------
# Lazy init
# ---------------------------
def _ensure_model_loaded():
    """Load config (optional) + instantiate model exactly once."""
    global _depth_model, _config

    if _depth_model is not None:
        return

    logger.info("Loading config (optional): %s", CONFIG_PATH)
    _config = load_config(CONFIG_PATH)  # returns None if missing (with our updated config.py)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Ensure Azure Files share is mounted to /app/checkpoints and MODEL_PATH is correct."
        )

    logger.info("Initializing depth model from %s ...", MODEL_PATH)
    _depth_model = get_depth_model(config=_config, model_path=MODEL_PATH)
    logger.info("✅ Depth model initialized successfully")


# ---------------------------
# Request guard: allow public paths, enforce auth, lazy-init only when needed
# ---------------------------
PUBLIC_PATHS = {
    "/health",
    "/openapi.json",
}
PUBLIC_PREFIXES = (
    "/docs",
)


@app.before_request
def enforce_auth_and_lazy_init():
    path = request.path or ""

    # Always allow public endpoints (no auth, no model init)
    if path in PUBLIC_PATHS or any(path.startswith(pfx) for pfx in PUBLIC_PREFIXES):
        return None

    # Enforce API key if configured
    if _is_auth_required() and not _check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    # Only load model for endpoints that need it
    if path in ("/estimate_depth", "/model_info"):
        try:
            _ensure_model_loaded()
        except Exception as e:
            logger.exception("❌ Failed to initialize depth model: %s", e)
            return jsonify({"error": "Model initialization failed", "detail": str(e)}), 500

    return None


# ---------------------------
# Routes
# ---------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint (no auth, no model load)."""
    return (
        jsonify(
            {
                "status": "ok",
                "service": "depth-model-api",
                "model_loaded": _depth_model is not None,
                "auth_enabled": _is_auth_required(),
            }
        ),
        200,
    )


@app.route("/model_info", methods=["GET"])
def model_info():
    """Triggers model load (auth required if enabled) and returns config."""
    # ensure loaded by before_request
    model_name = None
    device = None
    params = {}

    # If config.yaml is present, expose those fields; otherwise rely on env
    if isinstance(_config, dict) and "depth_model" in _config:
        dm = _config["depth_model"] or {}
        model_name = dm.get("name")
        device = dm.get("device")
        params = dm.get("params", {}) or {}

    return (
        jsonify(
            {
                "model_name": model_name or os.getenv("MODEL_TYPE", "ml_depth_pro"),
                "device": device or os.getenv("DEVICE", "cpu"),
                "parameters": params,
                "model_path": MODEL_PATH,
            }
        ),
        200,
    )


@app.route("/estimate_depth", methods=["POST"])
def estimate_depth_endpoint():
    """Estimate depth from an image.

    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "bounding_box": {"x1": int, "y1": int, "x2": int, "y2": int}
    }
    """
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "Missing 'image' in request"}), 400
    if "bounding_box" not in data:
        return jsonify({"error": "Missing 'bounding_box' in request"}), 400

    # Decode base64 image
    try:
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
    except Exception:
        return jsonify({"error": "Invalid base64 image payload"}), 400

    # Parse bounding box
    bbox = data["bounding_box"]
    for k in ("x1", "y1", "x2", "y2"):
        if k not in bbox:
            return jsonify({"error": f"Missing bounding_box.{k}"}), 400

    # Minimal Box wrapper to match .xyxy usage in model code
    class Box:
        def __init__(self, x1, y1, x2, y2):
            self.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]

    box = Box(bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])

    try:
        logger.info("Estimating depth...")
        depth_feet = float(_depth_model.estimate_depth(frame=frame, box=box))
        return (
            jsonify(
                {
                    "status": "success",
                    "depth_meters": depth_feet / 3.28084,
                    "depth_feet": depth_feet,
                }
            ),
            200,
        )
    except Exception as e:
        logger.exception("Error during depth estimation: %s", e)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Depth Model API service on port %s...", port)
    app.run(host="0.0.0.0", port=port, debug=False)
