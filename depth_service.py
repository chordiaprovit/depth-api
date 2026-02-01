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

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.config.config import get_depth_model, load_config  # noqa: E402

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "/app/checkpoints/depth_pro.pt")

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
        with open(spec_path, "r") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return (
            jsonify(
                {
                    "error": "openapi.json not found",
                    "hint": "Add openapi.json next to this file in the image.",
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
    return bool(os.getenv("API_KEY"))


def _check_api_key() -> bool:
    """Validate incoming request against API_KEY (if set)."""
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return True  # auth disabled

    # Prefer x-api-key
    got = request.headers.get("x-api-key", "").strip()
    if got and got == expected:
        return True

    # Support Authorization: Bearer <key>
    auth = request.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        if token == expected:
            return True

    return False


# ---------------------------
# Request guard: allow health + docs, enforce auth elsewhere, lazy-init model
# ---------------------------
PUBLIC_PATHS = {
    "/health",
    "/openapi.json",
}
PUBLIC_PREFIXES = (
    "/docs",  # swagger UI and its assets
)


@app.before_request
def enforce_auth_and_lazy_init():
    """Enforce API key (optional) and initialize model lazily (only for inference endpoints)."""
    global _depth_model, _config

    path = request.path or ""

    # Always allow public endpoints (no auth, no model init)
    if path in PUBLIC_PATHS or any(path.startswith(pfx) for pfx in PUBLIC_PREFIXES):
        return None

    # Enforce API key if configured
    if _is_auth_required() and not _check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    # Lazy init model AFTER auth passes (so unauthorized users don't trigger model load)
    if _depth_model is None:
        try:
            logger.info("Loading depth model configuration...")
            _config = load_config(config_path="./config.yaml")

            # Fail fast if mount/path is wrong
            if not os.path.exists(MODEL_PATH):
                logger.error("Model file not found at %s", MODEL_PATH)
                return (
                    jsonify(
                        {
                            "error": "Model file not found",
                            "model_path": MODEL_PATH,
                            "hint": "Ensure Azure Files share is mounted to /app/checkpoints "
                            "and MODEL_PATH points to the checkpoint filename.",
                        }
                    ),
                    500,
                )

            logger.info("Initializing depth model from %s ...", MODEL_PATH)
            _depth_model = get_depth_model(config=_config, model_path=MODEL_PATH)
            logger.info("✅ Depth model initialized successfully")
        except Exception as e:
            logger.exception("❌ Failed to initialize depth model: %s", e)
            return jsonify({"error": "Model initialization failed", "detail": str(e)}), 500

    return None


# ---------------------------
# Routes
# ---------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint (no auth)."""
    model_name = _config["depth_model"]["name"] if _config else "not-loaded"
    return (
        jsonify(
            {
                "status": "healthy",
                "service": "depth-model-api",
                "model": model_name,
                "auth_enabled": _is_auth_required(),
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
    try:
        data = request.get_json(silent=True) or {}
        if "image" not in data:
            return jsonify({"error": "Missing 'image' in request"}), 400
        if "bounding_box" not in data:
            return jsonify({"error": "Missing 'bounding_box' in request"}), 400

        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Parse bounding box
        bbox = data["bounding_box"]

        # Minimal Box wrapper to match .xyxy usage in model code
        class Box:
            def __init__(self, x1, y1, x2, y2):
                self.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]

        box = Box(bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])

        logger.info("Estimating depth...")
        depth_feet = float(_depth_model.estimate_depth(frame=frame, box=box))

        return (
            jsonify(
                {
                    "status": "success",
                    "depth_meters": depth_feet / 3.28084,
                    "depth_feet": depth_feet,
                    "model": _config["depth_model"]["name"],
                }
            ),
            200,
        )

    except Exception as e:
        logger.exception("Error during depth estimation: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """Get information about the loaded model."""
    if _config is None:
        return jsonify({"error": "Model not initialized yet"}), 503

    return (
        jsonify(
            {
                "model_name": _config["depth_model"]["name"],
                "device": _config["depth_model"]["device"],
                "parameters": _config["depth_model"].get("params", {}),
                "model_path": MODEL_PATH,
            }
        ),
        200,
    )


# ---------------------------
# Error handlers
# ---------------------------
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
