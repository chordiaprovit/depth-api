from __future__ import annotations

import base64
import json
import logging
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
API_KEY = os.getenv("API_KEY", "").strip()  # if empty -> auth disabled

# Globals (lazy initialized)
_depth_model = None
_config = None

_midas_model = None
_midas_transform = None
_midas_device = None


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

    got = request.headers.get("x-api-key", "").strip()
    if got and got == API_KEY:
        return True

    auth = request.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return token == API_KEY

    return False


PUBLIC_PATHS = {"/health", "/openapi.json"}
PUBLIC_PREFIXES = ("/docs",)


# ---------------------------
# Lazy init (DepthPro)
# ---------------------------
def _ensure_model_loaded():
    """Load config (optional) + instantiate DepthPro model exactly once."""
    global _depth_model, _config

    if _depth_model is not None:
        return

    logger.info("Loading config (optional): %s", CONFIG_PATH)
    _config = load_config(CONFIG_PATH)  # returns None if missing

    logger.info("Initializing depth model with MODEL_PATH=%s", MODEL_PATH)
    _depth_model = get_depth_model(model_path=MODEL_PATH)
    logger.info("✅ Depth model ready")


# ---------------------------
# Lazy init (MiDaS)
# ---------------------------
def _ensure_midas_loaded():
    global _midas_model, _midas_transform, _midas_device

    if _midas_model is not None:
        return

    logger.info("Loading MiDaS_small (relative depth)...")
    _midas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fast + stable choice:
    _midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    _midas_model.to(_midas_device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    _midas_transform = transforms.small_transform

    logger.info("✅ MiDaS loaded on %s", _midas_device)


@torch.inference_mode()
def _midas_depth_map_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = _midas_transform(frame_rgb).to(_midas_device)

    pred = _midas_model(inp)
    pred = F.interpolate(
        pred.unsqueeze(1),
        size=frame_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    return pred.detach().cpu().numpy().astype(np.float32)


# ---------------------------
# Shared helpers
# ---------------------------
def _clip(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def _box_from_xy(w: int, h: int, x: int, y: int, half: int) -> dict:
    x1 = _clip(x - half, 0, w - 1)
    y1 = _clip(y - half, 0, h - 1)
    x2 = _clip(x + half, 0, w - 1)
    y2 = _clip(y + half, 0, h - 1)
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class _Box:
    """Minimal wrapper to match model's `.xyxy` usage."""

    def __init__(self, bbox: dict):
        self.xyxy = [
            np.array([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], dtype=np.float32)
        ]


# ---------------------------
# Middleware
# ---------------------------
@app.before_request
def enforce_auth_and_lazy_init():
    path = request.path or ""

    # Public endpoints (no auth, no model init)
    if path in PUBLIC_PATHS or any(path.startswith(pfx) for pfx in PUBLIC_PREFIXES):
        return None

    # Enforce API key if configured
    if _is_auth_required() and not _check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    # Load DepthPro only when needed
    if path in ("/estimate_depth", "/model_info"):
        try:
            _ensure_model_loaded()
        except Exception as e:
            logger.exception("❌ Failed to initialize depth model: %s", e)
            return jsonify({"error": "Model initialization failed", "detail": str(e)}), 500

    # Load MiDaS only when needed
    if path in ("/estimate_depth_midas",):
        try:
            _ensure_midas_loaded()
        except Exception as e:
            logger.exception("❌ Failed to initialize MiDaS: %s", e)
            return jsonify({"error": "MiDaS initialization failed", "detail": str(e)}), 500

    return None


# ---------------------------
# Routes
# ---------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/model_info", methods=["GET"])
def model_info():
    return (
        jsonify(
            {
                "model_type": "ml_depth_pro",
                "device": DEVICE,
                "model_path": MODEL_PATH,
                "auth_enabled": _is_auth_required(),
            }
        ),
        200,
    )


@app.route("/estimate_depth", methods=["POST"])
def estimate_depth_endpoint():
    """
    DepthPro endpoint.
    Supports:
      1) multipart/form-data:
         - image (file)
         - x, y (int) + optional point_box
         - OR bounding_box (JSON string)
      2) application/json (legacy):
         - image (base64 string)
         - bounding_box {x1,y1,x2,y2}
    """
    ctype = request.content_type or ""

    # ---------------- multipart/form-data (preferred) ----------------
    if "multipart/form-data" in ctype:
        if "image" not in request.files:
            return jsonify({"error": "image file is required"}), 400

        img_bytes = request.files["image"].read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image"}), 400

        h, w = frame.shape[:2]
        point_box = int(request.form.get("point_box", 8))
        point_box = max(1, min(point_box, 64))

        # Priority: bounding_box → x/y
        if "bounding_box" in request.form and request.form["bounding_box"].strip():
            try:
                bbox = json.loads(request.form["bounding_box"])
            except Exception:
                return jsonify({"error": "invalid bounding_box JSON"}), 400
        else:
            if "x" not in request.form or "y" not in request.form:
                return jsonify({"error": "x and y are required (or provide bounding_box)"}), 400
            x = int(request.form["x"])
            y = int(request.form["y"])
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            bbox = _box_from_xy(w, h, x, y, point_box)

        box = _Box(bbox)

        try:
            depth_feet = float(_depth_model.estimate_depth(frame=frame, box=box))
            return (
                jsonify(
                    {
                        "status": "success",
                        "mode": "bbox" if "bounding_box" in request.form else "point",
                        "sample_box": bbox,
                        "depth_meters": depth_feet / 3.28084,
                        "depth_feet": depth_feet,
                    }
                ),
                200,
            )
        except Exception as e:
            logger.exception("Error during DepthPro estimation: %s", e)
            return jsonify({"error": str(e)}), 500

    # ---------------- legacy JSON (base64) ----------------
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "Missing 'image' in request"}), 400
    if "bounding_box" not in data:
        return jsonify({"error": "Missing 'bounding_box' in request"}), 400

    try:
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
    except Exception:
        return jsonify({"error": "Invalid base64 image payload"}), 400

    bbox = data["bounding_box"]
    for k in ("x1", "y1", "x2", "y2"):
        if k not in bbox:
            return jsonify({"error": f"Missing bounding_box.{k}"}), 400

    box = _Box(bbox)

    try:
        depth_feet = float(_depth_model.estimate_depth(frame=frame, box=box))
        return (
            jsonify(
                {
                    "status": "success",
                    "mode": "bbox_json",
                    "sample_box": bbox,
                    "depth_meters": depth_feet / 3.28084,
                    "depth_feet": depth_feet,
                }
            ),
            200,
        )
    except Exception as e:
        logger.exception("Error during DepthPro estimation: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/estimate_depth_midas", methods=["POST"])
def estimate_depth_midas():
    """
    MiDaS endpoint (FAST, RELATIVE depth).
    multipart/form-data:
      - image (file)
      - x, y (int)
      - point_box (optional)
      - max_side (optional, default 640) -> resize for speed
    """
    ctype = request.content_type or ""
    if "multipart/form-data" not in ctype:
        return jsonify({"error": "Use multipart/form-data with image + x + y"}), 400

    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400
    if "x" not in request.form or "y" not in request.form:
        return jsonify({"error": "x and y are required"}), 400

    img_bytes = request.files["image"].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    # Resize for speed (big win)
    max_side = int(request.form.get("max_side", 640))
    h0, w0 = frame.shape[:2]
    scale = max(h0, w0) / max_side
    if scale > 1:
        frame = cv2.resize(frame, (int(w0 / scale), int(h0 / scale)))

    h, w = frame.shape[:2]
    x = int(request.form["x"])
    y = int(request.form["y"])
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    point_box = int(request.form.get("point_box", 8))
    point_box = max(1, min(point_box, 64))

    bbox = _box_from_xy(w, h, x, y, point_box)

    depth_map = _midas_depth_map_bgr(frame)

    roi = depth_map[bbox["y1"] : bbox["y2"], bbox["x1"] : bbox["x2"]]
    if roi.size == 0:
        return jsonify({"error": "invalid ROI after clamping"}), 400

    depth_raw = float(np.mean(roi))

    dmin = float(np.min(depth_map))
    dmax = float(np.max(depth_map))
    depth_0_1 = 0.0 if (dmax - dmin) < 1e-6 else (depth_raw - dmin) / (dmax - dmin)

    return (
        jsonify(
            {
                "status": "success",
                "model": "MiDaS_small",
                "mode": "point",
                "x": x,
                "y": y,
                "sample_box": bbox,
                "depth_raw": depth_raw,
                "depth_0_1": depth_0_1,
                "note": "MiDaS provides relative depth (unitless). Not meters/feet unless calibrated.",
                "image_size_used": {"width": w, "height": h},
            }
        ),
        200,
    )


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404
