import base64
import json
import logging
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request

from src.models.ml_depth_pro import MlDepthPro

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_TYPE = os.getenv("MODEL_TYPE", "ml_depth_pro")
DEVICE = os.getenv("DEVICE", "cpu")

app = Flask(__name__)

_depth_model = None


def require_api_key():
    key = (
        request.headers.get("x-api-key")
        or request.headers.get("Authorization", "").replace("Bearer ", "")
    )
    return key == API_KEY


@app.before_request
def auth_middleware():
    if request.path in ["/health", "/docs", "/openapi.json"]:
        return
    if not require_api_key():
        return jsonify({"error": "Unauthorized"}), 401


def load_model():
    global _depth_model
    if _depth_model is None:
        logger.info("Loading DepthPro model...")
        _depth_model = MlDepthPro(
            device=DEVICE,
            model_type=MODEL_TYPE,
            model_path=MODEL_PATH,
        )
        logger.info("DepthPro model loaded")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/model_info", methods=["GET"])
def model_info():
    load_model()
    return jsonify(
        {
            "model_type": MODEL_TYPE,
            "device": DEVICE,
            "model_path": MODEL_PATH,
        }
    )


def clip(v, lo, hi):
    return int(max(lo, min(hi, v)))


def box_from_xy(w, h, x, y, half):
    x1 = clip(x - half, 0, w - 1)
    y1 = clip(y - half, 0, h - 1)
    x2 = clip(x + half, 0, w - 1)
    y2 = clip(y + half, 0, h - 1)
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class Box:
    def __init__(self, b):
        self.xyxy = [
            np.array([b["x1"], b["y1"], b["x2"], b["y2"]], dtype=np.float32)
        ]


@app.route("/estimate_depth", methods=["POST"])
def estimate_depth():
    load_model()

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

        # Priority: bounding_box → x/y
        if "bounding_box" in request.form:
            try:
                bbox = json.loads(request.form["bounding_box"])
            except Exception:
                return jsonify({"error": "invalid bounding_box JSON"}), 400

            box = Box(bbox)
            depth_feet = float(_depth_model.estimate_depth(frame, box))
            return jsonify(
                {
                    "status": "success",
                    "mode": "bbox",
                    "depth_meters": depth_feet / 3.28084,
                    "depth_feet": depth_feet,
                }
            )

        if "x" not in request.form or "y" not in request.form:
            return jsonify({"error": "x and y are required"}), 400

        x = int(request.form["x"])
        y = int(request.form["y"])

        bbox = box_from_xy(w, h, x, y, point_box)
        box = Box(bbox)

        depth_feet = float(_depth_model.estimate_depth(frame, box))

        return jsonify(
            {
                "status": "success",
                "mode": "point",
                "x": x,
                "y": y,
                "sample_box": bbox,
                "depth_meters": depth_feet / 3.28084,
                "depth_feet": depth_feet,
            }
        )

    # ---------------- legacy JSON (base64) ----------------
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid JSON"}), 400

    if "image" not in data or "bounding_box" not in data:
        return jsonify({"error": "image and bounding_box required"}), 400

    try:
        img_bytes = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "invalid base64 image"}), 400

    box = Box(data["bounding_box"])
    depth_feet = float(_depth_model.estimate_depth(frame, box))

    return jsonify(
        {
            "status": "success",
            "mode": "bbox_json",
            "depth_meters": depth_feet / 3.28084,
            "depth_feet": depth_feet,
        }
    )
