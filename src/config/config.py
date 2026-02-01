import os
from typing import Any, Dict, Optional

import yaml

# Local-dev only: don't fail if python-dotenv isn't installed in prod
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.models.ml_depth_pro import MlDepthPro


def load_config(config_path: str) -> Any:
    if not config_path or not os.path.exists(config_path):
        return None
    with open(file=config_path, mode="r", encoding="utf-8") as conf_file:
        return yaml.safe_load(conf_file)

def get_depth_model(config: Optional[Dict[str, Any]] = None, model_path: str | None = None) -> Any:
    """
    Create and return the depth model.

    Priority:
    1) ACA env vars (MODEL_TYPE, DEVICE, MODEL_PATH)
    2) config.yaml (if provided)
    """
    # --- Defaults from env (ACA-friendly) ---
    name = (os.getenv("MODEL_TYPE") or "ml_depth_pro").strip()
    device = (os.getenv("DEVICE") or "cpu").strip()
    effective_model_path = model_path or os.getenv("MODEL_PATH")

    model_params: Dict[str, Any] = {}
    if effective_model_path:
        model_params["model_path"] = effective_model_path

    # --- Override from config file if present ---
    if config and isinstance(config, dict) and "depth_model" in config:
        model_config = config["depth_model"] or {}
        name = model_config.get("name", name)
        device = model_config.get("device", device)
        model_params.update((model_config.get("params") or {}).copy())

        # Allow runtime override (Azure Files mount)
        if effective_model_path:
            model_params["model_path"] = effective_model_path

    # --- Instantiate DepthPro only ---
    if name == "ml_depth_pro":
        return MlDepthPro(device=device, **model_params)

    raise ValueError(f"Unknown Depth model: {name}")
