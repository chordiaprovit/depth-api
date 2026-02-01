import torch
import cv2
import numpy as np
from PIL import Image
from typing import Any

from src.models.depth_model import DepthModel


class MlDepthPro(DepthModel):
    """
    Apple ML Depth Pro model wrapper for estimating object depth from an image.

    - Lazy-load model (safe for ACA cold starts)
    - Uses DepthProConfig object (never dict)
    - Loads checkpoint from Azure Files mount (MODEL_PATH)
    """

    def __init__(
        self,
        device: str | None = None,
        model_type: str | None = None,
        model_path: str | None = None,
        **_kwargs,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model_type=model_type, device=device)

        self.model_path = model_path
        self.model = None
        self.transformer = None

    def load_model(self) -> None:
        if self.model is not None and self.transformer is not None:
            return

        try:
            import depth_pro  # lazy import
        except Exception as e:
            raise RuntimeError(
                "depth_pro is not installed in the container. "
                "Ensure Dockerfile installs git and requirements include "
                "git+https://github.com/apple/ml-depth-pro.git"
            ) from e

        DepthProConfig = getattr(depth_pro, "DepthProConfig", None)
        if DepthProConfig is None:
            raise RuntimeError("depth_pro imported but DepthProConfig missing (wrong package).")

        cfg = DepthProConfig()
        if self.model_path:
            cfg.checkpoint_uri = self.model_path

        model, transform = depth_pro.create_model_and_transforms(
            config=cfg,
            device=torch.device(self.device),
            precision=torch.float32,
        )

        model = model.to(self.device)
        model.eval()

        self.model, self.transformer = model, transform

    def preprocess_image(self, frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call load_model() first.")

        # IMPORTANT: do NOT resize the whole frame here unless you also scale bbox
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        f_px = max(pil_img.size) * 1.2
        f_px_tensor = torch.tensor([f_px], dtype=torch.float32, device=self.device)

        img_t = self.transformer(pil_img).unsqueeze(0).to(self.device)
        return img_t, f_px_tensor

    def estimate_depth(self, frame: np.ndarray, box: Any) -> float:
        self.load_model()

        img_t, f_px_tensor = self.preprocess_image(frame=frame)

        with torch.no_grad():
            pred = self.model.infer(img_t, f_px=f_px_tensor)

        depth_m = pred["depth"]
        if isinstance(depth_m, torch.Tensor):
            depth_m = depth_m.detach().cpu().numpy()

        depth_m = np.squeeze(depth_m)
        if depth_m.ndim != 2:
            raise RuntimeError(f"Depth map is not 2D. shape={depth_m.shape}")

        return float(self.calculate_median_depth(depth=depth_m, box=box))

    @staticmethod
    def calculate_median_depth(depth: np.ndarray, box: Any) -> float:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        h, w = depth.shape

        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        obj_depth = depth[y1:y2, x1:x2]
        if obj_depth.size == 0:
            return 0.0

        median_depth_m = float(np.nanmedian(obj_depth))
        return float(median_depth_m * 3.28084)
