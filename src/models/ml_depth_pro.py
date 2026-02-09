import torch
import cv2
import numpy as np
from PIL import Image
from typing import Any

from src.models.depth_model import DepthModel


class MlDepthPro(DepthModel):
    """
    Apple ML Depth Pro model wrapper for estimating object depth from an image.

    Design goals (Azure Container Apps safe):
    - Lazy-load model (cold start friendly)
    - DO NOT depend on DepthProConfig (not stable across installs)
    - Load checkpoint from Azure Files mount (MODEL_PATH)
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

    # --------------------------------------------------
    # Model loading (lazy)
    # --------------------------------------------------
    def load_model(self) -> None:
        if self.model is not None and self.transformer is not None:
            return

        try:
            import depth_pro
        except Exception as e:
            raise RuntimeError(
                "Apple ml-depth-pro package is not installed correctly. "
                "Ensure Dockerfile installs: git+https://github.com/apple/ml-depth-pro.git"
            ) from e

        if not hasattr(depth_pro, "create_model_and_transforms"):
            raise RuntimeError(
                "depth_pro package does not expose create_model_and_transforms. "
                "Wrong package installed."
            )

        # Ensure checkpoint exists (Azure Files mount)
        if self.model_path:
            if not isinstance(self.model_path, str):
                raise RuntimeError("model_path must be a string")
            # file existence check happens in service layer, not here

        # ---- Create model ----
        try:
            # Preferred path: pass checkpoint_uri directly (supported in current Apple repo)
            model, transform = depth_pro.create_model_and_transforms(
                checkpoint_uri=self.model_path,
                device=torch.device(self.device),
                precision=torch.float32,
            )
        except TypeError:
            # Fallback for older signatures
            model, transform = depth_pro.create_model_and_transforms(
                device=torch.device(self.device),
                precision=torch.float32,
            )

        model = model.to(self.device)
        model.eval()

        self.model = model
        self.transformer = transform

    # --------------------------------------------------
    # Preprocess
    # --------------------------------------------------
    def preprocess_image(self, frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call load_model() first.")

        # DO NOT resize whole frame unless bbox is scaled too
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Apple Depth Pro expects focal length estimate
        f_px = max(pil_img.size) * 1.2
        f_px_tensor = torch.tensor([f_px], dtype=torch.float32, device=self.device)

        img_t = self.transformer(pil_img).unsqueeze(0).to(self.device)
        return img_t, f_px_tensor

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def estimate_depth(self, frame: np.ndarray, box: Any) -> float:
        self.load_model()

        img_t, f_px_tensor = self.preprocess_image(frame)

        with torch.no_grad():
            pred = self.model.infer(img_t, f_px=f_px_tensor)

        depth_m = pred.get("depth")
        if depth_m is None:
            raise RuntimeError("Depth Pro output missing 'depth' key")

        if isinstance(depth_m, torch.Tensor):
            depth_m = depth_m.detach().cpu().numpy()

        depth_m = np.squeeze(depth_m)
        if depth_m.ndim != 2:
            raise RuntimeError(f"Depth map is not 2D (shape={depth_m.shape})")

        return float(self.calculate_median_depth(depth_m, box))

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------
    @staticmethod
    def calculate_median_depth(depth: np.ndarray, box: Any) -> float:
        """
        Compute median depth inside bounding box.
        Returns depth in FEET.
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        h, w = depth.shape

        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        obj_depth = depth[y1:y2, x1:x2]
        if obj_depth.size == 0:
            return 0.0

        median_depth_m = float(np.nanmedian(obj_depth))
        return median_depth_m * 3.28084
