import torch
import cv2
import numpy as np
from PIL import Image
import depth_pro

from src.models.depth_model import DepthModel


class MlDepthPro(DepthModel):
    """
    Apple ML Depth Pro model wrapper for estimating object depth from an image.

    Key design choices for Azure Container Apps:
    - Lazy-load the model (do not load at import/startup)
    - Use Apple DepthProConfig object (never pass dict config)
    - Use Azure Files mounted checkpoint via model_path (MODEL_PATH env var)
    """

    def __init__(
        self,
        device: str | None = None,
        model_type: str | None = None,
        model_path: str | None = None,
        **_kwargs,
    ):
        # Pick device
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model_type=model_type, device=device)

        # Azure Files mount checkpoint path
        self.model_path = model_path

        # Lazy-loaded members
        self.model = None
        self.transformer = None

    # ---------------------------
    # Model loading
    # ---------------------------
    def load_model(self) -> None:
        """Load Apple ML Depth Pro model + transforms (lazy)."""
        if self.model is not None and self.transformer is not None:
            return

        print(f"🔧 Using {self.device.upper()} to load and run ML Depth Pro model...")

        # Ensure we always pass a real DepthProConfig object (not a dict)
        DepthProConfig = getattr(depth_pro, "DepthProConfig", None)
        if DepthProConfig is None:
            raise RuntimeError(
                "Apple Depth Pro package not installed correctly: depth_pro.DepthProConfig is missing. "
                "Ensure requirements install: git+https://github.com/apple/ml-depth-pro.git "
                "and uninstall conflicting PyPI depth_pro/depth-pro packages."
            )

        cfg = DepthProConfig()

        if self.model_path:
            cfg.checkpoint_uri = self.model_path

        try:
            model, transform = depth_pro.create_model_and_transforms(
                config=cfg,
                device=torch.device(self.device),
                precision=torch.float32,
            )
            model.eval()

            self.model = model
            self.transformer = transform

            print(f"✅ Depth Pro model loaded successfully on {self.device.upper()}.")

        except (FileNotFoundError, EOFError, torch.serialization.pickle.UnpicklingError) as e:
            print(f"❌ Model load error: {type(e).__name__}: {e}")
            print("Check that MODEL_PATH points to the mounted checkpoint file.")
            raise
        except Exception as e:
            print(f"❌ Unexpected error during model initialization: {e}")
            raise

    # ---------------------------
    # Preprocess
    # ---------------------------
    def preprocess_image(self, frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input tensor for Depth Pro.

        NOTE: This keeps your original logic but ensures model is loaded first.
        """
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call load_model() first.")

        frame = cv2.resize(frame, (320, 320))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        f_px = max(pil_img.size) * 1.2
        f_px_tensor = torch.tensor([f_px], dtype=torch.float32, device=self.device)

        img_t = self.transformer(pil_img).unsqueeze(0).to(self.device)
        return img_t, f_px_tensor

    # ---------------------------
    # Inference
    # ---------------------------
    def estimate_depth(self, frame: np.ndarray, box: np.ndarray) -> float:
        """
        Estimate object depth in feet given a frame and a YOLO-like box object
        that exposes `box.xyxy[0] -> [x1,y1,x2,y2]`.
        """
        try:
            # Lazy load model on first request
            self.load_model()

            print("Estimating the depth...")
            img_t, f_px_tensor = self.preprocess_image(frame=frame)

            with torch.no_grad():
                pred = self.model.infer(img_t, f_px=f_px_tensor)

            depth_m = pred["depth"]
            if isinstance(depth_m, torch.Tensor):
                depth_m = depth_m.detach().cpu().numpy()

            depth_m = np.squeeze(depth_m)
            if depth_m.ndim != 2:
                print(f"Warning: depth map is not 2D, shape={depth_m.shape}, continuing anyway")

            depth_ft = MlDepthPro.calculate_median_depth(depth=depth_m, box=box)
            return float(depth_ft)

        except Exception as err:
            print(f"❌ Unexpected error during inference: {err}")
            raise

    # ---------------------------
    # Utility
    # ---------------------------
    @staticmethod
    def calculate_median_depth(depth: np.ndarray, box: np.ndarray) -> float:
        """
        Compute median depth within the bounding box. Returns feet.
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        h, w = depth.shape

        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        obj_depth = depth[y1:y2, x1:x2]
        if obj_depth.size == 0:
            print("Warning: empty box region; returning 0.")
            return 0.0

        median_depth_m = float(np.nanmedian(obj_depth))
        print(f"Estimated object depth is: {median_depth_m} meters")

        depth_ft = median_depth_m * 3.28084
        print(f"Estimated object depth is: {depth_ft} feet")
        return float(depth_ft)
