import torch
import cv2
import numpy as np
from PIL import Image
import depth_pro

from src.models.depth_model import DepthModel


class MlDepthPro(DepthModel):
    """
    Implements ML Depth Pro model to estimate depth of an object from an image.
    """

    def __init__(self, device: str = None, model_type: str = None, model_path: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model_type=model_type, device=device)

        # If using Azure Files mount, pass MODEL_PATH here:
        self.model_path = model_path

        self.model, self.transformer = self.load_model(device=device, model_path=model_path)

    def load_model(self, device: str, model_path: str | None = None) -> tuple:
        """Load ML Depth Pro model + transforms."""
        print(f"🔧 Using {device.upper()} to load and run ML Depth Pro model...")

        try:
            # Prefer Apple Depth Pro config object if available
            config = None
            if model_path:
                try:
                    DepthProConfig = getattr(depth_pro, "DepthProConfig", None)
                    if DepthProConfig is not None:
                        config = DepthProConfig(checkpoint_uri=model_path)
                    else:
                        # Fallback: Apple code also supports dict-like config in some examples
                        config = {"checkpoint_uri": model_path}
                except Exception:
                    config = {"checkpoint_uri": model_path}

            if config is not None:
                model, transform = depth_pro.create_model_and_transforms(
                    config=config,
                    device=torch.device(device),
                    precision=torch.float32,
                )
            else:
                model, transform = depth_pro.create_model_and_transforms(
                    device=torch.device(device),
                    precision=torch.float32,
                )

            model.eval()
            print(f"✅ Depth Pro model loaded successfully on {device.upper()}.")
            return model, transform

        except (FileNotFoundError, EOFError, torch.serialization.pickle.UnpicklingError) as e:
            print(f"❌ Model load error: {type(e).__name__}: {e}")
            print("Check that checkpoint_uri points to the mounted file.")
            raise


    def preprocess_image(self, frame: np.ndarray) -> tuple:
        frame = cv2.resize(frame, (320, 320))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        f_px = max(pil_img.size) * 1.2
        f_px_tensor = torch.tensor([f_px], dtype=torch.float32, device=self.device)

        img_t = self.transformer(pil_img).unsqueeze(0).to(self.device)
        return img_t, f_px_tensor

    def estimate_depth(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        try:
            print("Estimating the depth...")
            img_t, f_px_tensor = self.preprocess_image(frame=frame)
            with torch.no_grad():
                pred = self.model.infer(img_t, f_px=f_px_tensor)

            depth_m = pred["depth"]
            if isinstance(depth_m, torch.Tensor):
                depth_m = depth_m.detach().cpu().numpy()

            depth_m = np.squeeze(depth_m)
            if depth_m.ndim != 2:
                print(f"Warning: depth map is not 2D, shape={depth_m.shape}, skipping frame")

            depth = MlDepthPro.calculate_median_depth(depth=depth_m, box=box)
            return depth

        except Exception as err:
            print(f"❌ Unexpected error during inference: {err}")
            raise

    @staticmethod
    def calculate_median_depth(depth: np.array, box: np.array):
        median_depth = 0
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        h, w = depth.shape

        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        obj_depth = depth[y1:y2, x1:x2]
        if obj_depth.size > 0:
            median_depth = np.nanmedian(obj_depth)

        print(f"Estimated object depth is: {median_depth} meters")
        depth_ft = median_depth * 3.28084
        print(f"Estimated object depth is: {depth_ft} feet")
        return depth_ft
