
import os
from typing import Any
from dotenv import load_dotenv
import yaml

from src.models.ml_depth_pro import MlDepthPro
from src.models.midas import DepthEstimator


# Load .env file into environment variables
load_dotenv()

# Access variables from environment
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")

depth_model_name_to_class_mapping = {
    "ml_depth_pro": MlDepthPro,
    "midas": DepthEstimator
}

def load_config(config_path: str) -> Any:
    """
    Safely reads yaml files and returns the resulting Python object

    :param config_path: path of the config file to load.
    :return: resulting python object after yaml read.
    """
    with open(file=config_path, mode="r", encoding="utf-8") as conf_file:
        return yaml.safe_load(conf_file)
    
def get_depth_model(config: Any, model_path: str | None = None) -> Any:
    """Creates depth model config and instantiates the model class"""
    model_config = config["depth_model"]
    name = model_config["name"]
    device = model_config["device"]
    model_params = model_config.get("params", {}).copy()

    # Allow runtime override (Azure Files mount)
    if model_path:
        model_params["model_path"] = model_path

    if name in depth_model_name_to_class_mapping:
        return depth_model_name_to_class_mapping[name](device=device, **model_params)

    raise ValueError(f"Unknown Depth model: {name}")

    