from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class DepthModel(ABC):
    """Abstract base class defining a depth estimation interface."""
    def __init__(
            self,
            model_type: str = None,
            device: str = None,
    ):
        self.model_type = model_type
        self.device = device
    
    @abstractmethod
    def load_model(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError()