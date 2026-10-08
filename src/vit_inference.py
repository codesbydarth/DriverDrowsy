"""Vision Transformer (ViT) production inference module for Phase 3.

Provides ViTInferenceEngine to load the production ViT checkpoint once at startup
and perform deterministic, non-blocking single-frame inference on preprocessed
facial ROIs. Strictly outputs the complete 4-class softmax probability vector:
    [P(ALERT), P(LOW_VIGILANCE), P(DROWSY), P(MICROSLEEP)]

Architectural Invariants:
    - Loads production checkpoint once at initialization; never per-frame.
    - Operates in strict evaluation mode (torch.no_grad / eval).
    - Never uses hard argmax as the output; returns continuous evidence vector.
    - If production checkpoint is missing, logs error and fails safely (NO untrained fallback).
    - Thread-safe and portable across laptop, Windows, Linux, and Jetson platforms.
"""

from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from models.vit_classifier import ViTClassifier
from src.dataset import get_vit_transforms
from utils.config import DEFAULT_CONFIG, AppConfig, ProjectState
from utils.logger import setup_logger

logger = setup_logger("vit_inference")


class ViTInferenceEngine:
    """Production ViT inference runner for 4-class driver vigilance estimation."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[nn.Module] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        """Initialize ViT inference engine.

        Loads the trained checkpoint once at startup into evaluation mode.
        If the checkpoint is missing, logs an error and raises FileNotFoundError.
        Under no circumstances does it silently instantiate an untrained model
        in production. For unit testing, a lightweight model may be explicitly injected.

        Args:
            checkpoint_path: Path to vit_best_production.pt.
                             Defaults to config.checkpoints_dir / "vit_best_production.pt".
            device: Target torch device ('cpu', 'cuda', or torch.device).
                    Defaults to auto-detecting CUDA if available.
            model: Optional pre-constructed model instance (used for testing/mocking).
            config: Master application configuration instance.
        """
        self.config = config or DEFAULT_CONFIG
        self._lock = threading.Lock()

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[nn.Module] = None
        self.metadata: Dict[str, Any] = {}
        self._checkpoint_path: Optional[Path] = None

        if model is not None:
            # Model injection mode (primarily for testing and mocking)
            self.model = model.to(self.device)
            self.model.eval()
            self._checkpoint_path = Path("in_memory_model")
            logger.info("ViTInferenceEngine initialized with injected model instance on %s", self.device)
        else:
            # Checkpoint loading mode
            target_path = Path(checkpoint_path) if checkpoint_path else (
                self.config.checkpoints_dir / "vit_best_production.pt"
            )
            self._checkpoint_path = target_path.resolve()

            if not target_path.exists():
                logger.error(
                    "Production ViT checkpoint not found at: %s. "
                    "Per Phase 3 specifications, an untrained model fallback is strictly forbidden.",
                    self._checkpoint_path,
                )
                raise FileNotFoundError(
                    f"Production ViT checkpoint not found at '{self._checkpoint_path}'. "
                    "Phase 3 requires the trained production checkpoint; "
                    "untrained fallback models are strictly forbidden."
                )

            # Load checkpoint using existing ViTClassifier loader
            loaded_classifier, metadata = ViTClassifier.load_checkpoint(
                self._checkpoint_path,
                device=self.device
            )
            loaded_classifier.eval()
            self.model = loaded_classifier
            self.metadata = metadata
            logger.info(
                "ViTInferenceEngine successfully loaded production checkpoint from %s on %s",
                self._checkpoint_path,
                self.device
            )

        # Preprocessing transforms (ImageNet normalization, 224x224 RGB)
        self.transform = get_vit_transforms(
            is_training=False,
            image_size=self.config.vit.image_size,
            mean=self.config.vit.image_mean,
            std=self.config.vit.image_std,
        )

    @property
    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        return self.model is not None

    @property
    def checkpoint_path(self) -> Optional[Path]:
        """Return path of loaded checkpoint."""
        return self._checkpoint_path

    def infer(
        self,
        face_roi: Union[np.ndarray, Image.Image],
        is_bgr: bool = False,
    ) -> np.ndarray:
        """Execute single-frame ViT inference and return 4-class probability vector.

        Args:
            face_roi: Cropped facial region as a NumPy array or PIL Image.
            is_bgr: If True and face_roi is a NumPy array, converts BGR to RGB.

        Returns:
            NumPy array of shape (4,) representing continuous probabilities:
                [P(ALERT), P(LOW_VIGILANCE), P(DROWSY), P(MICROSLEEP)]
            Sum of probabilities equals 1.0 within numerical precision.
            NEVER returns hard argmax.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If input image is empty or invalid.
            TypeError: If input image is of unsupported type.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError("ViTInferenceEngine model is not loaded.")

        if face_roi is None:
            raise ValueError("Input face_roi cannot be None.")

        # Input normalization to RGB PIL Image
        if isinstance(face_roi, Image.Image):
            pil_img = face_roi.convert("RGB")
        elif isinstance(face_roi, np.ndarray):
            if face_roi.size == 0:
                raise ValueError("Input face_roi is an empty NumPy array.")
            if is_bgr and face_roi.ndim == 3 and face_roi.shape[2] == 3:
                rgb_arr = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_arr)
            else:
                pil_img = Image.fromarray(face_roi).convert("RGB")
        else:
            raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(face_roi)}")

        # Transform and device placement
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Thread-safe forward pass without gradients
        with self._lock, torch.no_grad():
            output = self.model(input_tensor)
            logits = output.logits if hasattr(output, "logits") else output
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)

        if probs.shape != (4,):
            raise ValueError(
                f"ViT output shape mismatch: expected (4,), got {probs.shape}"
            )

        return probs


# Convenient alias
ViTInference = ViTInferenceEngine
