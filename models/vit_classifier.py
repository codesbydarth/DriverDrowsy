"""Vision Transformer (ViT) classifier for driver drowsiness detection.

Wraps google/vit-base-patch16-224 with a 4-class classification head:
    0 = ALERT
    1 = LOW_VIGILANCE
    2 = DROWSY
    3 = MICROSLEEP

Provides:
    - Two-stage transfer learning (freeze_backbone, unfreeze_deep_layers)
    - Checkpoint serialization and restoration with metadata validation
    - Standalone single-frame and batch inference (predict)
    - Parameter counting and device placement
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import ViTConfig as HfViTConfig, ViTForImageClassification

from src.dataset import get_vit_transforms
from utils.config import DEFAULT_CONFIG, ProjectState, ViTConfig
from utils.logger import setup_logger

logger = setup_logger("vit_classifier")


class ViTClassifier(nn.Module):
    """Vision Transformer classifier for 4-class driver vigilance states."""

    def __init__(
        self,
        config: Optional[ViTConfig] = None,
        pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize ViTClassifier.

        Args:
            config: ViTConfig hyperparameters. Defaults to DEFAULT_CONFIG.vit.
            pretrained: If True, load pre-trained weights from HuggingFace Hub;
                        otherwise initialize random weights from ViTConfig.
            device: Target torch device. If None, auto-selects CUDA if available.
        """
        super().__init__()
        self.config = config or DEFAULT_CONFIG.vit

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = self.config.num_classes
        self.id2label = {st.value: st.label for st in ProjectState}
        self.label2id = {st.label: st.value for st in ProjectState}

        if pretrained:
            logger.info("Loading pre-trained ViT model: %s", self.config.model_name)
            self.model = ViTForImageClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.num_classes,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            logger.info("Initializing ViT model from scratch (pretrained=False)")
            hf_cfg = HfViTConfig.from_pretrained(self.config.model_name)
            hf_cfg.num_labels = self.num_classes
            hf_cfg.id2label = self.id2label
            hf_cfg.label2id = self.label2id
            self.model = ViTForImageClassification(hf_cfg)

        self.to(self.device)
        self.transform = get_vit_transforms(
            is_training=False,
            image_size=self.config.image_size,
            mean=self.config.image_mean,
            std=self.config.image_std,
        )

        counts = self.get_parameter_counts()
        logger.info(
            "ViTClassifier initialized on %s: Total=%d, Trainable=%d, Frozen=%d",
            self.device, counts["total"], counts["trainable"], counts["frozen"]
        )

    def freeze_backbone(self) -> None:
        """Stage 1: Freeze all backbone layers, leaving only the classification head trainable."""
        # ViT backbone consists of embeddings and encoder layers
        for param in self.model.vit.parameters():
            param.requires_grad = False

        # Ensure classification head is trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        counts = self.get_parameter_counts()
        logger.info(
            "Stage 1 Frozen Backbone: Trainable params = %d (head only), Frozen params = %d",
            counts["trainable"], counts["frozen"]
        )

    def unfreeze_deep_layers(self, num_blocks: Optional[int] = None) -> None:
        """Stage 2: Unfreeze the last N transformer encoder blocks for fine-tuning.

        Args:
            num_blocks: Number of deep encoder blocks to unfreeze.
                        Defaults to self.config.unfreeze_blocks (2).
        """
        # Resolve encoder layers across transformers versions
        if hasattr(self.model.vit, "layers"):
            encoder_layers = self.model.vit.layers
        elif hasattr(self.model.vit, "encoder") and hasattr(self.model.vit.encoder, "layer"):
            encoder_layers = self.model.vit.encoder.layer
        else:
            raise AttributeError("Cannot find transformer layers in ViTModel")

        total_layers = len(encoder_layers)
        blocks = num_blocks if num_blocks is not None else self.config.unfreeze_blocks

        # First freeze all backbone layers
        for param in self.model.vit.parameters():
            param.requires_grad = False

        # Unfreeze the last N layers
        start_idx = max(0, total_layers - blocks)
        for i in range(start_idx, total_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

        # Unfreeze layernorm if present
        if hasattr(self.model.vit, "layernorm"):
            for param in self.model.vit.layernorm.parameters():
                param.requires_grad = True

        # Ensure classifier head remains trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        counts = self.get_parameter_counts()
        logger.info(
            "Stage 2 Selective Fine-Tuning: Unfrozen last %d blocks (layers %d-%d). "
            "Trainable params = %d, Frozen params = %d",
            blocks, start_idx, total_layers - 1, counts["trainable"], counts["frozen"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT model.

        Args:
            x: Input batch tensor of shape (B, 3, 224, 224).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        outputs = self.model(pixel_values=x)
        return outputs.logits

    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        apply_softmax: bool = True,
    ) -> Dict[str, Any]:
        """Perform standalone inference on a single facial image.

        Args:
            image: Single image as numpy array (RGB) or PIL Image.
            apply_softmax: If True, compute probabilities using softmax.

        Returns:
            Dictionary containing:
                - 'predicted_state': Canonical ProjectState enum
                - 'state_id': Integer class index (0-3)
                - 'state_name': String label name
                - 'confidence': Probability of predicted class
                - 'probabilities': Dict mapping each state label to its probability
                - 'logits': Raw output logits list
        """
        self.eval()

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(image)}")

        curr_device = next(self.parameters()).device
        self.device = curr_device
        input_tensor = self.transform(pil_img).unsqueeze(0).to(curr_device)

        with torch.no_grad():
            logits = self.forward(input_tensor)
            if apply_softmax:
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            else:
                probs = logits.squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_state = ProjectState(pred_idx)

        prob_dict = {
            ProjectState(i).label: float(probs[i]) for i in range(self.num_classes)
        }

        return {
            "predicted_state": pred_state,
            "state_id": pred_idx,
            "state_name": pred_state.label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "logits": logits.squeeze(0).cpu().numpy().tolist(),
        }

    def get_parameter_counts(self) -> Dict[str, int]:
        """Compute parameter counts for logging and auditing."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint with configuration and metadata.

        Args:
            filepath: Destination path (.pt or .pth).
            metadata: Additional training metadata (epoch, val_f1, loss, etc.).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else dict(self.config),
            "id2label": self.id2label,
            "label2id": self.label2id,
            "metadata": metadata or {},
        }

        torch.save(checkpoint, path)
        logger.info("Saved ViT checkpoint to %s (metadata keys: %s)", path, list((metadata or {}).keys()))

    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple["ViTClassifier", Dict[str, Any]]:
        """Load ViTClassifier instance from a saved checkpoint file.

        Args:
            filepath: Path to checkpoint file.
            device: Target torch device.

        Returns:
            Tuple of (loaded_vit_classifier, metadata_dict).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)

        cfg_dict = checkpoint.get("config", {})
        config = ViTConfig(**{k: v for k, v in cfg_dict.items() if hasattr(ViTConfig, k)})

        # Initialize instance with random weights first, then load state_dict
        classifier = cls(config=config, pretrained=False, device=target_device)
        classifier.load_state_dict(checkpoint["state_dict"])
        classifier.eval()

        metadata = checkpoint.get("metadata", {})
        logger.info("Loaded ViT checkpoint from %s on %s", path, target_device)
        return classifier, metadata
