#!/usr/bin/env python3
"""Vision Transformer (ViT) Production Training Pipeline for Driver Vigilance States.

Implements Phase 2C requirements:
1. Model: google/vit-base-patch16-224 (85.8M params) with 4-class classification head:
   - 0 = ALERT
   - 1 = LOW_VIGILANCE
   - 2 = DROWSY
   - 3 = MICROSLEEP
2. Balanced inverse-frequency class-weighted CrossEntropyLoss calculated
   strictly from the training split only.
3. Two-stage transfer learning:
   - Stage 1: Freeze ViT backbone; train classification head (LR = 1e-3, 5 epochs).
   - Stage 2: Unfreeze deepest 2 transformer blocks; fine-tune with reduced LR (LR = 2e-5, 10 epochs).
4. Training-set only augmentations (RandomHorizontalFlip, subtle ColorJitter).
   Validation and test preprocessing is strictly deterministic.
5. Per-epoch monitoring:
   - Training loss, training accuracy
   - Validation loss, validation accuracy, balanced accuracy, macro-F1, per-class metrics
   - Checkpoint selection strictly based on validation macro-F1.
6. Held-out test set evaluation executed exactly ONCE on best checkpoint.
7. Standalone inference latency and batch throughput benchmarking.
8. Preserves production checkpoint separately at models/checkpoints/vit_best_production.pt.
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vit_classifier import ViTClassifier
from src.dataset import (
    DrowsinessDataset,
    calculate_class_weights,
    get_vit_transforms,
    load_fallback_sample_ids,
    load_manifest_csv,
    resolve_path,
    to_project_relative_path,
    validate_dataset_directory,
)
from src.landmark_detector import FaceMeshDetector
from utils.config import DEFAULT_CONFIG, FACE_ROI_MARGIN, ProjectState, ViTConfig
from utils.logger import setup_logger

logger = setup_logger("train_vit")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    stage_name: str,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Run a single training epoch with automatic mixed precision.

    Returns:
        Tuple of (epoch_loss, epoch_accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    use_cuda = device.type == "cuda"
    pbar = tqdm(dataloader, desc=f"[{stage_name}] Epoch {epoch} Training")
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total_samples += batch_size

        batch_acc = (preds == labels).float().mean().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc*100:.1f}%"})

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> Dict[str, Any]:
    """Evaluate model on a dataset split with comprehensive metrics.

    Returns:
        Structured metrics dictionary.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())

    mean_loss = running_loss / total_samples if total_samples > 0 else 0.0
    acc = float(accuracy_score(all_targets, all_preds)) if all_targets else 0.0
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds)) if all_targets else 0.0
    macro_p = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_r = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    target_names = [ProjectState(i).name for i in range(4)]
    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(4)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(all_targets, all_preds, labels=list(range(4)))

    return {
        "loss": mean_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "targets": all_targets,
        "preds": all_preds,
    }


def run_benchmarks(
    model: ViTClassifier,
    device: torch.device,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Benchmark inference latency and throughput on current device and CPU.

    Measures:
    1. Single-image ViT inference latency (CUDA and CPU).
    2. Batch inference throughput (batch size 32).
    3. End-to-end FaceMesh + ViT latency.
    """
    logger.info("=== Running Standalone Inference Benchmarks ===")
    results: Dict[str, Any] = {}

    dummy_pil = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 1. CUDA Benchmarks (if available)
    if device.type == "cuda":
        model.to(device)
        model.eval()

        # Single-image latency
        latencies_cuda: List[float] = []
        # Warmup
        for _ in range(10):
            _ = model.predict(dummy_pil)
        torch.cuda.synchronize()

        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model.predict(dummy_pil)
            torch.cuda.synchronize()
            latencies_cuda.append((time.perf_counter() - t0) * 1000)

        mean_lat_cuda = float(np.mean(latencies_cuda))
        p95_lat_cuda = float(np.percentile(latencies_cuda, 95))
        fps_single_cuda = 1000.0 / mean_lat_cuda if mean_lat_cuda > 0 else 0.0

        # Batch throughput (batch size 32)
        batch_tensor = torch.randn(32, 3, 224, 224, device=device)
        for _ in range(5):
            with torch.no_grad():
                _ = model(batch_tensor)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        batch_runs = 20
        for _ in range(batch_runs):
            with torch.no_grad():
                _ = model(batch_tensor)
        torch.cuda.synchronize()
        total_time_batch = time.perf_counter() - t0
        batch_fps_cuda = (batch_runs * 32) / total_time_batch

        results["cuda"] = {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "single_latency_mean_ms": mean_lat_cuda,
            "single_latency_p95_ms": p95_lat_cuda,
            "single_throughput_fps": fps_single_cuda,
            "batch_32_throughput_fps": batch_fps_cuda,
        }
        logger.info(
            "CUDA Benchmark: Single Latency=%.2f ms (%.1f FPS), Batch-32 Throughput=%.1f FPS",
            mean_lat_cuda, fps_single_cuda, batch_fps_cuda
        )

    # 2. CPU Benchmarks
    cpu_device = torch.device("cpu")
    model.to(cpu_device)
    model.eval()

    latencies_cpu: List[float] = []
    # Warmup
    for _ in range(5):
        _ = model.predict(dummy_pil)

    cpu_runs = min(num_runs, 30)
    for _ in range(cpu_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy_pil)
        latencies_cpu.append((time.perf_counter() - t0) * 1000)

    mean_lat_cpu = float(np.mean(latencies_cpu))
    p95_lat_cpu = float(np.percentile(latencies_cpu, 95))
    fps_single_cpu = 1000.0 / mean_lat_cpu if mean_lat_cpu > 0 else 0.0

    results["cpu"] = {
        "single_latency_mean_ms": mean_lat_cpu,
        "single_latency_p95_ms": p95_lat_cpu,
        "single_throughput_fps": fps_single_cpu,
    }
    logger.info(
        "CPU Benchmark: Single Latency=%.2f ms (%.1f FPS)",
        mean_lat_cpu, fps_single_cpu
    )

    # Restore model to original device
    model.to(device)

    # 3. End-to-End FaceMesh + ViT pipeline benchmark
    detector = FaceMeshDetector(static_image_mode=False, max_num_faces=1, refine_landmarks=False)
    e2e_latencies: List[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = detector.detect(cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB))
        _ = model.predict(dummy_pil)
        e2e_latencies.append((time.perf_counter() - t0) * 1000)

    mean_e2e_lat = float(np.mean(e2e_latencies))
    results["end_to_end"] = {
        "latency_mean_ms": mean_e2e_lat,
        "throughput_fps": 1000.0 / mean_e2e_lat if mean_e2e_lat > 0 else 0.0,
    }
    logger.info(
        "End-to-End (FaceMesh + ViT) Throughput: %.1f FPS (%.2f ms)",
        results["end_to_end"]["throughput_fps"], mean_e2e_lat
    )

    return results


def main() -> None:
    """CLI entry point for production ViT training."""
    parser = argparse.ArgumentParser(description="Train ViT Drowsiness Classifier (Production)")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_CONFIG.splits_dir,
        help="Path to directory containing train/val/test CSV manifests",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=DEFAULT_CONFIG.checkpoints_dir,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--production-checkpoint-name",
        type=str,
        default="vit_best_production.pt",
        help="Filename for the production trained checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Stage 1 epochs (default: 5)")
    parser.add_argument("--stage1-lr", type=float, default=1e-3, help="Stage 1 learning rate (default: 1e-3)")
    parser.add_argument("--stage2-epochs", type=int, default=10, help="Stage 2 epochs (default: 10)")
    parser.add_argument("--stage2-lr", type=float, default=2e-5, help="Stage 2 learning rate (default: 2e-5)")
    parser.add_argument("--unfreeze-blocks", type=int, default=2, help="Blocks to unfreeze in Stage 2 (default: 2)")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.dataset.random_seed, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-batch dry run for verification")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint .pt file to resume training from.",
    )

    args = parser.parse_args()

    # Reproducibility seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_manifest = resolve_path(args.splits_dir / "dev_train_manifest.csv")
    val_manifest = resolve_path(args.splits_dir / "dev_val_manifest.csv")
    test_manifest = resolve_path(args.splits_dir / "dev_test_manifest.csv")

    if not train_manifest.exists():
        logger.error(
            "NTHU dataset not found. Dataset preprocessing/training cannot be executed "
            "until the dataset is downloaded and configured."
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Production ViT training on device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA Device: %s (VRAM: %.2f GB)", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / (1024**3))

    # Load manifests
    train_samples = load_manifest_csv(train_manifest)
    val_samples = load_manifest_csv(val_manifest)
    test_samples = load_manifest_csv(test_manifest)

    if args.dry_run:
        train_samples = train_samples[:32]
        val_samples = val_samples[:16]
        test_samples = test_samples[:16]
        logger.info("DRY RUN MODE: Truncated samples for verification")

    logger.info(
        "Loaded manifests: Train=%d, Val=%d, Test=%d",
        len(train_samples), len(val_samples), len(test_samples)
    )

    # Calculate class weights strictly from training split
    class_weights = calculate_class_weights(train_samples, num_classes=4)
    logger.info(
        "Calculated training class weights (strictly from train split): %s (mean=%.2f)",
        [round(float(w), 4) for w in class_weights.tolist()], float(class_weights.mean())
    )

    # Setup datasets: training uses data augmentation, validation and test are deterministic
    train_transform = get_vit_transforms(is_training=True)
    eval_transform = get_vit_transforms(is_training=False)

    fallback_ids = load_fallback_sample_ids(args.splits_dir / "face_detection_failures.csv")
    if fallback_ids:
        logger.info("Loaded %d fallback face-crop sample IDs for audit tracking", len(fallback_ids))

    train_dataset = DrowsinessDataset(train_samples, transform=train_transform, use_processed=True, fallback_sample_ids=fallback_ids)
    val_dataset = DrowsinessDataset(val_samples, transform=eval_transform, use_processed=True, fallback_sample_ids=fallback_ids)
    test_dataset = DrowsinessDataset(test_samples, transform=eval_transform, use_processed=True, fallback_sample_ids=fallback_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Initialize ViT model
    model = ViTClassifier(device=device)
    train_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    val_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    param_counts = model.get_parameter_counts()
    logger.info("ViT Parameter Counts - Total: %d, Trainable: %d, Frozen: %d",
                param_counts["total"], param_counts["trainable"], param_counts["frozen"])

    best_val_f1 = -1.0
    best_val_metrics: Dict[str, Any] = {}
    best_checkpoint_path = args.checkpoints_dir / args.production_checkpoint_name
    training_start_time = time.time()

    # Resume state initialization
    resume_stage = None
    resume_completed_epoch = None
    start_epoch_s1 = 1
    start_epoch_s2 = 1
    restored_optimizer_state = None

    if args.resume_checkpoint is not None:
        resume_path = resolve_path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        logger.info("Loading resume checkpoint from %s...", resume_path)
        loaded_model, ckpt_meta = ViTClassifier.load_checkpoint(resume_path, device=device)

        # Validate compatibility
        ckpt_classes = ckpt_meta.get("num_classes", loaded_model.num_classes)
        if ckpt_classes != model.num_classes:
            raise ValueError(
                f"Checkpoint num_classes ({ckpt_classes}) is incompatible with model num_classes ({model.num_classes})"
            )

        resume_stage = ckpt_meta.get("stage")
        resume_completed_epoch = ckpt_meta.get("epoch")
        if resume_stage is None or resume_completed_epoch is None:
            raise ValueError(
                f"Checkpoint at {resume_path} is missing required 'stage' or 'epoch' metadata for resume."
            )

        # Load weights into model
        model.load_state_dict(loaded_model.state_dict())
        restored_optimizer_state = ckpt_meta.get("optimizer_state_dict")

        # Restore best validation metrics if present
        if "validation_metrics" in ckpt_meta and "macro_f1" in ckpt_meta["validation_metrics"]:
            best_val_f1 = float(ckpt_meta["validation_metrics"]["macro_f1"])
            best_val_metrics = ckpt_meta["validation_metrics"]
            logger.info(
                "Restored baseline best validation Macro-F1 from checkpoint: %.4f (Stage %d, Epoch %d)",
                best_val_f1, resume_stage, resume_completed_epoch
            )

        if resume_stage == 1:
            start_epoch_s1 = resume_completed_epoch + 1
            start_epoch_s2 = 1
            logger.info(
                "Resuming from Stage 1 Epoch %d (will execute Stage 1 Epochs %d through %d)",
                resume_completed_epoch, start_epoch_s1, args.stage1_epochs
            )
        elif resume_stage == 2:
            start_epoch_s1 = args.stage1_epochs + 1  # Skip Stage 1
            start_epoch_s2 = resume_completed_epoch + 1
            logger.info(
                "Resuming from Stage 2 Epoch %d (Stage 1 completed; will execute Stage 2 Epochs %d through %d)",
                resume_completed_epoch, start_epoch_s2, args.stage2_epochs
            )
        else:
            raise ValueError(f"Unrecognized stage '{resume_stage}' in checkpoint metadata.")

    # === STAGE 1: Frozen Backbone Training ===
    model.freeze_backbone()
    s1_counts = model.get_parameter_counts()
    logger.info("Stage 1 Trainable Parameters: %d (Linear Head only)", s1_counts["trainable"])

    optimizer_s1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.stage1_lr,
        weight_decay=0.01,
    )

    if resume_stage == 1 and restored_optimizer_state is not None:
        try:
            optimizer_s1.load_state_dict(restored_optimizer_state)
            logger.info("Successfully restored Stage 1 AdamW optimizer state from checkpoint.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to restore Stage 1 optimizer state from checkpoint: {e}. "
                "The optimizer state is incompatible with current trainable parameters."
            ) from e

    if start_epoch_s1 <= args.stage1_epochs:
        logger.info(
            "=== Starting Stage 1: Frozen Backbone Training (%d epochs, LR=%.1e, starting from epoch %d) ===",
            args.stage1_epochs, args.stage1_lr, start_epoch_s1
        )
        for epoch in range(start_epoch_s1, args.stage1_epochs + 1):
            ep_t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, train_criterion, optimizer_s1, device, epoch, "Stage 1", scaler=scaler
            )
            val_metrics = evaluate_split(
                model, val_loader, val_criterion, device, f"Stage 1 Val Epoch {epoch}"
            )
            ep_duration = time.time() - ep_t0

            logger.info(
                "Stage 1 Epoch %d/%d (%.1fs) | Train Loss=%.4f, Train Acc=%.2f%% | "
                "Val Loss=%.4f, Val Acc=%.2f%%, Val Bal Acc=%.2f%%, Val Macro-F1=%.4f",
                epoch, args.stage1_epochs, ep_duration, train_loss, train_acc * 100,
                val_metrics["loss"], val_metrics["accuracy"] * 100,
                val_metrics["balanced_accuracy"] * 100, val_metrics["macro_f1"]
            )

            # Per-class metrics logging
            for state_name in ["ALERT", "LOW_VIGILANCE", "DROWSY", "MICROSLEEP"]:
                c_m = val_metrics["classification_report"].get(state_name, {})
                logger.info(
                    "  [%s] Precision=%.4f, Recall=%.4f, F1=%.4f",
                    state_name, c_m.get("precision", 0), c_m.get("recall", 0), c_m.get("f1-score", 0)
                )

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_val_metrics = val_metrics
                checkpoint_metadata = {
                    "model_name": model.config.model_name,
                    "num_classes": 4,
                    "canonical_label_mapping": {st.value: st.name for st in ProjectState},
                    "stage": 1,
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer_s1.state_dict(),
                    "training_metrics": {"loss": train_loss, "accuracy": train_acc},
                    "validation_metrics": {
                        "loss": val_metrics["loss"],
                        "accuracy": val_metrics["accuracy"],
                        "balanced_accuracy": val_metrics["balanced_accuracy"],
                        "macro_precision": val_metrics["macro_precision"],
                        "macro_recall": val_metrics["macro_recall"],
                        "macro_f1": val_metrics["macro_f1"],
                        "weighted_f1": val_metrics["weighted_f1"],
                        "classification_report": val_metrics["classification_report"],
                        "confusion_matrix": val_metrics["confusion_matrix"],
                    },
                    "selected_model_criterion": "validation_macro_f1",
                    "random_seed": args.seed,
                    "dataset_manifest": to_project_relative_path(train_manifest),
                }
                model.save_checkpoint(best_checkpoint_path, metadata=checkpoint_metadata)
                logger.info("Saved new best model checkpoint (Val Macro-F1: %.4f) to %s", best_val_f1, best_checkpoint_path)
    else:
        logger.info(
            "Stage 1 already complete (%d epochs completed >= configured %d epochs). Skipping Stage 1.",
            resume_completed_epoch if resume_stage == 1 else args.stage1_epochs, args.stage1_epochs
        )

    # === STAGE 2: Selective Fine-Tuning ===
    logger.info("=== Preparing Stage 2: Selective Fine-Tuning (LR=%.1e, unfreeze %d blocks) ===",
                args.stage2_lr, args.unfreeze_blocks)
    model.unfreeze_deep_layers(num_blocks=args.unfreeze_blocks)
    s2_counts = model.get_parameter_counts()
    logger.info("Stage 2 Trainable Parameters: %d", s2_counts["trainable"])

    optimizer_s2 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.stage2_lr,
        weight_decay=0.01,
    )

    if resume_stage == 2 and restored_optimizer_state is not None:
        try:
            optimizer_s2.load_state_dict(restored_optimizer_state)
            logger.info("Successfully restored Stage 2 AdamW optimizer state from checkpoint.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to restore Stage 2 optimizer state from checkpoint: {e}. "
                "The optimizer state is incompatible with current trainable parameters."
            ) from e

    if start_epoch_s2 <= args.stage2_epochs:
        logger.info(
            "=== Running Stage 2: Selective Fine-Tuning (%d epochs, LR=%.1e, starting from epoch %d) ===",
            args.stage2_epochs, args.stage2_lr, start_epoch_s2
        )
        for epoch in range(start_epoch_s2, args.stage2_epochs + 1):
            ep_t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, train_criterion, optimizer_s2, device, epoch, "Stage 2", scaler=scaler
            )
            val_metrics = evaluate_split(
                model, val_loader, val_criterion, device, f"Stage 2 Val Epoch {epoch}"
            )
            ep_duration = time.time() - ep_t0

            logger.info(
                "Stage 2 Epoch %d/%d (%.1fs) | Train Loss=%.4f, Train Acc=%.2f%% | "
                "Val Loss=%.4f, Val Acc=%.2f%%, Val Bal Acc=%.2f%%, Val Macro-F1=%.4f",
                epoch, args.stage2_epochs, ep_duration, train_loss, train_acc * 100,
                val_metrics["loss"], val_metrics["accuracy"] * 100,
                val_metrics["balanced_accuracy"] * 100, val_metrics["macro_f1"]
            )

            for state_name in ["ALERT", "LOW_VIGILANCE", "DROWSY", "MICROSLEEP"]:
                c_m = val_metrics["classification_report"].get(state_name, {})
                logger.info(
                    "  [%s] Precision=%.4f, Recall=%.4f, F1=%.4f",
                    state_name, c_m.get("precision", 0), c_m.get("recall", 0), c_m.get("f1-score", 0)
                )

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_val_metrics = val_metrics
                checkpoint_metadata = {
                    "model_name": model.config.model_name,
                    "num_classes": 4,
                    "canonical_label_mapping": {st.value: st.name for st in ProjectState},
                    "stage": 2,
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer_s2.state_dict(),
                    "training_metrics": {"loss": train_loss, "accuracy": train_acc},
                    "validation_metrics": {
                        "loss": val_metrics["loss"],
                        "accuracy": val_metrics["accuracy"],
                        "balanced_accuracy": val_metrics["balanced_accuracy"],
                        "macro_precision": val_metrics["macro_precision"],
                        "macro_recall": val_metrics["macro_recall"],
                        "macro_f1": val_metrics["macro_f1"],
                        "weighted_f1": val_metrics["weighted_f1"],
                        "classification_report": val_metrics["classification_report"],
                        "confusion_matrix": val_metrics["confusion_matrix"],
                    },
                    "selected_model_criterion": "validation_macro_f1",
                    "random_seed": args.seed,
                    "dataset_manifest": to_project_relative_path(train_manifest),
                }
                model.save_checkpoint(best_checkpoint_path, metadata=checkpoint_metadata)
                logger.info("Saved new best model checkpoint (Val Macro-F1: %.4f) to %s", best_val_f1, best_checkpoint_path)
    else:
        logger.info(
            "Stage 2 already complete (%d epochs completed >= configured %d epochs).",
            resume_completed_epoch if resume_stage == 2 else args.stage2_epochs, args.stage2_epochs
        )

    total_train_duration = time.time() - training_start_time
    logger.info("Total Production Training Duration: %.1f seconds (%.2f minutes)", total_train_duration, total_train_duration / 60)

    # === Final Evaluation on Held-Out Test Split (Executed ONCE) ===
    logger.info("=== Loading Best Checkpoint for Single Test Set Evaluation ===")
    best_model, meta = ViTClassifier.load_checkpoint(best_checkpoint_path, device=device)
    logger.info("Loaded checkpoint from %s (Selected Stage: %s, Epoch: %s, Best Val F1: %.4f)",
                best_checkpoint_path, meta.get("stage"), meta.get("epoch"), meta.get("validation_metrics", {}).get("macro_f1", 0.0))

    test_metrics = evaluate_split(
        best_model, test_loader, val_criterion, device, "Development Test Split Evaluation"
    )

    logger.info("============================================================")
    logger.info("=== FINAL HELD-OUT DEVELOPMENT TEST SET EVALUATION ===")
    logger.info("============================================================")
    logger.info("Test Loss:              %.4f", test_metrics["loss"])
    logger.info("Test Accuracy:          %.4f (%.2f%%)", test_metrics["accuracy"], test_metrics["accuracy"] * 100)
    logger.info("Test Balanced Accuracy: %.4f (%.2f%%)", test_metrics["balanced_accuracy"], test_metrics["balanced_accuracy"] * 100)
    logger.info("Test Macro Precision:   %.4f", test_metrics["macro_precision"])
    logger.info("Test Macro Recall:      %.4f", test_metrics["macro_recall"])
    logger.info("Test Macro F1:          %.4f", test_metrics["macro_f1"])
    logger.info("Test Weighted F1:       %.4f", test_metrics["weighted_f1"])
    logger.info("Confusion Matrix (Labels: 0=ALERT, 1=LOW_VIGILANCE, 2=DROWSY, 3=MICROSLEEP):")
    logger.info("\n%s", np.array(test_metrics["confusion_matrix"]))

    logger.info("Detailed Classification Report:")
    for state_name in ["ALERT", "LOW_VIGILANCE", "DROWSY", "MICROSLEEP"]:
        c_m = test_metrics["classification_report"].get(state_name, {})
        logger.info(
            "  %-15s | Precision: %.4f | Recall: %.4f | F1: %.4f | Support: %d",
            state_name, c_m.get("precision", 0), c_m.get("recall", 0), c_m.get("f1-score", 0), int(c_m.get("support", 0))
        )

    logger.warning(
        "CRITICAL METHODOLOGY NOTE: The development test split evaluates on "
        "video sequences unseen during training (zero temporal/video leakage), "
        "but is NOT subject-independent because subjects 001, 002, and 005 "
        "appear across splits. Subject-generalization to unseen drivers "
        "must be evaluated via the LOSO cross-validation folds."
    )

    # === Inference Latency & Throughput Benchmarking ===
    benchmark_results = run_benchmarks(best_model, device=device)

    # Save complete evaluation artifact
    eval_artifact = {
        "production_checkpoint": to_project_relative_path(best_checkpoint_path),
        "total_training_duration_seconds": total_train_duration,
        "best_validation_metrics": best_val_metrics,
        "final_test_metrics": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "per_class": {
                st: test_metrics["classification_report"].get(st, {})
                for st in ["ALERT", "LOW_VIGILANCE", "DROWSY", "MICROSLEEP"]
            },
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "inference_benchmarks": benchmark_results,
        "parameter_counts": {
            "total": param_counts["total"],
            "stage1_trainable": s1_counts["trainable"],
            "stage2_trainable": s2_counts["trainable"],
        },
    }

    eval_output_path = args.splits_dir.parent / "production_evaluation_summary.json"
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_artifact, f, indent=2)
    logger.info("Saved complete production evaluation summary to %s", eval_output_path)


if __name__ == "__main__":
    main()
