#!/usr/bin/env python3
"""NTHU Dataset Preprocessing, Splitting, and Verification CLI.

Parses raw NTHU images, extracts metadata records, generates both:
1. Primary Development Split: Stratified Video-Sequence Grouping (~70% Train,
   ~15% Val, ~15% Test) guaranteeing zero temporal/video-sequence leakage.
2. Subject-Independent Evaluation: Leave-One-Subject-Out (LOSO) 4-fold cross-validation
   measuring generalization to unseen drivers, explicitly handling incomplete class
   coverage for Subject 006 without data fabrication.

Also provides Face ROI cropping with MediaPipe FaceMesh using a configurable margin.
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
from typing import Any, Dict, List, Optional, Tuple
import cv2
from tqdm import tqdm

from src.dataset import (
    CanonicalLabelMapper,
    SampleMetadata,
    crop_face_roi,
    generate_loso_splits,
    generate_stratified_sequence_splits,
    load_manifest_csv,
    resolve_path,
    save_manifest_csv,
    to_project_relative_path,
    validate_dataset_directory,
    verify_dataset_manifest,
)
from src.landmark_detector import FaceMeshDetector
from utils.config import DEFAULT_CONFIG, FACE_ROI_MARGIN, AppConfig, ProjectState
from utils.logger import setup_logger

logger = setup_logger("preprocess_nthu")


def parse_raw_dataset(
    raw_dir: Path,
    mapper: CanonicalLabelMapper,
) -> Tuple[List[SampleMetadata], Dict[str, int]]:
    """Scan raw directory, parse filenames, and construct SampleMetadata records.

    Expected directory structure:
        raw_dir/train/
            notdrowsy/
            drowsy/
                yawning/
                slowBlinkWithNodding/
                sleepyCombination/

    Filename format:
        {subject_id}_{condition}_{scenario}_{frame_index}_{raw_label}.jpg
    """
    train_dir = raw_dir / "train"
    if not train_dir.exists():
        logger.error(
            "NTHU dataset not found. Dataset preprocessing/training cannot be executed "
            "until the dataset is downloaded and configured."
        )
        return [], {}

    samples: List[SampleMetadata] = []
    error_counts: Dict[str, int] = {"missing_parts": 0, "unmapped_label": 0}

    # Find all JPG files
    image_paths = sorted(list(train_dir.glob("*/*.jpg")) + list(train_dir.glob("*/*/*.jpg")))
    logger.info("Found %d image files in %s", len(image_paths), train_dir)

    for img_path in image_paths:
        stem = img_path.stem
        parts = stem.split("_")
        if len(parts) < 5:
            error_counts["missing_parts"] += 1
            continue

        subject_id = parts[0]
        condition = parts[1]
        scenario = parts[2]
        try:
            frame_index = int(parts[3])
        except ValueError:
            error_counts["missing_parts"] += 1
            continue

        raw_label = parts[4]
        # The immediate parent directory represents the scenario/class category in this dataset
        dir_label = img_path.parent.name
        try:
            canonical_state = mapper.map_label(dir_label)
        except ValueError:
            try:
                canonical_state = mapper.map_label(raw_label)
            except ValueError:
                error_counts["unmapped_label"] += 1
                continue

        video_seq_id = f"{subject_id}_{condition}_{scenario}"

        meta = SampleMetadata(
            sample_id=img_path.name,
            subject_id=subject_id,
            condition=condition,
            scenario=scenario,
            frame_index=frame_index,
            video_sequence_id=video_seq_id,
            raw_label=raw_label,
            canonical_state=canonical_state,
            original_path=to_project_relative_path(img_path),
            processed_path="",
            split="unassigned",
            original_filename=img_path.name,
        )
        samples.append(meta)

    logger.info(
        "Successfully parsed %d samples. Errors: %s",
        len(samples), error_counts
    )
    return samples, error_counts


def run_manifest_generation(
    raw_dir: Path,
    splits_dir: Path,
    seed: int = 42,
) -> bool:
    """Generate both Development Split and LOSO Split manifests.

    Args:
        raw_dir: Path to raw dataset root.
        splits_dir: Path to output manifests directory.
        seed: Random seed for reproducibility.

    Returns:
        True if successful, False otherwise.
    """
    if not validate_dataset_directory(raw_dir):
        return False

    mapper = CanonicalLabelMapper()
    samples, errors = parse_raw_dataset(raw_dir, mapper)
    if not samples:
        logger.error(
            "NTHU dataset not found. Dataset preprocessing/training cannot be executed "
            "until the dataset is downloaded and configured."
        )
        return False

    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Primary Development Split (Stratified Video-Sequence Grouping)
    logger.info("Generating Primary Development Split (Stratified Video-Sequence Grouping)...")
    train_s, val_s, test_s = generate_stratified_sequence_splits(samples, random_seed=seed)

    save_manifest_csv(samples, splits_dir / "all_manifest.csv")
    save_manifest_csv(train_s, splits_dir / "dev_train_manifest.csv")
    save_manifest_csv(val_s, splits_dir / "dev_val_manifest.csv")
    save_manifest_csv(test_s, splits_dir / "dev_test_manifest.csv")

    dev_audit = verify_dataset_manifest(samples, split_type="development")
    logger.info("=== Development Split Audit Summary ===")
    logger.info("Train: %d frames (%.2f%%)", len(train_s), dev_audit["split_percentages"].get("train", 0))
    logger.info("Val:   %d frames (%.2f%%)", len(val_s), dev_audit["split_percentages"].get("val", 0))
    logger.info("Test:  %d frames (%.2f%%)", len(test_s), dev_audit["split_percentages"].get("test", 0))
    logger.info("Sequence Leakage Detected: %s (Count=%d)", dev_audit["sequence_leakage_detected"], dev_audit["sequence_leakage_count"])
    logger.info("Subject Overlap Detected:  %s (Subjects: %s)", dev_audit["subject_overlap_detected"], dev_audit["overlapping_subjects"])
    logger.info("Methodology Note: %s", dev_audit["methodology_note"])

    # 2. Generate Subject-Independent LOSO Folds
    logger.info("Generating Subject-Independent Leave-One-Subject-Out (LOSO) Folds...")
    loso_folds = generate_loso_splits(samples)

    for fold_name, fold_info in loso_folds.items():
        save_manifest_csv(fold_info.train_samples, splits_dir / f"{fold_name}_train.csv")
        save_manifest_csv(fold_info.eval_samples, splits_dir / f"{fold_name}_eval.csv")

        logger.info(
            "LOSO Fold '%s': Held-out Subject=%s, Train=%d, Eval=%d, Complete Coverage=%s, Absent Classes=%s",
            fold_name, fold_info.held_out_subject,
            len(fold_info.train_samples), len(fold_info.eval_samples),
            fold_info.is_complete_coverage, fold_info.absent_classes
        )

    logger.info("All manifest files successfully written to %s", splits_dir)
    return True


_worker_detector: Optional[FaceMeshDetector] = None


def _init_worker() -> None:
    """Initialize per-process MediaPipe FaceMeshDetector."""
    global _worker_detector
    _worker_detector = FaceMeshDetector(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False
    )


def _process_single_image(
    task_args: Tuple[Dict[str, Any], str, float, Tuple[int, int]]
) -> Tuple[str, Optional[str], bool, bool, str]:
    """Worker task to process a single image with MediaPipe FaceMesh.

    Args:
        task_args: (sample_dict, output_dir_str, margin, target_size)

    Returns:
        Tuple of (sample_id, processed_path_str, face_detected, is_corrupt, reason)
    """
    global _worker_detector
    sample_dict, output_dir_str, margin, target_size = task_args
    sample_id = sample_dict["sample_id"]
    original_path = sample_dict["original_path"]
    canonical_state_name = sample_dict["canonical_state_name"]
    original_filename = sample_dict["original_filename"]

    src_path = resolve_path(original_path)
    if not src_path.exists():
        return (sample_id, None, False, True, f"File not found: {original_path}")

    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        return (sample_id, None, False, True, "Corrupt or unreadable image file")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    assert _worker_detector is not None, "Worker detector not initialized"
    mesh_res = _worker_detector.detect(img_rgb)

    output_dir = Path(output_dir_str)
    dst_path = output_dir / canonical_state_name / original_filename
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    face_detected = False
    reason = ""

    if mesh_res is not None and len(mesh_res.pixel_landmarks) > 0:
        pil_crop, _ = crop_face_roi(
            img_rgb,
            landmarks=mesh_res.pixel_landmarks,
            margin=margin,
            target_size=target_size,
        )
        face_detected = True
    else:
        # Fallback to center crop so sample is preserved for auditing
        pil_crop, _ = crop_face_roi(
            img_rgb,
            landmarks=None,
            margin=margin,
            target_size=target_size,
        )
        face_detected = False
        reason = "No facial landmarks detected by MediaPipe; fallback center crop applied"

    # Save high-quality unnormalized JPEG (normalization happens dynamically in PyTorch transform)
    pil_crop.save(dst_path, format="JPEG", quality=95)
    return (sample_id, to_project_relative_path(dst_path), face_detected, False, reason)


def run_face_roi_extraction(
    manifest_path: Path,
    splits_dir: Path,
    output_dir: Path,
    margin: float = FACE_ROI_MARGIN,
    target_size: Tuple[int, int] = (224, 224),
    limit: Optional[int] = None,
    num_workers: int = 12,
) -> Dict[str, Any]:
    """Extract face ROIs from images listed in manifest using MediaPipe FaceMesh.

    Supports multiprocessing across CPU cores, audits face detection failures,
    and updates all split manifests with processed image paths.

    Args:
        manifest_path: Path to all_manifest.csv.
        splits_dir: Path to directory containing all CSV split manifests.
        output_dir: Output root directory for cropped images.
        margin: Face bounding box margin fraction (default: 0.20).
        target_size: Target square crop size (224, 224).
        limit: Optional maximum number of images to crop.
        num_workers: Number of parallel worker processes.

    Returns:
        Structured audit report dictionary.
    """
    import multiprocessing as mp
    import time

    all_samples = load_manifest_csv(manifest_path)
    samples = all_samples[:limit] if limit else all_samples

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting Face ROI extraction for %d images (workers=%d, margin=%.2f, target=%s) into %s",
        len(samples), num_workers, margin, target_size, output_dir
    )

    task_args = [
        (s.to_dict(), str(output_dir), margin, target_size)
        for s in samples
    ]

    t0 = time.time()
    processed_map: Dict[str, str] = {}
    failed_detections: List[Dict[str, str]] = []
    corrupt_images: List[Dict[str, str]] = []
    success_count = 0
    fallback_count = 0

    if num_workers > 1:
        with mp.Pool(processes=num_workers, initializer=_init_worker) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_single_image, task_args, chunksize=50),
                total=len(task_args),
                desc="Extracting Face ROIs (Multi-process)",
            ):
                sample_id, proc_path, face_detected, is_corrupt, reason = result
                if is_corrupt:
                    corrupt_images.append({"sample_id": sample_id, "reason": reason})
                elif face_detected:
                    success_count += 1
                    if proc_path:
                        processed_map[sample_id] = proc_path
                else:
                    fallback_count += 1
                    failed_detections.append({
                        "sample_id": sample_id,
                        "processed_path": proc_path or "",
                        "reason": reason,
                    })
                    if proc_path:
                        processed_map[sample_id] = proc_path
    else:
        _init_worker()
        for task in tqdm(task_args, desc="Extracting Face ROIs (Serial)"):
            sample_id, proc_path, face_detected, is_corrupt, reason = _process_single_image(task)
            if is_corrupt:
                corrupt_images.append({"sample_id": sample_id, "reason": reason})
            elif face_detected:
                success_count += 1
                if proc_path:
                    processed_map[sample_id] = proc_path
            else:
                fallback_count += 1
                failed_detections.append({
                    "sample_id": sample_id,
                    "processed_path": proc_path or "",
                    "reason": reason,
                })
                if proc_path:
                    processed_map[sample_id] = proc_path

    elapsed_time = time.time() - t0
    fps = len(samples) / elapsed_time if elapsed_time > 0 else 0.0

    # Write failure audit log if any detection failures occurred
    failures_csv_path = splits_dir / "face_detection_failures.csv"
    if failed_detections:
        with open(failures_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "processed_path", "reason"])
            writer.writeheader()
            writer.writerows(failed_detections)
        logger.warning(
            "Saved %d face detection failures to %s for auditing",
            len(failed_detections), failures_csv_path
        )
    else:
        logger.info("Zero face detection failures! 100.0%% detection rate achieved.")

    # Update processed_path across all samples in memory
    for s in all_samples:
        if s.sample_id in processed_map:
            s.processed_path = processed_map[s.sample_id]

    # Save updated master manifest
    save_manifest_csv(all_samples, manifest_path)

    # Update all other split manifest CSVs in splits_dir
    split_manifest_names = [
        "dev_train_manifest.csv",
        "dev_val_manifest.csv",
        "dev_test_manifest.csv",
        "loso_subject_001_train.csv",
        "loso_subject_001_eval.csv",
        "loso_subject_002_train.csv",
        "loso_subject_002_eval.csv",
        "loso_subject_005_train.csv",
        "loso_subject_005_eval.csv",
        "loso_subject_006_train.csv",
        "loso_subject_006_eval.csv",
    ]

    for sm_name in split_manifest_names:
        sm_path = splits_dir / sm_name
        if sm_path.exists():
            manifest_samples = load_manifest_csv(sm_path)
            for ms in manifest_samples:
                if ms.sample_id in processed_map:
                    ms.processed_path = processed_map[ms.sample_id]
            save_manifest_csv(manifest_samples, sm_path)
            logger.info("Updated processed_path in %s", sm_name)

    # Compute breakdown per split and per class
    split_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}

    for s in all_samples:
        if s.sample_id in processed_map:
            split_counts[s.split] = split_counts.get(s.split, 0) + 1
            class_counts[s.canonical_state.name] = class_counts.get(s.canonical_state.name, 0) + 1

    stats = {
        "total_attempted": len(samples),
        "successful_detections": success_count,
        "failed_detections": fallback_count,
        "detection_rate_pct": (success_count / len(samples) * 100) if samples else 0.0,
        "corrupt_images": len(corrupt_images),
        "elapsed_seconds": elapsed_time,
        "average_fps": fps,
        "images_generated_per_split": split_counts,
        "images_generated_per_class": class_counts,
    }

    logger.info("=== Face ROI Preprocessing Summary ===")
    logger.info("Total images attempted:    %d", stats["total_attempted"])
    logger.info("Successful detections:     %d (%.2f%%)", success_count, stats["detection_rate_pct"])
    logger.info("Failed detections:         %d", fallback_count)
    logger.info("Corrupt images:            %d", stats["corrupt_images"])
    logger.info("Processing time:           %.2f seconds (%.1f FPS)", elapsed_time, fps)
    logger.info("Images per split:          %s", split_counts)
    logger.info("Images per class:          %s", class_counts)

    return stats


def main() -> None:
    """CLI entry point for NTHU dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description="NTHU Drowsy Driver Dataset Preprocessing & Manifest Generator"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_CONFIG.raw_data_dir,
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_CONFIG.splits_dir,
        help="Path to output splits directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CONFIG.processed_data_dir,
        help="Path to output processed crops directory",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=FACE_ROI_MARGIN,
        help=f"Face bounding box margin fraction (default: {FACE_ROI_MARGIN})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.dataset.random_seed,
        help="Random seed for splitting",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Generate only manifest CSV files without extracting image crops",
    )
    parser.add_argument(
        "--extract-rois",
        action="store_true",
        help="Extract MediaPipe FaceMesh crops to disk",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of parallel worker processes for ROI extraction (default: 12)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing manifest CSV files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to crop (for testing)",
    )

    args = parser.parse_args()

    # If raw dataset directory is missing
    if not validate_dataset_directory(args.raw_dir):
        sys.exit(1)

    if args.verify:
        all_manifest = args.splits_dir / "all_manifest.csv"
        if not all_manifest.exists():
            logger.error("Manifest not found: %s. Run --manifest-only first.", all_manifest)
            sys.exit(1)
        samples = load_manifest_csv(all_manifest)
        audit = verify_dataset_manifest(samples, split_type="development")
        logger.info("Verification passed: %s", audit)
        return

    # Generate manifests if manifest-only or not existing
    all_manifest = args.splits_dir / "all_manifest.csv"
    if args.manifest_only or not all_manifest.exists():
        success = run_manifest_generation(args.raw_dir, args.splits_dir, seed=args.seed)
        if not success:
            sys.exit(1)

    # Extract Face ROIs if requested
    if args.extract_rois:
        run_face_roi_extraction(
            manifest_path=all_manifest,
            splits_dir=args.splits_dir,
            output_dir=args.output_dir,
            margin=args.margin,
            limit=args.limit,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()

