"""Phase 3 Runtime Performance Benchmark.

Measures the actual integrated Phase 3 end-to-end pipeline:
Camera Frame -> MediaPipe -> Face ROI -> ViT -> FeatureFusion -> TemporalDecisionLayer

Outputs high-precision per-frame metrics and component latencies to CSV.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import crop_face_roi
from src.detector import DetectionResult, DrowsinessDetector
from utils.config import DEFAULT_CONFIG, AppConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("benchmark_phase3")


class BenchmarkDetector(DrowsinessDetector):
    """Subclass of DrowsinessDetector that instruments per-component execution timings."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        enable_audio: bool = False,
        enable_vit: bool = True,
        vit_engine: Optional[object] = None,
    ) -> None:
        super().__init__(
            config=config,
            enable_audio=enable_audio,
            enable_vit=enable_vit,
            vit_engine=vit_engine,
        )

    def process_frame_instrumented(
        self, frame: Optional[np.ndarray]
    ) -> Tuple[DetectionResult, Dict[str, float]]:
        """Process a single frame with high-resolution per-component latency measurement.

        Args:
            frame: Input BGR video frame.

        Returns:
            Tuple of (DetectionResult, timings_dict in milliseconds).
        """
        timings: Dict[str, float] = {
            "landmark_ms": 0.0,
            "roi_ms": 0.0,
            "vit_ms": 0.0,
            "fusion_ms": 0.0,
            "temporal_ms": 0.0,
            "total_latency_ms": 0.0,
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total_start = time.perf_counter()

        if frame is None or frame.size == 0:
            temporal_dec = self.temporal_decision_layer.update(None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_total_end = time.perf_counter()
            timings["total_latency_ms"] = (t_total_end - t_total_start) * 1000.0
            return (
                DetectionResult(
                    state=temporal_dec.current_state,
                    face_detected=False,
                    reasons=["Invalid or empty video frame"],
                    temporal_decision=temporal_dec,
                ),
                timings,
            )

        frame_shape = frame.shape[:2]

        # 1. MediaPipe FaceMesh detection
        t0 = time.perf_counter()
        face_landmarks = self.landmark_detector.detect(frame)
        t1 = time.perf_counter()
        timings["landmark_ms"] = (t1 - t0) * 1000.0

        if face_landmarks is None:
            # Face not detected: safe fallback and freeze state
            baseline_res = self.baseline_detector.update(None)
            t_temp0 = time.perf_counter()
            temporal_dec = self.temporal_decision_layer.update(None)
            t_temp1 = time.perf_counter()
            timings["temporal_ms"] = (t_temp1 - t_temp0) * 1000.0

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_total_end = time.perf_counter()
            timings["total_latency_ms"] = (t_total_end - t_total_start) * 1000.0

            active_reasons = (
                temporal_dec.transition_reason
                if temporal_dec.transition_reason
                else ["No face detected"]
            )
            return (
                DetectionResult(
                    state=temporal_dec.current_state,
                    perclos=baseline_res.perclos,
                    face_detected=False,
                    reasons=[active_reasons] if isinstance(active_reasons, str) else active_reasons,
                    temporal_decision=temporal_dec,
                ),
                timings,
            )

        # 2. 3D Head Pose Estimation & Classical Baseline
        pose_res = self.head_pose_estimator.estimate_pose(
            face_landmarks.pixel_landmarks, frame_shape=frame_shape
        )
        pitch = pose_res.pitch if pose_res.success else 0.0
        yaw = pose_res.yaw if pose_res.success else 0.0

        baseline_res = self.baseline_detector.update(
            face_landmarks.pixel_landmarks, pitch=pitch, yaw=yaw
        )

        # 3. Extract Face ROI for ViT
        t0 = time.perf_counter()
        pil_crop, _ = crop_face_roi(
            frame,
            landmarks=face_landmarks.pixel_landmarks,
            bbox=face_landmarks.bbox,
            margin=self.config.dataset.face_margin,
            target_size=self.config.vit.image_size,
        )
        t1 = time.perf_counter()
        timings["roi_ms"] = (t1 - t0) * 1000.0

        # 4. ViT Visual Inference
        if self.vit_engine is not None and self.vit_engine.is_ready:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            vit_probs = self.vit_engine.infer(pil_crop)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings["vit_ms"] = (t1 - t0) * 1000.0
        else:
            vit_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
            timings["vit_ms"] = 0.0

        # 5. Multimodal Feature Fusion
        t0 = time.perf_counter()
        fusion_res = self.feature_fusion.fuse(
            vit_probabilities=vit_probs,
            ear=baseline_res.avg_ear,
            mar=baseline_res.mar,
            perclos=baseline_res.perclos,
            pitch=pitch,
            yaw=yaw,
            head_pose=pose_res,
        )
        t1 = time.perf_counter()
        timings["fusion_ms"] = (t1 - t0) * 1000.0

        # 6. Temporal Decision Layer
        t0 = time.perf_counter()
        temporal_dec = self.temporal_decision_layer.update(fusion_res)
        t1 = time.perf_counter()
        timings["temporal_ms"] = (t1 - t0) * 1000.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total_end = time.perf_counter()
        timings["total_latency_ms"] = (t_total_end - t_total_start) * 1000.0

        final_state = temporal_dec.current_state

        active_reasons = []
        if temporal_dec.transition_reason:
            active_reasons.append(temporal_dec.transition_reason)
        active_reasons.extend(baseline_res.reasons)

        result = DetectionResult(
            state=final_state,
            left_ear=baseline_res.left_ear,
            right_ear=baseline_res.right_ear,
            avg_ear=baseline_res.avg_ear,
            mar=baseline_res.mar,
            perclos=baseline_res.perclos,
            eyes_closed=baseline_res.eyes_closed,
            continuous_closed_frames=temporal_dec.continuous_closed_frames,
            drowsy_persistence_counter=temporal_dec.drowsy_persistence_counter,
            alert_reset_counter=temporal_dec.alert_reset_counter,
            is_yawning=baseline_res.is_yawning,
            is_nodding=baseline_res.is_nodding,
            is_distracted=baseline_res.is_distracted,
            reasons=active_reasons,
            face_detected=True,
            pixel_landmarks=face_landmarks.pixel_landmarks,
            normalized_landmarks=face_landmarks.normalized_landmarks,
            bbox=face_landmarks.bbox,
            head_pose=pose_res,
            fusion_result=fusion_res,
            temporal_decision=temporal_dec,
            vit_probabilities=vit_probs,
        )
        return result, timings


def run_benchmark(
    source: str = "0",
    target_frames: int = 500,
    warmup_frames: int = 10,
    output_csv: Path = PROJECT_ROOT / "data" / "performance" / "phase3_benchmark.csv",
    enable_audio: bool = False,
) -> None:
    """Execute Phase 3 benchmark on video source and record metrics.

    Args:
        source: Camera index (e.g. '0') or video file path.
        target_frames: Number of post-warmup frames to record.
        warmup_frames: Number of initial frames to process before recording.
        output_csv: Target CSV file path.
        enable_audio: Whether audio alerts are active during benchmark.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing BenchmarkDetector (Phase 3 integrated pipeline)...")
    detector = BenchmarkDetector(config=DEFAULT_CONFIG, enable_audio=enable_audio)

    # Determine input source
    if source.isdigit():
        video_src = int(source)
        is_camera = True
    else:
        video_src = str(Path(source).resolve())
        is_camera = False

    logger.info("Opening video source: %s", source)
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        logger.error("Could not open video source '%s'. Check device index or file path.", source)
        detector.close()
        sys.exit(1)

    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warmup phase
    if warmup_frames > 0:
        logger.info("Warming up pipeline for %d frames...", warmup_frames)
        w_count = 0
        while w_count < warmup_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_camera:
                    time.sleep(0.01)
                    continue
                else:
                    break
            detector.process_frame_instrumented(frame)
            w_count += 1
        logger.info("Warmup complete.")

    records: List[Dict] = []
    frame_idx = 0
    logger.info("Starting performance benchmark for %d frames...", target_frames)

    t_bench_start = time.perf_counter()

    try:
        while frame_idx < target_frames:
            t_loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_camera:
                    logger.warning("Camera dropped a frame; retrying...")
                    time.sleep(0.01)
                    continue
                else:
                    logger.info("Reached end of video stream.")
                    break

            frame_idx += 1

            # Process frame with component instrumentation
            result, timings = detector.process_frame_instrumented(frame)

            t_loop_end = time.perf_counter()
            loop_latency_ms = (t_loop_end - t_loop_start) * 1000.0

            proc_latency_ms = timings["total_latency_ms"]
            proc_fps = 1000.0 / proc_latency_ms if proc_latency_ms > 0 else 0.0
            capture_fps = 1000.0 / loop_latency_ms if loop_latency_ms > 0 else 0.0

            state_str = (
                result.state.name if hasattr(result.state, "name") else str(result.state)
            )

            record = {
                "frame_number": frame_idx,
                "total_latency_ms": round(proc_latency_ms, 4),
                "FPS": round(proc_fps, 2),
                "face_detected": bool(result.face_detected),
                "final_state": state_str,
                "landmark_ms": round(timings["landmark_ms"], 4),
                "roi_ms": round(timings["roi_ms"], 4),
                "vit_ms": round(timings["vit_ms"], 4),
                "fusion_ms": round(timings["fusion_ms"], 4),
                "temporal_ms": round(timings["temporal_ms"], 4),
                "capture_latency_ms": round(loop_latency_ms, 4),
                "capture_fps": round(capture_fps, 2),
            }
            records.append(record)

            if frame_idx % 50 == 0 or frame_idx == target_frames:
                logger.info(
                    "Progress: %d/%d frames | Latency: %.2f ms | Throughput: %.1f FPS | Face: %s",
                    frame_idx,
                    target_frames,
                    proc_latency_ms,
                    proc_fps,
                    result.face_detected,
                )

    finally:
        cap.release()
        detector.close()

    total_bench_duration = time.perf_counter() - t_bench_start
    logger.info("Benchmark capture finished. Total wall time: %.2f s", total_bench_duration)

    if not records:
        logger.error("No frames were recorded during benchmark.")
        return

    # Write records to CSV
    fieldnames = [
        "frame_number",
        "total_latency_ms",
        "FPS",
        "face_detected",
        "final_state",
        "landmark_ms",
        "roi_ms",
        "vit_ms",
        "fusion_ms",
        "temporal_ms",
        "capture_latency_ms",
        "capture_fps",
    ]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info("Benchmark data written to: %s", output_csv)

    # Compute summary statistics
    n_frames = len(records)
    latencies = np.array([r["total_latency_ms"] for r in records])
    fps_vals = np.array([r["FPS"] for r in records])
    face_detected_vals = np.array([r["face_detected"] for r in records])

    mean_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    max_latency = float(np.max(latencies))

    mean_fps = float(np.mean(fps_vals))
    median_fps = float(np.median(fps_vals))
    min_fps = float(np.min(fps_vals))

    pct_above_28 = float((np.sum(fps_vals >= 28.0) / n_frames) * 100.0)
    face_detection_rate = float((np.sum(face_detected_vals) / n_frames) * 100.0)

    # Component latencies across all frames
    mean_landmark = float(np.mean([r["landmark_ms"] for r in records]))
    mean_roi = float(np.mean([r["roi_ms"] for r in records]))
    mean_vit = float(np.mean([r["vit_ms"] for r in records]))
    mean_fusion = float(np.mean([r["fusion_ms"] for r in records]))
    mean_temporal = float(np.mean([r["temporal_ms"] for r in records]))

    # Component latencies specifically on face-detected frames (active pipeline)
    face_records = [r for r in records if r["face_detected"]]
    if face_records:
        active_mean_roi = float(np.mean([r["roi_ms"] for r in face_records]))
        active_mean_vit = float(np.mean([r["vit_ms"] for r in face_records]))
        active_mean_fusion = float(np.mean([r["fusion_ms"] for r in face_records]))
    else:
        active_mean_roi = 0.0
        active_mean_vit = 0.0
        active_mean_fusion = 0.0

    capture_latencies = np.array([r["capture_latency_ms"] for r in records])
    mean_capture_fps = float(np.mean([r["capture_fps"] for r in records]))

    print("\n" + "=" * 70)
    print("PHASE 3 RUNTIME PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Frames Processed:           {n_frames}")
    print(f"Face Detection Rate:              {face_detection_rate:.2f}% ({np.sum(face_detected_vals)}/{n_frames})")
    print("-" * 70)
    print("PROCESSING LATENCY (ms):")
    print(f"  Mean Latency:                   {mean_latency:.2f} ms")
    print(f"  Median Latency:                 {median_latency:.2f} ms")
    print(f"  95th Percentile (P95) Latency:  {p95_latency:.2f} ms")
    print(f"  Maximum Latency:                {max_latency:.2f} ms")
    print("-" * 70)
    print("PROCESSING THROUGHPUT (FPS):")
    print(f"  Mean FPS:                       {mean_fps:.2f} FPS")
    print(f"  Median FPS:                     {median_fps:.2f} FPS")
    print(f"  Minimum FPS:                    {min_fps:.2f} FPS")
    print(f"  Frames >= 28 FPS:               {pct_above_28:.2f}%")
    print("-" * 70)
    print("COMPONENT LATENCY BREAKDOWN (Mean):")
    print(f"  MediaPipe FaceMesh:             {mean_landmark:.2f} ms")
    print(f"  Face ROI Extraction:            {mean_roi:.2f} ms (active: {active_mean_roi:.2f} ms)")
    print(f"  ViT Visual Classification:      {mean_vit:.2f} ms (active: {active_mean_vit:.2f} ms)")
    print(f"  Multimodal Feature Fusion:      {mean_fusion:.2f} ms (active: {active_mean_fusion:.2f} ms)")
    print(f"  Temporal Decision Layer:        {mean_temporal:.2f} ms")
    print("-" * 70)
    print("HARDWARE WEBCAM SENSOR CAPTURE:")
    print(f"  Mean Physical Loop Latency:     {float(np.mean(capture_latencies)):.2f} ms")
    print(f"  Mean Physical Capture Rate:     {mean_capture_fps:.2f} FPS")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Runtime Performance Benchmark")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source index or path (default: '0' for primary webcam)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=500,
        help="Number of post-warmup frames to process (default: 500)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup frames (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "performance" / "phase3_benchmark.csv"),
        help="Destination CSV file path",
    )
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        help="Enable audio alert chimes during benchmark (default: disabled)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        source=args.source,
        target_frames=args.frames,
        warmup_frames=args.warmup,
        output_csv=Path(args.output),
        enable_audio=args.enable_audio,
    )
