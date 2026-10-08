"""Authoritative Physical Webcam Performance Validation for Phase 3 Pipeline.

Measures the actual integrated Phase 3 end-to-end pipeline directly on physical webcam (source 0):
Camera Frame -> MediaPipe FaceMesh -> Face ROI -> ViT -> FeatureFusion -> TemporalDecisionLayer

Instruments:
1. True Synchronized End-to-End Latency & FPS (with explicit torch.cuda.synchronize())
2. Application / HUD-style Loop Latency & FPS (exact replica of main.py loop)
3. Component-wise latency breakdown: MediaPipe, Face ROI, ViT, Feature Fusion, Temporal Decision, Overhead
4. Hardware and software environment metadata

Outputs:
- data/performance/phase3_webcam_benchmark.csv
- data/performance/phase3_webcam_benchmark_report.txt
- data/performance/webcam_fps_comparison.png
- data/performance/webcam_latency_over_frames.png
- data/performance/webcam_component_latency.png
"""

import argparse
import csv
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
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
logger = logging.getLogger("benchmark_webcam")


class WebcamBenchmarkDetector(DrowsinessDetector):
    """Subclass of DrowsinessDetector that measures synchronized per-component timings."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        enable_audio: bool = False,
    ) -> None:
        # Reuses exact production pipeline without modifications
        super().__init__(
            config=config,
            enable_audio=enable_audio,
            enable_vit=True,
            vit_engine=None,
        )

    def process_frame_synchronized(
        self, frame: Optional[np.ndarray]
    ) -> Tuple[DetectionResult, Dict[str, float]]:
        """Process a frame with strict CUDA synchronization and component isolation.

        Returns:
            Tuple of (DetectionResult, timings_dict in milliseconds).
        """
        timings: Dict[str, float] = {
            "mediapipe_ms": 0.0,
            "roi_ms": 0.0,
            "vit_ms": 0.0,
            "fusion_ms": 0.0,
            "temporal_ms": 0.0,
            "overhead_ms": 0.0,
            "sync_pipeline_latency_ms": 0.0,
        }

        # 0. Synchronize CUDA prior to starting pipeline timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_pipeline_start = time.perf_counter()

        if frame is None or frame.size == 0:
            temporal_dec = self.temporal_decision_layer.update(None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_pipeline_end = time.perf_counter()
            timings["sync_pipeline_latency_ms"] = (t_pipeline_end - t_pipeline_start) * 1000.0
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
        timings["mediapipe_ms"] = (t1 - t0) * 1000.0

        if face_landmarks is None:
            # Safe fallback when face is not detected
            baseline_res = self.baseline_detector.update(None)
            t_temp0 = time.perf_counter()
            temporal_dec = self.temporal_decision_layer.update(None)
            t_temp1 = time.perf_counter()
            timings["temporal_ms"] = (t_temp1 - t_temp0) * 1000.0

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_pipeline_end = time.perf_counter()
            total_ms = (t_pipeline_end - t_pipeline_start) * 1000.0
            timings["sync_pipeline_latency_ms"] = total_ms
            timings["overhead_ms"] = max(0.0, total_ms - (timings["mediapipe_ms"] + timings["temporal_ms"]))

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
        t_aux0 = time.perf_counter()
        pose_res = self.head_pose_estimator.estimate_pose(
            face_landmarks.pixel_landmarks, frame_shape=frame_shape
        )
        pitch = pose_res.pitch if pose_res.success else 0.0
        yaw = pose_res.yaw if pose_res.success else 0.0

        baseline_res = self.baseline_detector.update(
            face_landmarks.pixel_landmarks, pitch=pitch, yaw=yaw
        )
        t_aux1 = time.perf_counter()
        aux_ms = (t_aux1 - t_aux0) * 1000.0

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

        # 4. ViT Visual Inference with explicit CUDA synchronization
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

        # 7. Final Pipeline CUDA Synchronization & Latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_pipeline_end = time.perf_counter()
        total_ms = (t_pipeline_end - t_pipeline_start) * 1000.0
        timings["sync_pipeline_latency_ms"] = total_ms

        # Compute overhead (head pose solvePnP, classical baseline, dataclass packaging)
        comp_sum = (
            timings["mediapipe_ms"]
            + timings["roi_ms"]
            + timings["vit_ms"]
            + timings["fusion_ms"]
            + timings["temporal_ms"]
        )
        timings["overhead_ms"] = max(0.0, total_ms - comp_sum)

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


def run_webcam_benchmark(
    source_index: int = 0,
    target_frames: int = 500,
    warmup_frames: int = 10,
    output_dir: Path = PROJECT_ROOT / "data" / "performance",
) -> None:
    """Execute authoritative physical webcam validation benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase3_webcam_benchmark.csv"
    report_path = output_dir / "phase3_webcam_benchmark_report.txt"

    # Hardware & Environment Information
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU only)"
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    env_info = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "opencv_version": cv2.__version__,
        "mediapipe_version": mp.__version__,
    }

    logger.info("Initializing WebcamBenchmarkDetector (Phase 3 production pipeline)...")
    detector = WebcamBenchmarkDetector(config=DEFAULT_CONFIG, enable_audio=False)

    logger.info("Opening physical webcam source %d...", source_index)
    cap = cv2.VideoCapture(source_index)
    if not cap.isOpened():
        logger.error(
            "CRITICAL: Failed to open physical webcam device %d. Stop and verify webcam connection.",
            source_index,
        )
        detector.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 1. Warmup Phase (10 frames, not recorded)
    if warmup_frames > 0:
        logger.info("Executing %d warmup frames...", warmup_frames)
        w_done = 0
        while w_done < warmup_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            detector.process_frame_synchronized(frame)
            w_done += 1
        logger.info("Warmup phase complete.")

    records: List[Dict] = []
    fps_smooth_history: List[float] = []
    frame_idx = 0
    dropped_frames = 0

    logger.info("Starting authoritative webcam measurement for %d frames...", target_frames)
    t_bench_wall_start = time.perf_counter()

    try:
        while frame_idx < target_frames:
            # Measurement B: Application Loop / HUD-Style timing (matching main.py line 182)
            t_hud_loop_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                dropped_frames += 1
                logger.warning("Camera frame grab failed (dropped frame %d).", dropped_frames)
                time.sleep(0.01)
                continue

            frame_idx += 1

            # Measurement A: Synchronized pipeline timing (isolated pure pipeline execution)
            result, timings = detector.process_frame_synchronized(frame)

            t_hud_loop_end = time.perf_counter()
            hud_loop_latency_ms = (t_hud_loop_end - t_hud_loop_start) * 1000.0

            # HUD-style instant & smoothed FPS calculation (identical to main.py lines 204-208)
            hud_instant_fps = 1.0 / (hud_loop_latency_ms / 1000.0) if hud_loop_latency_ms > 0 else 30.0
            fps_smooth_history.append(hud_instant_fps)
            if len(fps_smooth_history) > 30:
                fps_smooth_history.pop(0)
            hud_smooth_fps = sum(fps_smooth_history) / len(fps_smooth_history)

            # True Synchronized End-to-End FPS
            sync_latency_ms = timings["sync_pipeline_latency_ms"]
            sync_fps = 1000.0 / sync_latency_ms if sync_latency_ms > 0 else 0.0

            state_name = result.state.name if hasattr(result.state, "name") else str(result.state)

            record = {
                "frame_number": frame_idx,
                "sync_pipeline_latency_ms": round(sync_latency_ms, 4),
                "sync_fps": round(sync_fps, 2),
                "hud_loop_latency_ms": round(hud_loop_latency_ms, 4),
                "hud_loop_fps": round(hud_instant_fps, 2),
                "hud_smooth_fps": round(hud_smooth_fps, 2),
                "face_detected": bool(result.face_detected),
                "final_state": state_name,
                "mediapipe_ms": round(timings["mediapipe_ms"], 4),
                "roi_ms": round(timings["roi_ms"], 4),
                "vit_ms": round(timings["vit_ms"], 4),
                "fusion_ms": round(timings["fusion_ms"], 4),
                "temporal_ms": round(timings["temporal_ms"], 4),
                "overhead_ms": round(timings["overhead_ms"], 4),
            }
            records.append(record)

            if frame_idx % 50 == 0 or frame_idx == target_frames:
                logger.info(
                    "Frame %d/%d | Sync: %.2f ms (%.1f FPS) | HUD Loop: %.2f ms (%.1f FPS) | Face: %s",
                    frame_idx,
                    target_frames,
                    sync_latency_ms,
                    sync_fps,
                    hud_loop_latency_ms,
                    hud_instant_fps,
                    result.face_detected,
                )

    finally:
        cap.release()
        detector.close()

    total_wall_time = time.perf_counter() - t_bench_wall_start
    logger.info("Webcam measurement complete: %d frames captured in %.2f s", len(records), total_wall_time)

    if not records:
        logger.error("No valid frames were recorded.")
        return

    # Write CSV
    fieldnames = [
        "frame_number",
        "sync_pipeline_latency_ms",
        "sync_fps",
        "hud_loop_latency_ms",
        "hud_loop_fps",
        "hud_smooth_fps",
        "face_detected",
        "final_state",
        "mediapipe_ms",
        "roi_ms",
        "vit_ms",
        "fusion_ms",
        "temporal_ms",
        "overhead_ms",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info("CSV saved to: %s", csv_path)

    # Compute Statistics
    n_frames = len(records)
    sync_latencies = np.array([r["sync_pipeline_latency_ms"] for r in records])
    sync_fps_vals = np.array([r["sync_fps"] for r in records])
    hud_loop_latencies = np.array([r["hud_loop_latency_ms"] for r in records])
    hud_fps_vals = np.array([r["hud_loop_fps"] for r in records])
    face_detected_vals = np.array([r["face_detected"] for r in records])

    mean_sync_lat = float(np.mean(sync_latencies))
    median_sync_lat = float(np.median(sync_latencies))
    p95_sync_lat = float(np.percentile(sync_latencies, 95))
    max_sync_lat = float(np.max(sync_latencies))

    mean_sync_fps = float(np.mean(sync_fps_vals))
    median_sync_fps = float(np.median(sync_fps_vals))
    min_sync_fps = float(np.min(sync_fps_vals))
    pct_above_28 = float((np.sum(sync_fps_vals >= 28.0) / n_frames) * 100.0)

    mean_hud_lat = float(np.mean(hud_loop_latencies))
    mean_hud_fps = float(np.mean(hud_fps_vals))
    median_hud_fps = float(np.median(hud_fps_vals))

    face_detection_rate = float((np.sum(face_detected_vals) / n_frames) * 100.0)

    # Component latencies across all frames
    mean_mediapipe = float(np.mean([r["mediapipe_ms"] for r in records]))
    mean_roi = float(np.mean([r["roi_ms"] for r in records]))
    mean_vit = float(np.mean([r["vit_ms"] for r in records]))
    mean_fusion = float(np.mean([r["fusion_ms"] for r in records]))
    mean_temporal = float(np.mean([r["temporal_ms"] for r in records]))
    mean_overhead = float(np.mean([r["overhead_ms"] for r in records]))

    # Active face frame breakdown
    face_recs = [r for r in records if r["face_detected"]]
    n_face = len(face_recs)
    if face_recs:
        act_sync_lat = float(np.mean([r["sync_pipeline_latency_ms"] for r in face_recs]))
        act_sync_fps = float(np.mean([r["sync_fps"] for r in face_recs]))
        act_roi = float(np.mean([r["roi_ms"] for r in face_recs]))
        act_vit = float(np.mean([r["vit_ms"] for r in face_recs]))
        act_fusion = float(np.mean([r["fusion_ms"] for r in face_recs]))
    else:
        act_sync_lat = act_sync_fps = act_roi = act_vit = act_fusion = 0.0

    # No face frame breakdown
    no_face_recs = [r for r in records if not r["face_detected"]]
    n_no_face = len(no_face_recs)
    if no_face_recs:
        no_face_sync_lat = float(np.mean([r["sync_pipeline_latency_ms"] for r in no_face_recs]))
        no_face_sync_fps = float(np.mean([r["sync_fps"] for r in no_face_recs]))
    else:
        no_face_sync_lat = no_face_sync_fps = 0.0

    # Write Human-Readable Report
    report_text = f"""======================================================================
PHASE 3 PHYSICAL WEBCAM VALIDATION REPORT
======================================================================
Date / Time:                      {time.strftime('%Y-%m-%d %H:%M:%S')}
Hardware Platform:               {env_info['processor']}
GPU Accelerator:                  {env_info['gpu_name']}
Operating System:                 {env_info['os']}
Python / PyTorch:                 {env_info['python_version']} / {env_info['pytorch_version']}
CUDA Enabled:                     {env_info['cuda_available']} (CUDA {env_info['cuda_version']})
OpenCV / MediaPipe:               {env_info['opencv_version']} / {env_info['mediapipe_version']}
Video Input Source:               Webcam Index {source_index} (640x480)
----------------------------------------------------------------------
EXECUTION & RELIABILITY:
  Target Post-Warmup Frames:      {target_frames}
  Successfully Processed:         {n_frames}
  Dropped / Failed Frames:        {dropped_frames}
  Face Detection Rate:            {face_detection_rate:.2f}% ({np.sum(face_detected_vals)}/{n_frames})
  Frames with Face Active:        {n_face}
  Frames without Face:            {n_no_face}
  Total Benchmark Wall Time:      {total_wall_time:.2f} s
----------------------------------------------------------------------
SYNCHRONIZED PIPELINE LATENCY (ms):
  Mean Latency:                   {mean_sync_lat:.2f} ms
  Median Latency:                 {median_sync_lat:.2f} ms
  95th Percentile (P95):          {p95_sync_lat:.2f} ms
  Maximum Latency:                {max_sync_lat:.2f} ms
  Mean Latency (Face Active):     {act_sync_lat:.2f} ms
  Mean Latency (No Face):         {no_face_sync_lat:.2f} ms
----------------------------------------------------------------------
THROUGHPUT & FPS COMPARISON:
  Synchronized End-to-End Mean:   {mean_sync_fps:.2f} FPS
  Synchronized End-to-End Median: {median_sync_fps:.2f} FPS
  Synchronized End-to-End Min:    {min_sync_fps:.2f} FPS
  Synchronized (Face Active Mean):{act_sync_fps:.2f} FPS
  Synchronized (No Face Mean):    {no_face_sync_fps:.2f} FPS
  Frames Meeting >= 28 FPS:       {pct_above_28:.2f}%

  Application / HUD Loop Mean:    {mean_hud_fps:.2f} FPS
  Application / HUD Loop Median:  {median_hud_fps:.2f} FPS
  Application / HUD Mean Latency: {mean_hud_lat:.2f} ms (includes camera I/O wait)
----------------------------------------------------------------------
COMPONENT LATENCY BREAKDOWN (Mean):
  MediaPipe FaceMesh:             {mean_mediapipe:.2f} ms
  Face ROI Extraction:            {mean_roi:.2f} ms (active: {act_roi:.2f} ms)
  ViT Inference (CUDA sync):      {mean_vit:.2f} ms (active: {act_vit:.2f} ms)
  Multimodal Feature Fusion:      {mean_fusion:.2f} ms (active: {act_fusion:.2f} ms)
  Temporal Decision Layer:        {mean_temporal:.2f} ms
  Overhead / HeadPose / Pipeline: {mean_overhead:.2f} ms
======================================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Report saved to: %s", report_path)
    print("\n" + report_text)

    # Generate Performance Graphs
    generate_webcam_plots(records, output_dir, mean_sync_fps, mean_hud_fps, mean_sync_lat)


def generate_webcam_plots(
    records: List[Dict],
    output_dir: Path,
    mean_sync_fps: float,
    mean_hud_fps: float,
    mean_sync_lat: float,
) -> None:
    """Generate 3 authoritative performance comparison graphs."""
    frames = np.array([r["frame_number"] for r in records])
    sync_fps = np.array([r["sync_fps"] for r in records])
    hud_fps = np.array([r["hud_loop_fps"] for r in records])
    hud_smooth = np.array([r["hud_smooth_fps"] for r in records])
    sync_lat = np.array([r["sync_pipeline_latency_ms"] for r in records])

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # =========================================================================
    # GRAPH 1: Webcam FPS Comparison (Synchronized vs Application / HUD Loop)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    ax1.plot(frames, sync_fps, color="#1f77b4", linewidth=1.1, alpha=0.85, label="Synchronized End-to-End FPS")
    ax1.plot(frames, hud_smooth, color="#ff7f0e", linewidth=1.4, linestyle="-", label="Application / HUD Loop FPS (Smoothed)")
    ax1.axhline(28.0, color="#d62728", linestyle="--", linewidth=2.0, label="Project Target (28 FPS)")
    ax1.axhline(mean_sync_fps, color="#1f77b4", linestyle=":", linewidth=1.6, label=f"Mean Synchronized ({mean_sync_fps:.1f} FPS)")
    ax1.axhline(mean_hud_fps, color="#ff7f0e", linestyle=":", linewidth=1.6, label=f"Mean HUD Loop ({mean_hud_fps:.1f} FPS)")

    ax1.set_title("Webcam FPS: Synchronized End-to-End vs Application / HUD Loop", fontsize=12.5, fontweight="bold", pad=12)
    ax1.set_xlabel("Frame Number", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("FPS", fontsize=11, fontweight="semibold")
    ax1.set_xlim(frames[0], frames[-1])
    ax1.set_ylim(0, max(float(np.max(sync_fps)), float(np.max(hud_fps)), 35.0) * 1.15)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)
    plt.tight_layout()

    g1_path = output_dir / "webcam_fps_comparison.png"
    fig1.savefig(g1_path, dpi=300)
    plt.close(fig1)
    logger.info("Saved Graph 1: %s", g1_path)

    # =========================================================================
    # GRAPH 2: Synchronized Processing Latency Over Frames
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    ax2.plot(frames, sync_lat, color="#2ca02c", linewidth=1.1, alpha=0.85, label="Measured Synchronized Latency (ms)")
    ax2.axhline(mean_sync_lat, color="#1f77b4", linestyle=":", linewidth=1.8, label=f"Mean Latency ({mean_sync_lat:.2f} ms)")
    target_budget_ms = 1000.0 / 28.0
    ax2.axhline(target_budget_ms, color="#d62728", linestyle="--", linewidth=2.0, label=f"Target Latency Budget ({target_budget_ms:.1f} ms for 28 FPS)")

    ax2.set_title("Webcam Synchronized Processing Latency Over Frames", fontsize=12.5, fontweight="bold", pad=12)
    ax2.set_xlabel("Frame Number", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Processing Latency (ms)", fontsize=11, fontweight="semibold")
    ax2.set_xlim(frames[0], frames[-1])
    ax2.set_ylim(0, max(float(np.max(sync_lat)) * 1.2, 50.0))
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)
    plt.tight_layout()

    g2_path = output_dir / "webcam_latency_over_frames.png"
    fig2.savefig(g2_path, dpi=300)
    plt.close(fig2)
    logger.info("Saved Graph 2: %s", g2_path)

    # =========================================================================
    # GRAPH 3: Component-wise Mean Latency Breakdown
    # =========================================================================
    face_recs = [r for r in records if r["face_detected"]]
    if face_recs:
        c_names = ["MediaPipe", "Face ROI", "ViT (CUDA)", "Feature Fusion", "Temporal Dec.", "Overhead"]
        c_vals = [
            float(np.mean([r["mediapipe_ms"] for r in records])),
            float(np.mean([r["roi_ms"] for r in face_recs])),
            float(np.mean([r["vit_ms"] for r in face_recs])),
            float(np.mean([r["fusion_ms"] for r in face_recs])),
            float(np.mean([r["temporal_ms"] for r in records])),
            float(np.mean([r["overhead_ms"] for r in records])),
        ]
    else:
        c_names = ["MediaPipe", "Temporal Dec.", "Overhead"]
        c_vals = [
            float(np.mean([r["mediapipe_ms"] for r in records])),
            float(np.mean([r["temporal_ms"] for r in records])),
            float(np.mean([r["overhead_ms"] for r in records])),
        ]

    fig3, ax3 = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#b07aa1"][:len(c_names)]
    bars = ax3.bar(c_names, c_vals, color=colors, width=0.55, edgecolor="#333333", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + max(c_vals) * 0.02,
            f"{h:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    ax3.set_title("Webcam Pipeline Component Mean Latency Breakdown", fontsize=12.5, fontweight="bold", pad=12)
    ax3.set_ylabel("Latency (ms)", fontsize=11, fontweight="semibold")
    ax3.set_ylim(0, max(c_vals) * 1.25 + 2.0)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.xticks(fontsize=10, fontweight="semibold")
    plt.tight_layout()

    g3_path = output_dir / "webcam_component_latency.png"
    fig3.savefig(g3_path, dpi=300)
    plt.close(fig3)
    logger.info("Saved Graph 3: %s", g3_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Authoritative Physical Webcam Performance Validation")
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Physical webcam source device index (default: 0)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=500,
        help="Number of post-warmup frames to measure (default: 500)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup frames to process (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "performance"),
        help="Destination directory for CSV and graphs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_webcam_benchmark(
        source_index=args.source,
        target_frames=args.frames,
        warmup_frames=args.warmup,
        output_dir=Path(args.output_dir),
    )
