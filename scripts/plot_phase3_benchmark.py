"""Phase 3 Performance Benchmark Visualization.

Generates the three required performance graphs from benchmark CSV data:
1. data/performance/fps_over_frames.png (FPS over frames with 28 FPS project target)
2. data/performance/latency_over_frames.png (Processing latency over frames)
3. data/performance/component_latency.png (Component-wise mean latency breakdown)
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("plot_phase3_benchmark")


def load_benchmark_csv(csv_path: Path) -> List[Dict]:
    """Read benchmark data from CSV."""
    if not csv_path.exists():
        logger.error("Benchmark CSV does not exist: %s", csv_path)
        sys.exit(1)

    records = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "frame_number": int(row["frame_number"]),
                "total_latency_ms": float(row["total_latency_ms"]),
                "FPS": float(row["FPS"]),
                "face_detected": row["face_detected"].strip().lower() in ("true", "1"),
                "final_state": row["final_state"],
                "landmark_ms": float(row.get("landmark_ms", 0.0)),
                "roi_ms": float(row.get("roi_ms", 0.0)),
                "vit_ms": float(row.get("vit_ms", 0.0)),
                "fusion_ms": float(row.get("fusion_ms", 0.0)),
                "temporal_ms": float(row.get("temporal_ms", 0.0)),
            })
    return records


def generate_plots(csv_path: Path, output_dir: Path) -> None:
    """Generate all three performance visualization graphs from CSV data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_benchmark_csv(csv_path)

    if not records:
        logger.error("No data records found in CSV.")
        return

    frames = np.array([r["frame_number"] for r in records])
    fps = np.array([r["FPS"] for r in records])
    latency = np.array([r["total_latency_ms"] for r in records])

    mean_fps = float(np.mean(fps))
    mean_latency = float(np.mean(latency))

    # Configure style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # =========================================================================
    # GRAPH 1: FPS over frame number
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    ax1.plot(frames, fps, color="#1f77b4", linewidth=1.2, alpha=0.85, label="Measured Throughput (FPS)")
    ax1.axhline(
        28.0,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="Project Target (28 FPS)",
    )
    ax1.axhline(
        mean_fps,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label=f"Mean Throughput ({mean_fps:.1f} FPS)",
    )

    ax1.set_title("Phase 3 Pipeline Processing Throughput (FPS)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Frame Number", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("FPS", fontsize=11, fontweight="semibold")
    ax1.set_xlim(frames[0], frames[-1])
    ax1.set_ylim(bottom=0, top=max(float(np.max(fps)) * 1.15, 35.0))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    fps_plot_path = output_dir / "fps_over_frames.png"
    fig1.savefig(fps_plot_path, dpi=300)
    plt.close(fig1)
    logger.info("Saved Graph 1: %s", fps_plot_path)

    # =========================================================================
    # GRAPH 2: Latency over frame number
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=300)
    ax2.plot(frames, latency, color="#ff7f0e", linewidth=1.2, alpha=0.85, label="Measured Latency (ms)")
    ax2.axhline(
        mean_latency,
        color="#1f77b4",
        linestyle=":",
        linewidth=1.8,
        label=f"Mean Latency ({mean_latency:.2f} ms)",
    )
    # Target latency corresponding to 28 FPS is 1000/28 = 35.71 ms
    target_latency = 1000.0 / 28.0
    ax2.axhline(
        target_latency,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Target Latency Budget (35.7 ms for 28 FPS)",
    )

    ax2.set_title("Phase 3 End-to-End Processing Latency", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Frame Number", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Processing Latency (ms)", fontsize=11, fontweight="semibold")
    ax2.set_xlim(frames[0], frames[-1])
    ax2.set_ylim(bottom=0, top=max(float(np.max(latency)) * 1.2, 50.0))
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    latency_plot_path = output_dir / "latency_over_frames.png"
    fig2.savefig(latency_plot_path, dpi=300)
    plt.close(fig2)
    logger.info("Saved Graph 2: %s", latency_plot_path)

    # =========================================================================
    # GRAPH 3: Component-wise mean latency
    # =========================================================================
    # Component breakdown: MediaPipe, Face ROI, ViT, Feature Fusion, Temporal Decision
    component_names = [
        "MediaPipe",
        "Face ROI",
        "ViT",
        "Feature Fusion",
        "Temporal Decision",
    ]

    mean_mediapipe = float(np.mean([r["landmark_ms"] for r in records]))
    # For components that execute on face-detected frames, use active mean
    face_records = [r for r in records if r["face_detected"]]
    if face_records:
        mean_roi = float(np.mean([r["roi_ms"] for r in face_records]))
        mean_vit = float(np.mean([r["vit_ms"] for r in face_records]))
        mean_fusion = float(np.mean([r["fusion_ms"] for r in face_records]))
    else:
        mean_roi = float(np.mean([r["roi_ms"] for r in records]))
        mean_vit = float(np.mean([r["vit_ms"] for r in records]))
        mean_fusion = float(np.mean([r["fusion_ms"] for r in records]))

    mean_temporal = float(np.mean([r["temporal_ms"] for r in records]))

    component_means = [
        mean_mediapipe,
        mean_roi,
        mean_vit,
        mean_fusion,
        mean_temporal,
    ]

    fig3, ax3 = plt.subplots(figsize=(9, 5.5), dpi=300)
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    bars = ax3.bar(component_names, component_means, color=colors, width=0.55, edgecolor="#333333", linewidth=0.8)

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.3,
            f"{h:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    ax3.set_title("Component-wise Mean Latency Breakdown", fontsize=13, fontweight="bold", pad=12)
    ax3.set_ylabel("Latency (ms)", fontsize=11, fontweight="semibold")
    ax3.set_ylim(0, max(component_means) * 1.25 + 2.0)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.xticks(fontsize=10.5, fontweight="semibold")
    plt.tight_layout()

    component_plot_path = output_dir / "component_latency.png"
    fig3.savefig(component_plot_path, dpi=300)
    plt.close(fig3)
    logger.info("Saved Graph 3: %s", component_plot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 3 Benchmark Metrics")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "performance" / "phase3_benchmark.csv"),
        help="Input benchmark CSV path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "performance"),
        help="Output directory for generated plots",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_plots(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
    )
