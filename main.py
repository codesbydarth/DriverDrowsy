"""Main executable entry point for Driver Drowsiness Detection System.

Supports live webcam feeds, video file inputs, and synthetic/headless test mode.
Strictly calculates actual measured FPS from high-resolution frame timestamps.
Decouples detection from visualization following the architecture:
Frame -> Detector -> DetectionResult -> Visualizer -> Annotated Frame.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from src.detector import DetectionResult, DrowsinessDetector
from utils.config import DEFAULT_CONFIG, AppConfig, ProjectState
from utils.logger import setup_logger
from utils.visualizer import HUDVisualizer

logger = setup_logger("main")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System (Phase 1 Classical Baseline)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam device index (e.g. '0') or path to video file."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on synthetic generated test frames to verify pipeline execution without webcam."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process in synthetic benchmark mode (default: 100)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (cv2.imshow)."
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable acoustic audio alert sounds."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture frame width (default: 640)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture frame height (default: 480)."
    )
    return parser.parse_args()


def run_synthetic_benchmark(
    detector: DrowsinessDetector,
    visualizer: HUDVisualizer,
    num_frames: int = 100,
    headless: bool = False
) -> None:
    """Execute pipeline on synthetic frames to verify execution and failure handling.

    Args:
        detector: Initialized DrowsinessDetector instance.
        visualizer: Initialized HUDVisualizer instance.
        num_frames: Total number of test frames to process.
        headless: If True, suppress display window.
    """
    logger.info("Starting synthetic pipeline benchmark (%d frames)...", num_frames)

    width, height = 640, 480
    frame_times = []

    for i in range(num_frames):
        t_start = time.perf_counter()

        # Generate a test image (alternating blank and patterned to test robustness)
        synthetic_frame = np.full((height, width, 3), fill_value=40, dtype=np.uint8)
        if i % 10 == 0:
            # Draw simple geometry to test renderer
            cv2.circle(synthetic_frame, (320, 240), 80, (120, 120, 120), -1)

        # 1. Detection
        result: DetectionResult = detector.process_frame(synthetic_frame)

        # 2. Visualization
        annotated_frame = visualizer.draw_hud(synthetic_frame, result, fps=0.0)

        elapsed = time.perf_counter() - t_start
        frame_times.append(elapsed)

        if not headless:
            cv2.imshow("Driver Drowsiness Detection (Synthetic Test)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Early exit requested by user.")
                break

    if not headless:
        cv2.destroyAllWindows()

    avg_time = sum(frame_times) / len(frame_times) if frame_times else 0.0
    actual_fps = (1.0 / avg_time) if avg_time > 0 else 0.0
    logger.info(
        "Synthetic benchmark completed: %d frames processed | "
        "Avg frame time: %.2f ms | Actual throughput: %.1f FPS",
        len(frame_times), avg_time * 1000, actual_fps
    )


def run_video_stream(
    source_str: str,
    detector: DrowsinessDetector,
    visualizer: HUDVisualizer,
    width: int = 640,
    height: int = 480,
    headless: bool = False
) -> None:
    """Run real-time drowsiness detection loop from camera or video file.

    Args:
        source_str: Camera index string or path to video file.
        detector: DrowsinessDetector pipeline instance.
        visualizer: HUDVisualizer instance.
        width: Requested capture width.
        height: Requested capture height.
        headless: If True, suppress OpenCV window display.
    """
    # Determine camera index vs video file path
    if source_str.isdigit():
        source = int(source_str)
        is_camera = True
    else:
        source = Path(source_str)
        if not source.exists():
            logger.error("Specified video file does not exist: %s", source)
            return
        source = str(source)
        is_camera = False

    logger.info("Opening video source: %s", source_str)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(
            "Failed to open video source '%s'. Verify camera connection or file permissions.",
            source_str
        )
        return

    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps_history = []
    fps_smooth = 0.0
    frame_count = 0
    max_frames = getattr(detector.config, "max_frames", 0)

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_camera:
                    logger.warning("Failed to grab frame from camera. Retrying...")
                    time.sleep(0.05)
                    continue
                else:
                    logger.info("Reached end of video stream.")
                    break

            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                logger.info("Reached requested frame limit (%d frames).", max_frames)
                break

            # 1. Detection Pipeline (pure data processing)
            result: DetectionResult = detector.process_frame(frame)

            # 2. Calculate true measured FPS
            t_elapsed = time.perf_counter() - t_start
            fps_instant = 1.0 / t_elapsed if t_elapsed > 0 else 30.0
            fps_history.append(fps_instant)
            if len(fps_history) > 30:
                fps_history.pop(0)
            fps_smooth = sum(fps_history) / len(fps_history)

            # 3. Visualization Pipeline
            annotated_frame = visualizer.draw_hud(frame, result, fps=fps_smooth)

            if not headless:
                cv2.imshow("Driver Drowsiness Detection System", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exit requested via 'q' key.")
                    break
                elif key == ord('r'):
                    detector.reset()
                    logger.info("State reset requested via 'r' key.")

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        logger.info("Video capture released. Processed %d frames.", frame_count)


def main() -> None:
    """Main application routine."""
    args = parse_arguments()
    config = DEFAULT_CONFIG
    if args.frames > 0:
        setattr(config, "max_frames", args.frames)

    enable_audio = not args.no_audio
    logger.info("Initializing Driver Drowsiness Detection System (Phase 1 Classical Baseline)...")

    try:
        with DrowsinessDetector(config=config, enable_audio=enable_audio) as detector:
            visualizer = HUDVisualizer(config=config)

            if args.synthetic:
                run_synthetic_benchmark(
                    detector=detector,
                    visualizer=visualizer,
                    num_frames=args.frames,
                    headless=args.headless
                )
            else:
                run_video_stream(
                    source_str=args.source,
                    detector=detector,
                    visualizer=visualizer,
                    width=args.width,
                    height=args.height,
                    headless=args.headless
                )
    except Exception as err:
        logger.error("Fatal error during execution: %s", err, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
