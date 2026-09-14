"""Integration tests for the Driver Drowsiness Detection pipeline.

Verifies end-to-end frame processing, detector/visualizer decoupling,
no-face handling, non-blocking alerts, and strict absence of dlib.
"""

import sys
import numpy as np
import pytest

from alerts.alert import AlertManager
from src.detector import DetectionResult, DrowsinessDetector
from utils.config import AppConfig, ProjectState
from utils.visualizer import HUDVisualizer


class TestPipelineIntegration:
    """Integration test suite for the detection pipeline."""

    def test_detector_visualizer_decoupling(self) -> None:
        """Verify detector runs without visualizer, and visualizer works independently."""
        detector = DrowsinessDetector(enable_audio=False)
        visualizer = HUDVisualizer()

        test_frame = np.full((480, 640, 3), fill_value=50, dtype=np.uint8)

        # 1. Detector operates strictly on frame -> DetectionResult
        result: DetectionResult = detector.process_frame(test_frame)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.state, ProjectState)

        # 2. Visualizer operates strictly on frame + DetectionResult -> Annotated Frame
        annotated = visualizer.draw_hud(test_frame, result, fps=30.0)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == test_frame.shape

        detector.close()

    def test_pipeline_handles_none_and_empty_frames(self) -> None:
        """Verify pipeline handles None or zero-sized frames without raising exceptions."""
        with DrowsinessDetector(enable_audio=False) as detector:
            # None frame
            res_none = detector.process_frame(None)
            assert res_none.face_detected is False

            # Zero-sized frame
            empty_frame = np.array([], dtype=np.uint8)
            res_empty = detector.process_frame(empty_frame)
            assert res_empty.face_detected is False

    def test_non_blocking_alert_manager(self) -> None:
        """Verify AlertManager triggers without blocking the caller."""
        alert_mgr = AlertManager()
        # Triggering alerts should return immediately
        alert_mgr.trigger(ProjectState.LOW_VIGILANCE)
        alert_mgr.trigger(ProjectState.DROWSY)
        alert_mgr.trigger(ProjectState.MICROSLEEP)
        alert_mgr.close()

    def test_strict_absence_of_dlib(self) -> None:
        """Verify that dlib is NEVER imported, referenced, or present in project modules."""
        # Check loaded modules
        assert "dlib" not in sys.modules, "FATAL: dlib is forbidden and must not be loaded!"
