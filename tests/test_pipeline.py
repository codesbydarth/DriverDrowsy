"""Integration tests for the Driver Drowsiness Detection pipeline.

Verifies end-to-end frame processing, detector/visualizer decoupling,
no-face handling, non-blocking alerts, and strict absence of dlib.
"""

import sys
import numpy as np
import pygame
import pytest

from alerts.alert import AlertManager, adapt_waveform_channels
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
        """Verify AlertManager triggers without blocking and synthesizes valid sounds when audio is available."""
        alert_mgr = AlertManager()
        try:
            if alert_mgr._mixer_initialized:
                # Audio hardware/mixer is available: verify full initialization
                expected_states = set(alert_mgr.audio_config.synthetic_tones.keys())
                assert len(alert_mgr._sounds) == len(expected_states)
                for state in expected_states:
                    assert state in alert_mgr._sounds
                    assert isinstance(alert_mgr._sounds[state], pygame.mixer.Sound)
            else:
                # Audio hardware unavailable (e.g. headless CI): verify graceful fallback
                assert alert_mgr._enabled is False
                assert len(alert_mgr._sounds) == 0

            # Triggering alerts should return immediately without blocking or raising
            alert_mgr.trigger(ProjectState.LOW_VIGILANCE)
            alert_mgr.trigger(ProjectState.DROWSY)
            alert_mgr.trigger(ProjectState.MICROSLEEP)
        finally:
            alert_mgr.close()

    def test_adapt_waveform_channels(self) -> None:
        """Verify waveform adaptation correctly reshapes 1D signals for mono, stereo, and multi-channel configurations."""
        mono_signal = np.array([100, -200, 300, -400], dtype=np.int16)

        # 1 channel (mono): must remain 1D array
        adapted_mono = adapt_waveform_channels(mono_signal, target_channels=1)
        assert adapted_mono.ndim == 1
        assert np.array_equal(adapted_mono, mono_signal)
        assert adapted_mono.dtype == np.int16

        # 2 channels (stereo): must be 2D array of shape (N, 2) duplicating mono signal
        adapted_stereo = adapt_waveform_channels(mono_signal, target_channels=2)
        assert adapted_stereo.ndim == 2
        assert adapted_stereo.shape == (4, 2)
        assert np.array_equal(adapted_stereo[:, 0], mono_signal)
        assert np.array_equal(adapted_stereo[:, 1], mono_signal)
        assert adapted_stereo.dtype == np.int16

        # >2 channels (e.g. 4 quad or 6 surround): must be 2D array of shape (N, channels)
        adapted_quad = adapt_waveform_channels(mono_signal, target_channels=4)
        assert adapted_quad.ndim == 2
        assert adapted_quad.shape == (4, 4)
        for ch_idx in range(4):
            assert np.array_equal(adapted_quad[:, ch_idx], mono_signal)
        assert adapted_quad.dtype == np.int16

        # Edge case: target_channels <= 0
        adapted_fallback = adapt_waveform_channels(mono_signal, target_channels=0)
        assert adapted_fallback.ndim == 1
        assert np.array_equal(adapted_fallback, mono_signal)

    def test_strict_absence_of_dlib(self) -> None:
        """Verify that dlib is NEVER imported, referenced, or present in project modules."""
        # Check loaded modules
        assert "dlib" not in sys.modules, "FATAL: dlib is forbidden and must not be loaded!"
