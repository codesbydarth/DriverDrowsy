"""Unit tests for classical baseline drowsiness detection logic.

Tests EAR, MAR, PERCLOS buffer, head pose estimation, and temporal state transitions
using deterministic geometric mock landmarks without requiring a webcam.
"""

import numpy as np
import pytest

from models.baseline import (
    BaselineDetector,
    PERCLOSBuffer,
    calculate_ear,
    calculate_mar,
)
from src.head_pose import HeadPoseEstimator, HeadPoseResult
from utils.config import AppConfig, LandmarkIndices, ProjectState


@pytest.fixture
def landmark_indices() -> LandmarkIndices:
    """Fixture providing standard project landmark indices."""
    return LandmarkIndices()


@pytest.fixture
def mock_open_eye_landmarks(landmark_indices: LandmarkIndices) -> np.ndarray:
    """Deterministic mock landmarks for a face with open eyes and closed mouth."""
    landmarks = np.zeros((468, 2), dtype=np.float32)

    # Left eye: [33, 160, 158, 133, 153, 144]
    # Horizontal: 33 -> 133 (distance = 40.0)
    # Vertical 1: 160 -> 144 (distance = 14.0)
    # Vertical 2: 158 -> 153 (distance = 14.0)
    # Expected EAR = (14 + 14) / (2 * 40) = 28 / 80 = 0.35 (> 0.25 threshold)
    landmarks[33] = [100.0, 100.0]
    landmarks[133] = [140.0, 100.0]
    landmarks[160] = [115.0, 93.0]
    landmarks[144] = [115.0, 107.0]
    landmarks[158] = [125.0, 93.0]
    landmarks[153] = [125.0, 107.0]

    # Right eye: [362, 385, 387, 263, 373, 380]
    landmarks[362] = [200.0, 100.0]
    landmarks[263] = [240.0, 100.0]
    landmarks[385] = [215.0, 93.0]
    landmarks[380] = [215.0, 107.0]
    landmarks[387] = [225.0, 93.0]
    landmarks[373] = [225.0, 107.0]

    # Mouth: [61, 291, 39, 181, 0, 17, 269, 405]
    # Horizontal: 61 -> 291 (distance = 60.0)
    # Vertical distances: 6.0, 8.0, 6.0 -> sum = 20.0
    # Expected MAR = 20 / (2 * 60) = 0.167 (< 0.60 threshold)
    landmarks[61] = [140.0, 200.0]
    landmarks[291] = [200.0, 200.0]
    landmarks[39] = [155.0, 197.0]
    landmarks[181] = [155.0, 203.0]
    landmarks[0] = [170.0, 196.0]
    landmarks[17] = [170.0, 204.0]
    landmarks[269] = [185.0, 197.0]
    landmarks[405] = [185.0, 203.0]

    return landmarks


@pytest.fixture
def mock_closed_eye_landmarks(mock_open_eye_landmarks: np.ndarray) -> np.ndarray:
    """Mock landmarks where eyes are closed."""
    landmarks = mock_open_eye_landmarks.copy()
    # Collapse eye vertical distance to 2.0 px
    # Expected EAR = (2 + 2) / (2 * 40) = 0.05 (< 0.25 threshold)
    landmarks[160] = [115.0, 99.0]
    landmarks[144] = [115.0, 101.0]
    landmarks[158] = [125.0, 99.0]
    landmarks[153] = [125.0, 101.0]

    landmarks[385] = [215.0, 99.0]
    landmarks[380] = [215.0, 101.0]
    landmarks[387] = [225.0, 99.0]
    landmarks[373] = [225.0, 101.0]

    return landmarks


@pytest.fixture
def mock_yawning_landmarks(mock_open_eye_landmarks: np.ndarray) -> np.ndarray:
    """Mock landmarks where mouth is wide open (yawning)."""
    landmarks = mock_open_eye_landmarks.copy()
    # Expand mouth vertical distances: 35.0, 45.0, 35.0 -> sum = 115.0
    # Expected MAR = 115 / (2 * 60) = 0.958 (> 0.60 threshold)
    landmarks[39] = [155.0, 180.0]
    landmarks[181] = [155.0, 215.0]
    landmarks[0] = [170.0, 175.0]
    landmarks[17] = [170.0, 220.0]
    landmarks[269] = [185.0, 180.0]
    landmarks[405] = [185.0, 215.0]

    return landmarks


class TestSignalCalculations:
    """Tests for EAR and MAR mathematical computations."""

    def test_ear_open_vs_closed(
        self,
        mock_open_eye_landmarks: np.ndarray,
        mock_closed_eye_landmarks: np.ndarray,
        landmark_indices: LandmarkIndices
    ) -> None:
        """Verify EAR is high for open eyes and drops significantly when closed."""
        ear_open_l = calculate_ear(mock_open_eye_landmarks, landmark_indices.LEFT_EYE)
        ear_open_r = calculate_ear(mock_open_eye_landmarks, landmark_indices.RIGHT_EYE)
        assert ear_open_l > 0.30
        assert ear_open_r > 0.30

        ear_closed_l = calculate_ear(mock_closed_eye_landmarks, landmark_indices.LEFT_EYE)
        ear_closed_r = calculate_ear(mock_closed_eye_landmarks, landmark_indices.RIGHT_EYE)
        assert ear_closed_l < 0.10
        assert ear_closed_r < 0.10

    def test_mar_closed_vs_yawning(
        self,
        mock_open_eye_landmarks: np.ndarray,
        mock_yawning_landmarks: np.ndarray,
        landmark_indices: LandmarkIndices
    ) -> None:
        """Verify MAR is low for normal mouth and rises above 0.60 during a yawn."""
        mar_normal = calculate_mar(mock_open_eye_landmarks, landmark_indices.MOUTH)
        assert mar_normal < 0.30

        mar_yawn = calculate_mar(mock_yawning_landmarks, landmark_indices.MOUTH)
        assert mar_yawn > 0.60

    def test_perclos_buffer_rolling_window(self) -> None:
        """Verify PERCLOS calculates rolling proportion of closed eye frames correctly."""
        buf = PERCLOSBuffer(window_size=10)
        assert buf.value == 0.0

        # Feed 3 closed frames
        for _ in range(3):
            buf.update(True)
        assert pytest.approx(buf.value, 0.01) == 0.30

        # Feed 7 open frames -> buffer now holds 10 frames (3 closed, 7 open)
        for _ in range(7):
            buf.update(False)
        assert pytest.approx(buf.value, 0.01) == 0.30

        # Feed 10 open frames -> all closed frames pushed out of rolling window
        for _ in range(10):
            buf.update(False)
        assert buf.value == 0.0


class TestTemporalLogicAndSafety:
    """Tests for temporal logic, false positive rejection, and state transitions."""

    def test_normal_blinking_remains_alert(
        self,
        mock_open_eye_landmarks: np.ndarray,
        mock_closed_eye_landmarks: np.ndarray
    ) -> None:
        """Verify normal eye blinks (1-4 frames) DO NOT trigger LOW_VIGILANCE or DROWSY."""
        detector = BaselineDetector()

        # Simulate 30 frames of normal driving
        for _ in range(30):
            res = detector.update(mock_open_eye_landmarks)
            assert res.state == ProjectState.ALERT

        # Simulate a quick 2-frame normal blink
        for _ in range(2):
            res = detector.update(mock_closed_eye_landmarks)
            # CRITICAL: Must NOT trigger LOW_VIGILANCE or DROWSY on a normal blink
            assert res.state == ProjectState.ALERT

        # Eyes reopen
        for _ in range(10):
            res = detector.update(mock_open_eye_landmarks)
            assert res.state == ProjectState.ALERT

    def test_sustained_eye_closure_triggers_microsleep(
        self,
        mock_open_eye_landmarks: np.ndarray,
        mock_closed_eye_landmarks: np.ndarray
    ) -> None:
        """Verify continuous eye closure >= 10 frames triggers MICROSLEEP."""
        detector = BaselineDetector()

        # Warm up with alert frames
        for _ in range(10):
            detector.update(mock_open_eye_landmarks)

        # 9 consecutive frames of eye closure (not yet microsleep)
        for i in range(1, 10):
            res = detector.update(mock_closed_eye_landmarks)
            assert res.continuous_closed_frames == i
            assert res.state != ProjectState.MICROSLEEP

        # 10th consecutive frame reaches MICROSLEEP_PERSISTENCE
        res = detector.update(mock_closed_eye_landmarks)
        assert res.continuous_closed_frames == 10
        assert res.state == ProjectState.MICROSLEEP

    def test_isolated_yawn_does_not_trigger_drowsy(
        self,
        mock_open_eye_landmarks: np.ndarray,
        mock_yawning_landmarks: np.ndarray
    ) -> None:
        """Verify a single isolated yawn does not immediately trigger DROWSY."""
        detector = BaselineDetector()

        for _ in range(15):
            detector.update(mock_open_eye_landmarks)

        # A quick 5-frame mouth open (e.g. talking or short yawn)
        for _ in range(5):
            res = detector.update(mock_yawning_landmarks)
            # Isolated mouth opening must not directly escalate to full DROWSY
            assert res.state != ProjectState.DROWSY

    def test_sustained_fatigue_escalates_to_drowsy(
        self,
        mock_yawning_landmarks: np.ndarray
    ) -> None:
        """Verify prolonged fatigue cues (yawn persistence + drowsy persistence) trigger DROWSY."""
        detector = BaselineDetector()

        # Feed sustained yawning for >= 45 frames (15 frames to qualify yawn + 30 frames persistence)
        for _ in range(50):
            res = detector.update(mock_yawning_landmarks)

        assert res.state == ProjectState.DROWSY

    def test_alert_recovery_mechanism(
        self,
        mock_open_eye_landmarks: np.ndarray,
        mock_yawning_landmarks: np.ndarray
    ) -> None:
        """Verify driver recovers to ALERT after 15 consecutive normal frames."""
        detector = BaselineDetector()

        # Push to DROWSY state
        for _ in range(50):
            detector.update(mock_yawning_landmarks)
        assert detector.current_state == ProjectState.DROWSY

        # Feed 14 normal frames -> not quite recovered yet
        for _ in range(14):
            res = detector.update(mock_open_eye_landmarks)
            assert res.state == ProjectState.DROWSY

        # 15th normal frame triggers full recovery to ALERT
        res = detector.update(mock_open_eye_landmarks)
        assert res.state == ProjectState.ALERT

    def test_missing_face_handling(self) -> None:
        """Verify baseline detector gracefully handles None (missing face) without crashing."""
        detector = BaselineDetector()
        res = detector.update(None)
        assert res.state == ProjectState.ALERT
        assert "No face detected" in res.reasons


class TestHeadPoseEstimation:
    """Tests for 3D Head Pose estimation."""

    def test_head_pose_neutral_face(
        self,
        mock_open_eye_landmarks: np.ndarray
    ) -> None:
        """Verify head pose estimation computes angles for valid landmarks."""
        estimator = HeadPoseEstimator()

        # Fill key head pose points in 640x480 frame
        landmarks = mock_open_eye_landmarks.copy()
        landmarks[1] = [320.0, 240.0]   # Nose tip
        landmarks[152] = [320.0, 350.0] # Chin
        landmarks[33] = [245.0, 183.0]  # Left eye corner
        landmarks[263] = [395.0, 183.0] # Right eye corner
        landmarks[61] = [270.0, 290.0]  # Left mouth corner
        landmarks[291] = [370.0, 290.0] # Right mouth corner

        res: HeadPoseResult = estimator.estimate_pose(landmarks, frame_shape=(480, 640))
        assert res.success is True
        # Neutral face angles should be reasonably small
        assert abs(res.pitch) < 30.0
        assert abs(res.yaw) < 30.0
        assert len(res.nose_end_point2d) == 2

    def test_head_pose_invalid_input(self) -> None:
        """Verify head pose handles empty or insufficient landmarks gracefully."""
        estimator = HeadPoseEstimator()
        res = estimator.estimate_pose(None, frame_shape=(480, 640))
        assert res.success is False

        short_landmarks = np.zeros((10, 2), dtype=np.float32)
        res = estimator.estimate_pose(short_landmarks, frame_shape=(480, 640))
        assert res.success is False
