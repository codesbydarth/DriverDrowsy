"""Main Driver Drowsiness Detector module.

Coordinates MediaPipe landmark detection, 3D head pose estimation,
classical baseline signal analysis (EAR, MAR, PERCLOS, temporal state logic),
and non-blocking acoustic alerts. Fully decoupled from GUI/HUD visualization,
adhering to the pipeline architecture:
Frame -> Detector -> DetectionResult -> Visualizer -> Annotated Frame.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from alerts.alert import AlertManager
from models.baseline import BaselineDetector
from src.head_pose import HeadPoseEstimator, HeadPoseResult
from src.landmark_detector import FaceLandmarks, FaceMeshDetector
from utils.config import DEFAULT_CONFIG, AppConfig, ProjectState
from utils.logger import setup_logger

logger = setup_logger("detector")


@dataclass
class DetectionResult:
    """Comprehensive output container for a single analyzed video frame.

    Attributes:
        state: Active ProjectState (ALERT, LOW_VIGILANCE, DROWSY, MICROSLEEP).
        left_ear: Left Eye Aspect Ratio.
        right_ear: Right Eye Aspect Ratio.
        avg_ear: Combined Eye Aspect Ratio.
        mar: Mouth Aspect Ratio.
        perclos: Rolling window PERCLOS percentage in [0.0, 1.0].
        eyes_closed: Current frame eye closure indicator.
        continuous_closed_frames: Consecutive eye-closure frame count.
        drowsy_persistence_counter: Accumulated fatigue persistence frame count.
        alert_reset_counter: Consecutive alert recovery frame count.
        is_yawning: Sustained mouth opening indicator.
        is_nodding: Sustained head dropping indicator.
        is_distracted: Lateral head diversion indicator.
        reasons: Trigger descriptions justifying the active state.
        face_detected: True if a human face was successfully located.
        pixel_landmarks: (468, 2) facial landmark coordinates in pixels.
        normalized_landmarks: (468, 3) normalized facial landmark coordinates.
        bbox: Bounding box tuple (x_min, y_min, x_max, y_max) in pixel space.
        head_pose: HeadPoseResult with 3D angles and projection endpoints.
    """
    state: ProjectState = ProjectState.ALERT
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    eyes_closed: bool = False
    continuous_closed_frames: int = 0
    drowsy_persistence_counter: int = 0
    alert_reset_counter: int = 0
    is_yawning: bool = False
    is_nodding: bool = False
    is_distracted: bool = False
    reasons: List[str] = field(default_factory=list)
    face_detected: bool = False
    pixel_landmarks: Optional[np.ndarray] = None
    normalized_landmarks: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    head_pose: Optional[HeadPoseResult] = None


class DrowsinessDetector:
    """Core drowsiness detection pipeline orchestrator.

    Extracts facial landmarks via MediaPipe, estimates 3D head pose via solvePnP,
    evaluates EAR, MAR, and rolling PERCLOS against configurable thresholds,
    applies temporal state logic, and triggers non-blocking acoustic warnings.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        enable_audio: bool = True
    ) -> None:
        """Initialize pipeline subcomponents.

        Args:
            config: Master application configuration.
            enable_audio: Whether to activate acoustic alert system.
        """
        self.config = config or DEFAULT_CONFIG

        # Sub-modules
        self.landmark_detector = FaceMeshDetector(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False
        )
        self.head_pose_estimator = HeadPoseEstimator(config=self.config)
        self.baseline_detector = BaselineDetector(config=self.config)

        self.enable_audio = enable_audio
        self.alert_manager: Optional[AlertManager] = None
        if self.enable_audio and self.config.audio.enabled:
            self.alert_manager = AlertManager(config=self.config)

        logger.info("DrowsinessDetector pipeline initialized successfully.")

    def process_frame(self, frame: Optional[np.ndarray]) -> DetectionResult:
        """Execute drowsiness detection pipeline on an input video frame.

        Args:
            frame: BGR video frame as a NumPy ndarray.

        Returns:
            DetectionResult populated with all biometric signals and state.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to process_frame.")
            return DetectionResult(
                state=self.baseline_detector.current_state,
                face_detected=False,
                reasons=["Invalid or empty video frame"]
            )

        frame_shape = frame.shape[:2]

        # 1. Extract 468 MediaPipe FaceMesh landmarks
        face_landmarks: Optional[FaceLandmarks] = self.landmark_detector.detect(frame)

        if face_landmarks is None:
            # Missing face: update baseline with None to handle gracefully
            baseline_res = self.baseline_detector.update(None)
            return DetectionResult(
                state=baseline_res.state,
                perclos=baseline_res.perclos,
                face_detected=False,
                reasons=baseline_res.reasons
            )

        # 2. 3D Head Pose Estimation
        pose_res: HeadPoseResult = self.head_pose_estimator.estimate_pose(
            face_landmarks.pixel_landmarks,
            frame_shape=frame_shape
        )

        pitch = pose_res.pitch if pose_res.success else 0.0
        yaw = pose_res.yaw if pose_res.success else 0.0

        # 3. Classical Baseline Analysis & Temporal State Machine
        baseline_res = self.baseline_detector.update(
            face_landmarks.pixel_landmarks,
            pitch=pitch,
            yaw=yaw
        )

        # 4. Trigger Non-blocking Audio Alert on State Escalation
        if self.alert_manager is not None:
            self.alert_manager.trigger(baseline_res.state)

        return DetectionResult(
            state=baseline_res.state,
            left_ear=baseline_res.left_ear,
            right_ear=baseline_res.right_ear,
            avg_ear=baseline_res.avg_ear,
            mar=baseline_res.mar,
            perclos=baseline_res.perclos,
            eyes_closed=baseline_res.eyes_closed,
            continuous_closed_frames=baseline_res.continuous_closed_frames,
            drowsy_persistence_counter=baseline_res.drowsy_persistence_counter,
            alert_reset_counter=baseline_res.alert_reset_counter,
            is_yawning=baseline_res.is_yawning,
            is_nodding=baseline_res.is_nodding,
            is_distracted=baseline_res.is_distracted,
            reasons=baseline_res.reasons,
            face_detected=True,
            pixel_landmarks=face_landmarks.pixel_landmarks,
            normalized_landmarks=face_landmarks.normalized_landmarks,
            bbox=face_landmarks.bbox,
            head_pose=pose_res
        )

    def reset(self) -> None:
        """Reset temporal state counters and rolling buffers."""
        self.baseline_detector.reset()

    def close(self) -> None:
        """Cleanly release all pipeline resources."""
        self.landmark_detector.close()
        if self.alert_manager is not None:
            self.alert_manager.close()
            self.alert_manager = None
        logger.info("DrowsinessDetector cleanly closed.")

    def __enter__(self) -> "DrowsinessDetector":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
