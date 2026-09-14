"""Head pose estimation module using MediaPipe landmarks and OpenCV solvePnP.

Estimates 3D head rotation (pitch, yaw, roll) using perspective-n-point (PnP)
mapping between 2D MediaPipe facial landmarks and canonical 3D facial geometry.
Tracks driver distraction (yaw > 20 deg) and head dropping/nodding (pitch > 15 deg).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from utils.config import CANONICAL_FACE_3D_POINTS, DEFAULT_CONFIG, AppConfig
from utils.logger import setup_logger

logger = setup_logger("head_pose")


@dataclass
class HeadPoseResult:
    """Estimated 3D head pose metrics.

    Attributes:
        pitch: Up/down tilt in degrees (positive = tilted downward / nodding).
        yaw: Left/right turn in degrees (positive = turned right).
        roll: In-plane tilt in degrees.
        nose_tip: 2D pixel coordinates of the nose tip (origin).
        nose_end_point2d: Projected 2D pixel coordinates of forward gaze vector.
        success: True if solvePnP succeeded, False if estimation failed.
        is_nodding: True if pitch exceeds threshold (fatigue cue).
        is_distracted: True if yaw exceeds threshold (inattention cue).
    """
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    nose_tip: Tuple[int, int] = (0, 0)
    nose_end_point2d: Tuple[int, int] = (0, 0)
    success: bool = False
    is_nodding: bool = False
    is_distracted: bool = False


class HeadPoseEstimator:
    """Estimates head pose Euler angles from MediaPipe facial landmark points."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize head pose estimator with model points and thresholds.

        Args:
            config: Master application configuration instance.
        """
        self.config = config or DEFAULT_CONFIG
        self.model_points_3d = CANONICAL_FACE_3D_POINTS
        self.indices = self.config.landmarks.head_pose_indices
        self.yaw_threshold = self.config.thresholds.yaw_threshold_deg
        self.pitch_threshold = self.config.thresholds.pitch_threshold_deg

    def estimate_pose(
        self,
        pixel_landmarks: Optional[np.ndarray],
        frame_shape: Tuple[int, int]
    ) -> HeadPoseResult:
        """Estimate 3D head orientation angles from 2D pixel landmarks.

        Args:
            pixel_landmarks: (N, 2) array of landmark coordinates in pixels.
            frame_shape: (height, width) tuple of the video frame.

        Returns:
            HeadPoseResult with Euler angles, projection endpoints, and status.
        """
        if pixel_landmarks is None or len(pixel_landmarks) < 468:
            return HeadPoseResult(success=False)

        height, width = frame_shape

        try:
            # Extract 2D image coordinates corresponding to canonical 3D model points
            image_points_2d = np.array([
                pixel_landmarks[idx][:2] for idx in self.indices
            ], dtype=np.float64)

            # Camera matrix approximation based on frame dimensions
            focal_length = float(width)
            center = (width / 2.0, height / 2.0)
            camera_matrix = np.array([
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            # Assume zero lens distortion for standard webcam
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Solve Perspective-n-Point
            pnp_success, rvec, tvec = cv2.solvePnP(
                self.model_points_3d,
                image_points_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not pnp_success:
                logger.warning("cv2.solvePnP failed to converge.")
                return HeadPoseResult(success=False)

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # RQ decomposition to extract Euler angles in degrees
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            raw_pitch = float(angles[0])
            raw_yaw = float(angles[1])
            raw_roll = float(angles[2])

            # Normalize angles to intuitive coordinate frame:
            # Pitch: positive = tilting downward (head nod), negative = looking up
            # Yaw: absolute value for lateral distraction
            pitch = raw_pitch
            yaw = raw_yaw
            roll = raw_roll

            # Project 3D vector extending forward from the nose tip (Z = +500mm)
            nose_tip_2d = (int(image_points_2d[0][0]), int(image_points_2d[0][1]))
            axis_3d = np.array([[0.0, 0.0, 500.0]], dtype=np.float64)
            projected_2d, _ = cv2.projectPoints(
                axis_3d, rvec, tvec, camera_matrix, dist_coeffs
            )
            nose_end_point = (
                int(projected_2d[0][0][0]),
                int(projected_2d[0][0][1])
            )

            is_nodding = abs(pitch) > self.pitch_threshold
            is_distracted = abs(yaw) > self.yaw_threshold

            return HeadPoseResult(
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                nose_tip=nose_tip_2d,
                nose_end_point2d=nose_end_point,
                success=True,
                is_nodding=is_nodding,
                is_distracted=is_distracted
            )

        except Exception as err:
            logger.error("Error during head pose estimation: %s", err)
            return HeadPoseResult(success=False)
