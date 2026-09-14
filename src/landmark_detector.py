"""Facial landmark detection module using MediaPipe FaceMesh.

Provides a robust, non-intrusive facial landmark detector extracting 468 base
landmarks. Strictly uses MediaPipe without any dlib dependency. Gracefully
handles missing faces, corrupted frames, and multiple faces.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("landmark_detector")


@dataclass
class FaceLandmarks:
    """Extracted facial landmark data for a single face.

    Attributes:
        pixel_landmarks: (468, 2) array of (x, y) coordinates in pixel space.
        normalized_landmarks: (468, 3) array of (x, y, z) normalized coordinates.
        bbox: Bounding box tuple (x_min, y_min, x_max, y_max) in pixel space.
        frame_shape: Dimensions of the analyzed frame (height, width).
    """
    pixel_landmarks: np.ndarray
    normalized_landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    frame_shape: Tuple[int, int]


class FaceMeshDetector:
    """MediaPipe FaceMesh detector wrapper with graceful failure handling.

    Tracks 468 base 3D facial landmarks without relying on dlib or external models.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ) -> None:
        """Initialize MediaPipe FaceMesh detector.

        Args:
            static_image_mode: Whether to treat input images as static or a video stream.
            max_num_faces: Maximum number of faces to detect (primary face tracked).
            refine_landmarks: Whether to refine with iris landmarks (False uses base 468).
            min_detection_confidence: Detection confidence threshold in [0.0, 1.0].
            min_tracking_confidence: Tracking confidence threshold in [0.0, 1.0].
        """
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize the internal MediaPipe FaceMesh instance."""
        try:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=self.static_image_mode,
                max_num_faces=self.max_num_faces,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            logger.info("MediaPipe FaceMesh detector initialized successfully.")
        except Exception as err:
            logger.error("Failed to initialize MediaPipe FaceMesh: %s", err)
            raise

    def detect(self, frame: Optional[np.ndarray]) -> Optional[FaceLandmarks]:
        """Detect facial landmarks in an input image frame.

        Args:
            frame: Input BGR image frame as a NumPy ndarray.

        Returns:
            FaceLandmarks object if a face is detected, otherwise None.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or invalid frame passed to FaceMeshDetector.")
            return None

        if self._face_mesh is None:
            logger.error("FaceMesh detector is not initialized.")
            return None

        height, width = frame.shape[:2]

        try:
            # MediaPipe expects RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make image writeable=False for performance optimization in MediaPipe
            rgb_frame.flags.writeable = False
            results = self._face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if not results.multi_face_landmarks:
                return None

            # Primary face (first detected face)
            raw_landmarks = results.multi_face_landmarks[0].landmark

            # Extract normalized coordinates (468, 3)
            num_landmarks = len(raw_landmarks)
            norm_coords = np.zeros((num_landmarks, 3), dtype=np.float32)
            pixel_coords = np.zeros((num_landmarks, 2), dtype=np.float32)

            for i, lm in enumerate(raw_landmarks):
                norm_coords[i] = [lm.x, lm.y, lm.z]
                pixel_coords[i] = [lm.x * width, lm.y * height]

            # Compute bounding box with clipping to frame bounds
            x_min = int(np.clip(np.min(pixel_coords[:, 0]), 0, width - 1))
            y_min = int(np.clip(np.min(pixel_coords[:, 1]), 0, height - 1))
            x_max = int(np.clip(np.max(pixel_coords[:, 0]), 0, width - 1))
            y_max = int(np.clip(np.max(pixel_coords[:, 1]), 0, height - 1))

            bbox = (x_min, y_min, x_max, y_max)

            return FaceLandmarks(
                pixel_landmarks=pixel_coords,
                normalized_landmarks=norm_coords,
                bbox=bbox,
                frame_shape=(height, width)
            )

        except Exception as err:
            logger.error("Error during landmark detection: %s", err)
            return None

    def close(self) -> None:
        """Release MediaPipe resources cleanly."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            logger.info("FaceMesh detector resources released.")

    def __enter__(self) -> "FaceMeshDetector":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
