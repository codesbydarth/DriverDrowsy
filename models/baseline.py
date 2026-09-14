"""Classical baseline drowsiness detection algorithms and temporal state machine.

Implements EAR, MAR, rolling-window PERCLOS, fatigue evidence evaluation,
and temporal persistence logic. Decoupled from GUI, webcam, and MediaPipe.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
import numpy as np

from utils.config import (
    DEFAULT_CONFIG,
    AppConfig,
    LandmarkIndices,
    ProjectState,
)
from utils.logger import setup_logger

logger = setup_logger("baseline")


def euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Compute 2D Euclidean distance between two points.

    Args:
        pt1: First point (x, y).
        pt2: Second point (x, y).

    Returns:
        Euclidean distance as float.
    """
    return float(np.linalg.norm(pt1[:2] - pt2[:2]))


def calculate_ear(
    landmarks: np.ndarray,
    eye_indices: Tuple[int, ...]
) -> float:
    """Compute Eye Aspect Ratio (EAR) from 6 eye landmarks.

    Uses Soukupová and Cech's formulation:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        landmarks: (N, 2) array of pixel coordinates.
        eye_indices: 6 landmark indices [p1, p2, p3, p4, p5, p6].

    Returns:
        Eye aspect ratio float value. Returns 0.0 on invalid input.
    """
    if landmarks is None or len(eye_indices) != 6:
        return 0.0

    try:
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        # Vertical distances
        v1 = euclidean_distance(p2, p6)
        v2 = euclidean_distance(p3, p5)

        # Horizontal distance
        h = euclidean_distance(p1, p4)

        if h < 1e-6:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)
    except Exception as err:
        logger.debug("Error computing EAR: %s", err)
        return 0.0


def calculate_mar(
    landmarks: np.ndarray,
    mouth_indices: Tuple[int, ...]
) -> float:
    """Compute Mouth Aspect Ratio (MAR) from 8 mouth landmarks.

    Formulation:
    MAR = (||p39 - p181|| + ||p0 - p17|| + ||p269 - p405||) / (2 * ||p61 - p291||)

    Args:
        landmarks: (N, 2) array of pixel coordinates.
        mouth_indices: 8 landmark indices [p61, p291, p39, p181, p0, p17, p269, p405].

    Returns:
        Mouth aspect ratio float value. Returns 0.0 on invalid input.
    """
    if landmarks is None or len(mouth_indices) != 8:
        return 0.0

    try:
        p_left = landmarks[mouth_indices[0]]    # 61
        p_right = landmarks[mouth_indices[1]]   # 291
        p_tl = landmarks[mouth_indices[2]]      # 39
        p_bl = landmarks[mouth_indices[3]]      # 181
        p_tm = landmarks[mouth_indices[4]]      # 0
        p_bm = landmarks[mouth_indices[5]]      # 17
        p_tr = landmarks[mouth_indices[6]]      # 269
        p_br = landmarks[mouth_indices[7]]      # 405

        v1 = euclidean_distance(p_tl, p_bl)
        v2 = euclidean_distance(p_tm, p_bm)
        v3 = euclidean_distance(p_tr, p_br)
        h = euclidean_distance(p_left, p_right)

        if h < 1e-6:
            return 0.0

        mar = (v1 + v2 + v3) / (2.0 * h)
        return float(mar)
    except Exception as err:
        logger.debug("Error computing MAR: %s", err)
        return 0.0


class PERCLOSBuffer:
    """Rolling-window buffer tracking eye-closure frequency over time.

    PERCLOS represents the proportion of time the eyes are closed
    within a fixed historical window of frames (e.g. 90 frames / 3 seconds).
    """

    def __init__(self, window_size: int = 90) -> None:
        """Initialize rolling window buffer.

        Args:
            window_size: Maximum capacity of the rolling window in frames.
        """
        self.window_size = max(1, window_size)
        self._buffer: Deque[bool] = deque(maxlen=self.window_size)

    def update(self, is_eye_closed: bool) -> float:
        """Append current frame eye status and return updated PERCLOS ratio.

        Args:
            is_eye_closed: True if eye closure detected in the current frame.

        Returns:
            PERCLOS ratio in [0.0, 1.0].
        """
        self._buffer.append(bool(is_eye_closed))
        return self.value

    @property
    def value(self) -> float:
        """Current PERCLOS ratio within the rolling window capacity."""
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / self.window_size

    def reset(self) -> None:
        """Clear the rolling buffer."""
        self._buffer.clear()

    @property
    def is_full(self) -> bool:
        """Return True if the rolling buffer has reached maximum capacity."""
        return len(self._buffer) >= self.window_size


@dataclass
class BaselineResult:
    """Comprehensive output of the classical baseline detection step.

    Attributes:
        state: ProjectState enum (ALERT, LOW_VIGILANCE, DROWSY, MICROSLEEP).
        left_ear: Left eye aspect ratio.
        right_ear: Right eye aspect ratio.
        avg_ear: Average eye aspect ratio.
        mar: Mouth aspect ratio.
        perclos: Rolling window PERCLOS value.
        eyes_closed: Boolean flag for current frame eye closure.
        continuous_closed_frames: Consecutive frames eyes have remained closed.
        drowsy_persistence_counter: Accumulated fatigue persistence count.
        alert_reset_counter: Consecutive alert frames towards state recovery.
        is_yawning: True if mouth opening exceeds persistence threshold.
        is_nodding: True if head pitch drop exceeds persistence threshold.
        is_distracted: True if head yaw deviation exceeds threshold.
        reasons: List of active trigger descriptions explaining current state.
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


class BaselineDetector:
    """Evaluates classical biometric signals using temporal state logic."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize baseline detector with thresholds and temporal configs.

        Args:
            config: Application configuration instance.
        """
        self.config = config or DEFAULT_CONFIG
        self.indices: LandmarkIndices = self.config.landmarks
        self.thresholds = self.config.thresholds
        self.temporal = self.config.temporal

        self.perclos_buffer = PERCLOSBuffer(window_size=self.temporal.perclos_window_size)

        # Internal temporal counters
        self.continuous_closed_frames: int = 0
        self.drowsy_persistence_counter: int = 0
        self.alert_reset_counter: int = 0
        self.yawn_counter: int = 0
        self.nod_counter: int = 0

        # Current system state
        self.current_state: ProjectState = ProjectState.ALERT

    def update(
        self,
        landmarks: Optional[np.ndarray],
        pitch: float = 0.0,
        yaw: float = 0.0
    ) -> BaselineResult:
        """Process facial landmarks and head pose to compute current state.

        Args:
            landmarks: (N, 2) pixel coordinates of face landmarks, or None.
            pitch: Head pitch in degrees (positive = nodding downward).
            yaw: Head yaw in degrees (lateral head turn).

        Returns:
            BaselineResult containing metrics and active state.
        """
        reasons: List[str] = []

        if landmarks is None:
            # Face missing: hold or reset eye counters gracefully
            self.continuous_closed_frames = 0
            self.perclos_buffer.update(False)
            return BaselineResult(
                state=self.current_state,
                perclos=self.perclos_buffer.value,
                reasons=["No face detected"]
            )

        # 1. Compute EAR for both eyes
        left_ear = calculate_ear(landmarks, self.indices.LEFT_EYE)
        right_ear = calculate_ear(landmarks, self.indices.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        # 2. Compute MAR
        mar = calculate_mar(landmarks, self.indices.MOUTH)

        # 3. Eye Closure Detection
        is_eye_closed = avg_ear < self.thresholds.ear_threshold

        # 4. Update PERCLOS rolling buffer
        perclos = self.perclos_buffer.update(is_eye_closed)

        # 5. Track continuous eye closure (for blink vs microsleep)
        if is_eye_closed:
            self.continuous_closed_frames += 1
        else:
            self.continuous_closed_frames = 0

        # 6. Yawning persistence (sustained mouth opening qualifies as yawn)
        if mar > self.thresholds.mar_threshold:
            self.yawn_counter += 1
            is_yawning = self.yawn_counter >= self.temporal.yawn_persistence
        else:
            self.yawn_counter = 0
            is_yawning = False

        # 7. Head nodding persistence (sustained head dip qualifies as nodding off)
        if abs(pitch) > self.thresholds.pitch_threshold_deg:
            self.nod_counter += 1
            is_nodding = self.nod_counter >= self.temporal.head_nod_persistence
        else:
            self.nod_counter = 0
            is_nodding = False

        # 8. Distraction indicator (head turned away)
        is_distracted = abs(yaw) > self.thresholds.yaw_threshold_deg

        # --- TEMPORAL DECISION LOGIC & STATE MACHINE ---

        # RULE A: MICROSLEEP (Highest priority)
        # Sustained abnormal eye closure: eyes continuously closed >= MICROSLEEP_PERSISTENCE
        # Note: 10 frames is an initial engineering prototype threshold
        if self.continuous_closed_frames >= self.temporal.microsleep_persistence:
            self.current_state = ProjectState.MICROSLEEP
            self.alert_reset_counter = 0
            reasons.append(
                f"Continuous eye closure ({self.continuous_closed_frames} frames >= "
                f"{self.temporal.microsleep_persistence})"
            )

        else:
            # Check for drowsiness cues (yawning, head nodding, or elevated PERCLOS)
            drowsy_trigger = False

            if is_yawning:
                drowsy_trigger = True
                reasons.append(f"Sustained yawning (MAR {mar:.2f} > {self.thresholds.mar_threshold:.2f})")

            if is_nodding:
                drowsy_trigger = True
                reasons.append(f"Head nodding (Pitch {pitch:.1f} deg > {self.thresholds.pitch_threshold_deg:.1f} deg)")

            # Elevated PERCLOS indicates slow eye closures / drooping lids
            if perclos > self.thresholds.perclos_threshold:
                drowsy_trigger = True
                reasons.append(
                    f"Elevated PERCLOS ({perclos * 100:.1f}% > "
                    f"{self.thresholds.perclos_threshold * 100:.1f}%)"
                )

            # Abnormal eye closure longer than a normal blink (> max_normal_blink_frames)
            # but less than microsleep
            if self.continuous_closed_frames > self.temporal.max_normal_blink_frames:
                drowsy_trigger = True
                reasons.append(
                    f"Prolonged eye closure ({self.continuous_closed_frames} frames > "
                    f"{self.temporal.max_normal_blink_frames})"
                )

            # Accumulate drowsy persistence
            if drowsy_trigger:
                self.drowsy_persistence_counter += 1
                self.alert_reset_counter = 0
            else:
                self.drowsy_persistence_counter = max(0, self.drowsy_persistence_counter - 1)
                self.alert_reset_counter += 1

            # RULE B: DROWSY
            # Triggered when accumulated drowsiness cues persist for >= DROWSY_PERSISTENCE frames
            if self.drowsy_persistence_counter >= self.temporal.drowsy_persistence:
                self.current_state = ProjectState.DROWSY
                reasons.append(
                    f"Drowsiness persisted for {self.drowsy_persistence_counter} frames"
                )

            # RULE C: LOW VIGILANCE
            # Fatigue signals present but haven't reached full DROWSY persistence yet,
            # or elevated PERCLOS, or head distraction.
            # Normal blinks (<= max_normal_blink_frames) MUST NOT trigger this!
            elif (
                drowsy_trigger
                or (self.drowsy_persistence_counter > self.temporal.low_vigilance_persistence)
                or is_distracted
            ):
                if self.current_state != ProjectState.DROWSY and self.current_state != ProjectState.MICROSLEEP:
                    self.current_state = ProjectState.LOW_VIGILANCE
                    if is_distracted:
                        reasons.append(f"Driver inattention / head turn ({yaw:.1f} deg)")

            # RULE D: ALERT RECOVERY
            # If all signals normal for >= alert_reset_persistence frames, recover to ALERT
            if self.alert_reset_counter >= self.temporal.alert_reset_persistence:
                if self.current_state != ProjectState.ALERT:
                    logger.info(
                        "Driver recovered to ALERT state after %d normal frames.",
                        self.alert_reset_counter
                    )
                self.current_state = ProjectState.ALERT
                self.drowsy_persistence_counter = 0

        return BaselineResult(
            state=self.current_state,
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            mar=mar,
            perclos=perclos,
            eyes_closed=is_eye_closed,
            continuous_closed_frames=self.continuous_closed_frames,
            drowsy_persistence_counter=self.drowsy_persistence_counter,
            alert_reset_counter=self.alert_reset_counter,
            is_yawning=is_yawning,
            is_nodding=is_nodding,
            is_distracted=is_distracted,
            reasons=reasons
        )

    def reset(self) -> None:
        """Reset all counters, buffer, and internal state back to initial."""
        self.continuous_closed_frames = 0
        self.drowsy_persistence_counter = 0
        self.alert_reset_counter = 0
        self.yawn_counter = 0
        self.nod_counter = 0
        self.current_state = ProjectState.ALERT
        self.perclos_buffer.reset()
        logger.info("BaselineDetector reset to initial ALERT state.")
