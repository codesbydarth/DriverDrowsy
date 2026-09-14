"""OpenCV Heads-Up Display (HUD) visualizer module.

Renders real-time telemetry, state banners, metric bars (EAR, MAR, PERCLOS),
head pose vectors, persistence progress, and live FPS metrics onto video frames.
Operates independently from detector logic following a strict pipeline design:
Frame -> Detector -> DetectionResult -> Visualizer -> Annotated Frame.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from utils.config import DEFAULT_CONFIG, AppConfig, ProjectState
from utils.logger import setup_logger

logger = setup_logger("visualizer")


class HUDVisualizer:
    """Renders professional telemetry HUD and facial overlays onto OpenCV frames."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize visualizer with display configuration and color scheme.

        Args:
            config: Master application configuration instance.
        """
        self.config = config or DEFAULT_CONFIG
        self.colors = self.config.visualizer.state_colors
        self.thresholds = self.config.thresholds
        self.temporal = self.config.temporal

    def draw_hud(
        self,
        frame: np.ndarray,
        detection_result: any,
        fps: float = 0.0
    ) -> np.ndarray:
        """Render complete HUD overlay on top of the input video frame.

        Args:
            frame: Input BGR image frame (NumPy ndarray).
            detection_result: DetectionResult object containing signals and state.
            fps: Measured real-time frames per second.

        Returns:
            Annotated BGR frame with HUD overlays.
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        height, width = canvas.shape[:2]

        state: ProjectState = getattr(detection_result, "state", ProjectState.ALERT)
        theme_color = self.colors.get(state, (0, 255, 0))

        # 1. Draw Facial Mesh / Points if present
        pixel_landmarks = getattr(detection_result, "pixel_landmarks", None)
        if pixel_landmarks is not None and self.config.visualizer.show_landmarks:
            self._draw_landmarks(canvas, pixel_landmarks, theme_color)

        # 2. Draw Bounding Box if present
        bbox = getattr(detection_result, "bbox", None)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), theme_color, 2)

        # 3. Draw Head Pose Vector
        head_pose = getattr(detection_result, "head_pose", None)
        if head_pose is not None and head_pose.success and self.config.visualizer.show_head_pose_axes:
            self._draw_pose_vector(canvas, head_pose)

        # 4. Draw Header Banner (State & FPS)
        self._draw_header_banner(canvas, state, theme_color, fps, width)

        # 5. Draw Telemetry Dashboard Card (Top Left / Left Sidebar)
        self._draw_telemetry_card(canvas, detection_result, theme_color)

        # 6. Draw Persistence Progress Card (Bottom Center / Left)
        self._draw_persistence_card(canvas, detection_result, height)

        return canvas

    def _draw_landmarks(
        self,
        canvas: np.ndarray,
        landmarks: np.ndarray,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw key eye and mouth landmark points on the face."""
        indices = self.config.landmarks
        # Draw eyes
        for idx in indices.LEFT_EYE:
            pt = (int(landmarks[idx][0]), int(landmarks[idx][1]))
            cv2.circle(canvas, pt, 2, (0, 255, 255), -1)

        for idx in indices.RIGHT_EYE:
            pt = (int(landmarks[idx][0]), int(landmarks[idx][1]))
            cv2.circle(canvas, pt, 2, (0, 255, 255), -1)

        # Draw mouth
        for idx in indices.MOUTH:
            pt = (int(landmarks[idx][0]), int(landmarks[idx][1]))
            cv2.circle(canvas, pt, 2, (255, 128, 0), -1)

    def _draw_pose_vector(self, canvas: np.ndarray, head_pose: any) -> None:
        """Draw forward projection line indicating head gaze direction."""
        p1 = head_pose.nose_tip
        p2 = head_pose.nose_end_point2d
        # Draw arrowed line from nose tip outward
        cv2.arrowedLine(canvas, p1, p2, (255, 0, 0), 2, tipLength=0.2)

    def _draw_header_banner(
        self,
        canvas: np.ndarray,
        state: ProjectState,
        theme_color: Tuple[int, int, int],
        fps: float,
        width: int
    ) -> None:
        """Draw top banner with status label and measured FPS."""
        # Top panel background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, 42), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Status badge
        badge_text = f"STATE: {state.label}"
        cv2.putText(
            canvas, badge_text, (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, theme_color, 2, cv2.LINE_AA
        )

        # FPS indicator
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            canvas, fps_text, (width - 120, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA
        )

    def _draw_telemetry_card(
        self,
        canvas: np.ndarray,
        result: any,
        theme_color: Tuple[int, int, int]
    ) -> None:
        """Draw semi-transparent sidebar displaying EAR, MAR, PERCLOS, and angles."""
        card_w, card_h = 240, 190
        x0, y0 = 10, 50

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + card_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + card_w, y0 + card_h), theme_color, 1)

        avg_ear = getattr(result, "avg_ear", 0.0)
        mar = getattr(result, "mar", 0.0)
        perclos = getattr(result, "perclos", 0.0)
        head_pose = getattr(result, "head_pose", None)

        pitch = head_pose.pitch if head_pose and head_pose.success else 0.0
        yaw = head_pose.yaw if head_pose and head_pose.success else 0.0

        # EAR Metric Bar
        ear_status_color = (0, 255, 0) if avg_ear >= self.thresholds.ear_threshold else (0, 0, 255)
        self._draw_bar(
            canvas, x0 + 10, y0 + 30, 140, 12,
            val=avg_ear, max_val=0.5, thresh=self.thresholds.ear_threshold,
            label=f"EAR: {avg_ear:.2f}", color=ear_status_color
        )

        # MAR Metric Bar
        mar_status_color = (0, 0, 255) if mar > self.thresholds.mar_threshold else (0, 255, 0)
        self._draw_bar(
            canvas, x0 + 10, y0 + 75, 140, 12,
            val=mar, max_val=1.2, thresh=self.thresholds.mar_threshold,
            label=f"MAR: {mar:.2f}", color=mar_status_color
        )

        # PERCLOS Metric Bar
        perclos_status_color = (0, 0, 255) if perclos > self.thresholds.perclos_threshold else (0, 255, 0)
        self._draw_bar(
            canvas, x0 + 10, y0 + 120, 140, 12,
            val=perclos, max_val=0.5, thresh=self.thresholds.perclos_threshold,
            label=f"PERCLOS: {perclos * 100:.1f}%", color=perclos_status_color
        )

        # Head Pose Readouts
        pose_text = f"Pitch: {pitch:+.1f}   Yaw: {yaw:+.1f}"
        cv2.putText(
            canvas, pose_text, (x0 + 10, y0 + 165),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
        )

    def _draw_persistence_card(
        self,
        canvas: np.ndarray,
        result: any,
        height: int
    ) -> None:
        """Draw bottom status panel showing persistence counters and trigger reasons."""
        closed_frames = getattr(result, "continuous_closed_frames", 0)
        drowsy_frames = getattr(result, "drowsy_persistence_counter", 0)
        alert_reset = getattr(result, "alert_reset_counter", 0)
        reasons = getattr(result, "reasons", [])

        card_h = 50
        y0 = height - card_h - 10
        x0 = 10
        card_w = 420

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + card_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + card_w, y0 + card_h), (80, 80, 80), 1)

        # Counter strings
        micro_str = f"Eye Closed: {closed_frames}/{self.temporal.microsleep_persistence}f"
        drowsy_str = f"Drowsy: {drowsy_frames}/{self.temporal.drowsy_persistence}f"
        recover_str = f"Recovery: {alert_reset}/{self.temporal.alert_reset_persistence}f"
        stats_line = f"{micro_str} | {drowsy_str} | {recover_str}"

        cv2.putText(
            canvas, stats_line, (x0 + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA
        )

        reason_str = " | ".join(reasons) if reasons else "Normal driving behavior"
        # Truncate reason string if too long for card
        if len(reason_str) > 52:
            reason_str = reason_str[:49] + "..."

        cv2.putText(
            canvas, reason_str, (x0 + 8, y0 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA
        )

    def _draw_bar(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        val: float,
        max_val: float,
        thresh: float,
        label: str,
        color: Tuple[int, int, int]
    ) -> None:
        """Helper to draw a filled metric bar with a threshold indicator line."""
        # Label
        cv2.putText(
            canvas, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA
        )
        # Background bar
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (45, 45, 45), -1)

        # Value fill
        fill_ratio = float(np.clip(val / max_val, 0.0, 1.0))
        fill_w = int(w * fill_ratio)
        if fill_w > 0:
            cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), color, -1)

        # Border
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 1)

        # Threshold mark
        thresh_ratio = float(np.clip(thresh / max_val, 0.0, 1.0))
        thresh_x = int(x + w * thresh_ratio)
        cv2.line(canvas, (thresh_x, y - 2), (thresh_x, y + h + 2), (0, 255, 255), 2)
