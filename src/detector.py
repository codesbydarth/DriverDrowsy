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
from src.dataset import crop_face_roi
from src.feature_fusion import FeatureFusion, FusionResult
from src.head_pose import HeadPoseEstimator, HeadPoseResult
from src.landmark_detector import FaceLandmarks, FaceMeshDetector
from src.temporal_fusion import TemporalDecision, TemporalDecisionLayer
from src.vit_inference import ViTInferenceEngine
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
    fusion_result: Optional[FusionResult] = None
    temporal_decision: Optional[TemporalDecision] = None
    vit_probabilities: Optional[np.ndarray] = None


class DrowsinessDetector:
    """Core drowsiness detection pipeline orchestrator.

    Extracts facial landmarks via MediaPipe, estimates 3D head pose via solvePnP,
    evaluates EAR, MAR, and rolling PERCLOS against configurable thresholds,
    applies temporal state logic, and triggers non-blocking acoustic warnings.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        enable_audio: bool = True,
        vit_engine: Optional[ViTInferenceEngine] = None,
        enable_vit: bool = True,
    ) -> None:
        """Initialize pipeline subcomponents.

        Args:
            config: Master application configuration.
            enable_audio: Whether to activate acoustic alert system.
            vit_engine: Optional pre-loaded ViTInferenceEngine (e.g. for testing).
            enable_vit: Whether to load the ViT model checkpoint if vit_engine is None.
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

        # Phase 3 Feature Fusion & Temporal Decision Layer
        self.feature_fusion = FeatureFusion(config=self.config)
        self.temporal_decision_layer = TemporalDecisionLayer(
            config=self.config,
            alert_manager=self.alert_manager
        )

        # ViT Inference Engine (single instance, loaded once at startup)
        self.vit_engine: Optional[ViTInferenceEngine] = None
        if vit_engine is not None:
            self.vit_engine = vit_engine
        elif enable_vit:
            ckpt_path = self.config.checkpoints_dir / "vit_best_production.pt"
            if ckpt_path.exists():
                try:
                    self.vit_engine = ViTInferenceEngine(checkpoint_path=ckpt_path)
                    logger.info("ViTInferenceEngine production model loaded successfully.")
                except Exception as err:
                    logger.error("Failed to load ViT production checkpoint: %s", err)
                    self.vit_engine = None
            else:
                logger.warning(
                    "ViT production checkpoint not found at %s. Running in classical fusion mode.",
                    ckpt_path
                )

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
            temporal_dec = self.temporal_decision_layer.update(None)
            return DetectionResult(
                state=temporal_dec.current_state,
                face_detected=False,
                reasons=["Invalid or empty video frame"],
                temporal_decision=temporal_dec,
            )

        frame_shape = frame.shape[:2]

        # 1. Single FaceMesh pass per frame (ONE MediaPipe pass!)
        face_landmarks: Optional[FaceLandmarks] = self.landmark_detector.detect(frame)

        if face_landmarks is None:
            # Missing face: update baseline and temporal decision safely (no false escalation)
            baseline_res = self.baseline_detector.update(None)
            temporal_dec = self.temporal_decision_layer.update(None)
            active_reasons = temporal_dec.transition_reason if temporal_dec.transition_reason else ["No face detected"]
            return DetectionResult(
                state=temporal_dec.current_state,
                perclos=baseline_res.perclos,
                face_detected=False,
                reasons=[active_reasons] if isinstance(active_reasons, str) else active_reasons,
                temporal_decision=temporal_dec,
            )

        # 2. 3D Head Pose Estimation (solvePnP from FaceMesh landmarks)
        pose_res: HeadPoseResult = self.head_pose_estimator.estimate_pose(
            face_landmarks.pixel_landmarks,
            frame_shape=frame_shape
        )

        pitch = pose_res.pitch if pose_res.success else 0.0
        yaw = pose_res.yaw if pose_res.success else 0.0

        # 3. Classical Baseline Analysis (EAR, MAR, PERCLOS buffer)
        baseline_res = self.baseline_detector.update(
            face_landmarks.pixel_landmarks,
            pitch=pitch,
            yaw=yaw
        )

        # 4. Extract Face ROI for ViT (reusing FaceMesh landmarks & frame, NO second detector pass!)
        pil_crop, _ = crop_face_roi(
            frame,
            landmarks=face_landmarks.pixel_landmarks,
            bbox=face_landmarks.bbox,
            margin=self.config.dataset.face_margin,
            target_size=self.config.vit.image_size,
        )

        # 5. ViT Inference (continuous 4-class probability vector)
        if self.vit_engine is not None and self.vit_engine.is_ready:
            vit_probs = self.vit_engine.infer(pil_crop)
        else:
            vit_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        # 6. Multimodal Feature Fusion (Step 2)
        fusion_res = self.feature_fusion.fuse(
            vit_probabilities=vit_probs,
            ear=baseline_res.avg_ear,
            mar=baseline_res.mar,
            perclos=baseline_res.perclos,
            pitch=pitch,
            yaw=yaw,
            head_pose=pose_res,
        )

        # 7. Temporal Decision Layer (Step 3 - persistent across frames!)
        temporal_dec = self.temporal_decision_layer.update(fusion_res)

        # Final state comes STRICTLY from temporal_dec.current_state (NOT ViT argmax, NOT dominant_state)
        final_state = temporal_dec.current_state

        active_reasons = []
        if temporal_dec.transition_reason:
            active_reasons.append(temporal_dec.transition_reason)
        active_reasons.extend(baseline_res.reasons)

        return DetectionResult(
            state=final_state,
            left_ear=baseline_res.left_ear,
            right_ear=baseline_res.right_ear,
            avg_ear=baseline_res.avg_ear,
            mar=baseline_res.mar,
            perclos=baseline_res.perclos,
            eyes_closed=baseline_res.eyes_closed,
            continuous_closed_frames=temporal_dec.continuous_closed_frames,
            drowsy_persistence_counter=temporal_dec.drowsy_persistence_counter,
            alert_reset_counter=temporal_dec.alert_reset_counter,
            is_yawning=baseline_res.is_yawning,
            is_nodding=baseline_res.is_nodding,
            is_distracted=baseline_res.is_distracted,
            reasons=active_reasons,
            face_detected=True,
            pixel_landmarks=face_landmarks.pixel_landmarks,
            normalized_landmarks=face_landmarks.normalized_landmarks,
            bbox=face_landmarks.bbox,
            head_pose=pose_res,
            fusion_result=fusion_res,
            temporal_decision=temporal_dec,
            vit_probabilities=vit_probs,
        )

    def reset(self) -> None:
        """Reset temporal state counters and rolling buffers."""
        self.baseline_detector.reset()
        self.temporal_decision_layer.reset()

    def close(self) -> None:
        """Cleanly release all pipeline resources."""
        self.landmark_detector.close()
        self.temporal_decision_layer.close()
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
