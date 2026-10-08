"""Configuration module for Driver Drowsiness Detection System.

Defines all constants, landmark indices, project states, thresholds,
temporal logic parameters, display settings, and filesystem paths.
Strictly eliminates magic numbers and hardcoded parameters.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


class ProjectState(IntEnum):
    """Enumeration of project drowsiness states."""
    ALERT = 0
    LOW_VIGILANCE = 1
    DROWSY = 2
    MICROSLEEP = 3

    @property
    def label(self) -> str:
        """Return human-readable state label."""
        labels = {
            ProjectState.ALERT: "ALERT",
            ProjectState.LOW_VIGILANCE: "LOW VIGILANCE",
            ProjectState.DROWSY: "DROWSY",
            ProjectState.MICROSLEEP: "MICROSLEEP"
        }
        return labels[self]


@dataclass(frozen=True)
class LandmarkIndices:
    """MediaPipe FaceMesh landmark indices used for facial signal computation.

    Uses base 468 landmarks. Strictly adheres to project specification:
    Left eye: [33, 160, 158, 133, 153, 144]
    Right eye: [362, 385, 387, 263, 373, 380]
    Mouth: [61, 291, 39, 181, 0, 17, 269, 405]
    Head pose 3D correspondences: nose tip, chin, eye corners, mouth corners.
    """
    # Left eye landmarks: [p1_outer, p2_top1, p3_top2, p4_inner, p5_bottom2, p6_bottom1]
    LEFT_EYE: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)

    # Right eye landmarks: [p1_inner, p2_top1, p3_top2, p4_outer, p5_bottom2, p6_bottom1]
    RIGHT_EYE: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

    # Mouth landmarks: [p1_left, p2_right, p3_top_l, p4_bot_l, p5_top_mid, p6_bot_mid, p7_top_r, p8_bot_r]
    MOUTH: Tuple[int, ...] = (61, 291, 39, 181, 0, 17, 269, 405)

    # Head Pose Landmark Indices (OpenCV solvePnP 2D-to-3D correspondence)
    NOSE_TIP: int = 1
    CHIN: int = 152
    LEFT_EYE_CORNER: int = 33
    RIGHT_EYE_CORNER: int = 263
    LEFT_MOUTH_CORNER: int = 61
    RIGHT_MOUTH_CORNER: int = 291

    @property
    def head_pose_indices(self) -> Tuple[int, ...]:
        """Return tuple of landmark indices used for 3D head pose estimation."""
        return (
            self.NOSE_TIP,
            self.CHIN,
            self.LEFT_EYE_CORNER,
            self.RIGHT_EYE_CORNER,
            self.LEFT_MOUTH_CORNER,
            self.RIGHT_MOUTH_CORNER
        )


@dataclass
class ClassicalThresholds:
    """Configurable thresholds for classical biometric signals."""
    # Eye Aspect Ratio threshold (eyes closed when EAR < threshold)
    ear_threshold: float = 0.18

    # Mouth Aspect Ratio threshold (mouth open/yawning when MAR > threshold)
    mar_threshold: float = 0.60

    # PERCLOS rolling-window threshold (fatigued when eye-closure ratio > threshold)
    perclos_threshold: float = 0.25

    # Head pose deviation thresholds in degrees
    yaw_threshold_deg: float = 20.0
    pitch_threshold_deg: float = 15.0


@dataclass
class TemporalLogicConfig:
    """Temporal persistence and window parameters.

    Never hardcode these values; all state transition requirements
    are managed here.
    """
    # Drowsy persistence: frames of accumulated/sustained fatigue cues needed to trigger DROWSY
    drowsy_persistence: int = 60        #30

    # Microsleep persistence: consecutive frames of eye closure to trigger MICROSLEEP
    # Note: Initial engineering prototype threshold, subject to calibration during evaluation
    microsleep_persistence: int = 20    #10

    # Alert reset persistence: consecutive frames of normal alert signals required to recover to ALERT
    alert_reset_persistence: int = 30   #15

    # Rolling window size (in frames) for calculating PERCLOS (~3 seconds at 30 FPS)
    perclos_window_size: int = 180      #90

    # Normal blink duration filter: blinks lasting <= max_blink_frames are considered normal
    # and MUST NOT trigger LOW_VIGILANCE or DROWSY
    max_normal_blink_frames: int = 13   #4

    # Sustained yawn persistence: frames of mouth opening required to qualify as fatigue yawn
    yawn_persistence: int = 40          #15

    # Sustained head nod persistence: frames of head pitch down required to qualify as nod
    head_nod_persistence: int = 20      #10

    # Low vigilance persistence: frames of accumulated fatigue cues before triggering LOW_VIGILANCE
    low_vigilance_persistence: int = 20  #5


@dataclass
class AudioConfig:
    """Audio alert settings."""
    enabled: bool = True
    volume: float = 0.7
    cooldown_seconds: float = 2.0
    sample_rate: int = 22050

    # Synthesized beep frequencies (Hz) and durations (sec) per state
    # Form: state -> list of (frequency_hz, duration_sec) pulses
    synthetic_tones: Dict[ProjectState, list] = field(default_factory=lambda: {
        ProjectState.LOW_VIGILANCE: [(600, 0.15)],
        ProjectState.DROWSY: [(750, 0.2), (750, 0.2)],
        ProjectState.MICROSLEEP: [(1100, 0.15), (1100, 0.15), (1100, 0.25)]
    })


@dataclass
class VisualizerConfig:
    """OpenCV HUD Visualizer display configuration."""
    frame_width: int = 640
    frame_height: int = 480
    show_landmarks: bool = True
    show_head_pose_axes: bool = True
    show_telemetry_bars: bool = True

    # Color palette (BGR) for project states
    state_colors: Dict[ProjectState, Tuple[int, int, int]] = field(default_factory=lambda: {
        ProjectState.ALERT: (0, 220, 0),         # Vibrant Green
        ProjectState.LOW_VIGILANCE: (0, 215, 255), # Yellow
        ProjectState.DROWSY: (0, 128, 255),       # Orange
        ProjectState.MICROSLEEP: (0, 0, 255)      # Red
    })


FACE_ROI_MARGIN: float = 0.20


@dataclass
class DatasetConfig:
    """NTHU dataset paths, splitting parameters, and preprocessing settings."""
    raw_subdir: str = "Multi class"
    processed_subdir: str = "processed"
    splits_subdir: str = "splits"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    face_margin: float = FACE_ROI_MARGIN
    target_image_size: Tuple[int, int] = (224, 224)


@dataclass
class ViTConfig:
    """Vision Transformer (ViT) architecture and training hyperparameters."""
    model_name: str = "google/vit-base-patch16-224"
    num_classes: int = 4
    image_size: Tuple[int, int] = (224, 224)
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 32
    num_workers: int = 4
    stage1_lr: float = 1e-3
    stage1_epochs: int = 5
    stage2_lr: float = 2e-5
    stage2_epochs: int = 10
    unfreeze_blocks: int = 2
    weight_decay: float = 0.01


@dataclass
class FeatureFusionConfig:
    """Configuration for Phase 3 Multimodal Feature Fusion.

    Defines engineering starting weights and signal normalization bounds
    for combining continuous ViT probabilities with classical biometric signals.
    Note: These are initial engineering parameters, not scientifically validated constants.
    """
    # Top-level multimodal fusion weights (Initial engineering starting values; sum to 1.0)
    vit_weight: float = 0.60
    ear_weight: float = 0.20
    mar_weight: float = 0.10
    head_pose_weight: float = 0.10

    # Signal reference ranges for classical evidence normalization
    ear_closed_min: float = 0.10       # EAR representing completely closed eye
    ear_open_ref: float = 0.26         # EAR representing fully open alert eye
    mar_normal_ref: float = 0.30       # MAR representing normal closed mouth
    mar_yawn_max: float = 0.85         # MAR representing maximum yawn expansion
    perclos_saturation: float = 0.30   # PERCLOS level corresponding to 100% fatigue evidence
    pitch_nod_ref: float = 25.0        # Head pitch angle (deg) corresponding to 100% nod evidence
    yaw_distract_ref: float = 35.0     # Head yaw angle (deg) corresponding to 100% distraction evidence

    # Per-state classical evidence internal mixing weights
    # Alert state classical weights
    alert_ear_weight: float = 0.50
    alert_mar_weight: float = 0.30
    alert_pose_weight: float = 0.20

    # Low vigilance classical weights (yawn dominant, supported by distraction and mild eye closure)
    low_vigilance_mar_weight: float = 0.65
    low_vigilance_yaw_weight: float = 0.20
    low_vigilance_ear_weight: float = 0.15

    # Drowsy state classical weights (PERCLOS & head nod primary, supported by low EAR & yawn)
    drowsy_perclos_weight: float = 0.40
    drowsy_nod_weight: float = 0.30
    drowsy_ear_weight: float = 0.20
    drowsy_mar_weight: float = 0.10

    # Microsleep state classical weights (Eye closure heavily dominant, supported by PERCLOS & nod)
    microsleep_ear_weight: float = 0.70
    microsleep_perclos_weight: float = 0.20
    microsleep_nod_weight: float = 0.10


@dataclass
class TemporalDecisionConfig:
    """Configuration for Phase 3 Temporal Decision Layer.

    Defines threshold gates, score boundaries, and history buffer size
    for temporal state machine evidence accumulation.
    """
    # Evidence score thresholds to activate fatigue persistence accumulation
    drowsy_score_threshold: float = 0.45
    low_vigilance_score_threshold: float = 0.45
    alert_score_threshold: float = 0.50

    # Classical evidence gate thresholds (normalized in [0.0, 1.0])
    eye_closure_threshold: float = 0.50     # Normalized eye closure
    yawn_threshold: float = 0.50            # Normalized yawn evidence
    nod_threshold: float = 0.50             # Normalized pitch nod evidence
    distraction_threshold: float = 0.50     # Normalized yaw distraction evidence

    # Rolling diagnostic history capacity (frames)
    history_size: int = 180


@dataclass
class AppConfig:
    """Master application configuration coordinating all sub-configs."""
    landmarks: LandmarkIndices = field(default_factory=LandmarkIndices)
    thresholds: ClassicalThresholds = field(default_factory=ClassicalThresholds)
    temporal: TemporalLogicConfig = field(default_factory=TemporalLogicConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    fusion: FeatureFusionConfig = field(default_factory=FeatureFusionConfig)
    temporal_decision: TemporalDecisionConfig = field(default_factory=TemporalDecisionConfig)

    # Base paths using pathlib.Path
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def data_dir(self) -> Path:
        """Path to data directory."""
        return self.base_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Path to raw dataset directory."""
        return self.data_dir / "raw" / self.dataset.raw_subdir

    @property
    def processed_data_dir(self) -> Path:
        """Path to preprocessed facial crops."""
        return self.data_dir / self.dataset.processed_subdir

    @property
    def splits_dir(self) -> Path:
        """Path to dataset split manifests."""
        return self.data_dir / self.dataset.splits_subdir

    @property
    def models_dir(self) -> Path:
        """Path to models directory."""
        return self.base_dir / "models"

    @property
    def checkpoints_dir(self) -> Path:
        """Path to model checkpoints directory."""
        return self.models_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        """Path to log files directory."""
        return self.base_dir / "logs"


# Standard 3D Facial Model Points in World Coordinates (mm)
# Origin centered at nose tip (Landmark 1).
# [X: Right -> Left, Y: Down -> Up, Z: In -> Out]
CANONICAL_FACE_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (Landmark 1)
    [0.0, 330.0, -65.0],      # Chin (Landmark 152)
    [-225.0, -170.0, -135.0], # Left eye corner (Landmark 33)
    [225.0, -170.0, -135.0],  # Right eye corner (Landmark 263)
    [-150.0, 150.0, -125.0],  # Left mouth corner (Landmark 61)
    [150.0, 150.0, -125.0]    # Right mouth corner (Landmark 291)
], dtype=np.float64)


# Default global configuration instance
DEFAULT_CONFIG = AppConfig()
