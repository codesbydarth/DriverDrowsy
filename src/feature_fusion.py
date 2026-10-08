"""Multimodal Feature Fusion module for Phase 3.

Combines the continuous 4-class Vision Transformer (ViT) probability vector with
classical computer vision signals (EAR, MAR, PERCLOS, head pose) in a deterministic,
explainable, and computationally lightweight framework.

Architectural Invariants:
    - Never converts ViT probabilities to hard argmax for decision making.
    - Deterministic and explainable: exposes exact attribution for each state.
    - All weights and reference thresholds originate from utils.config.AppConfig.
    - Bounded outputs: all evidence and fused scores strictly lie in [0.0, 1.0].
    - Zero temporal state-machine logic in this module (managed in Step 3).
    - Zero audio/alert or GUI/HUD dependencies.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from utils.config import DEFAULT_CONFIG, AppConfig, FeatureFusionConfig, ProjectState
from utils.logger import setup_logger

logger = setup_logger("feature_fusion")


@dataclass(frozen=True)
class ClassicalEvidence:
    """Normalized evidence scores for individual classical modalities in [0.0, 1.0].

    Attributes:
        eye_closure: Evidence of eyelid closure derived from EAR (1.0 = fully closed).
        eye_openness: Evidence of alert eye openness derived from EAR (1.0 = fully open).
        yawn: Evidence of sustained mouth opening derived from MAR (1.0 = wide yawn).
        perclos: Evidence of cumulative ocular fatigue from PERCLOS buffer.
        head_nod: Evidence of downward head tilting from pitch angle.
        head_distraction: Evidence of lateral diversion from yaw angle.
    """
    eye_closure: float
    eye_openness: float
    yawn: float
    perclos: float
    head_nod: float
    head_distraction: float


@dataclass(frozen=True)
class StateExplanation:
    """Explainability container detailing why evidence accumulated for a state.

    Attributes:
        state: The canonical ProjectState being evaluated.
        total_score: Final fused score in [0.0, 1.0].
        vit_contribution: Absolute weighted score contributed by ViT probability.
        classical_contribution: Absolute weighted score contributed by classical metrics.
        component_breakdown: Named breakdown of individual modality contributions.
    """
    state: ProjectState
    total_score: float
    vit_contribution: float
    classical_contribution: float
    component_breakdown: Dict[str, float]


@dataclass(frozen=True)
class FusionResult:
    """Output container produced by the feature fusion layer for a single video frame.

    Attributes:
        vit_probabilities: Validated 4-class ViT softmax probability vector [ALERT, LOW, DROWSY, MICRO].
        classical_evidence: Individual modality evidence scores.
        state_scores: Dictionary mapping each ProjectState to its continuous fused score in [0.0, 1.0].
        dominant_state: Informational single-frame ranking indicating the state with the highest
                        instantaneous fused score. IMPORTANT: This is strictly an instantaneous diagnostic
                        indicator and must NOT be used directly as the final system state, nor used to
                        bypass temporal persistence or trigger alerts. Temporal decision logic belongs
                        exclusively to Step 3 (Temporal Decision Layer).
        confidence: Instantaneous fused score corresponding to dominant_state.
        explanations: Detailed attribution breakdown for each canonical state.
    """
    vit_probabilities: np.ndarray
    classical_evidence: ClassicalEvidence
    state_scores: Dict[ProjectState, float]
    dominant_state: ProjectState
    confidence: float
    explanations: Dict[ProjectState, StateExplanation]


class FeatureFusion:
    """Multimodal evidence fusion combining ViT probabilities and classical biometric cues."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize feature fusion layer with configuration.

        Args:
            config: Master application configuration instance. Defaults to DEFAULT_CONFIG.
        """
        self.config = config or DEFAULT_CONFIG
        self.fusion_config: FeatureFusionConfig = getattr(
            self.config, "fusion", FeatureFusionConfig()
        )

    def _validate_and_normalize_vit_probs(
        self,
        vit_probabilities: Union[np.ndarray, Sequence[float]]
    ) -> np.ndarray:
        """Validate input ViT probabilities and return normalized float32 array of shape (4,).

        Args:
            vit_probabilities: Array or sequence of 4 class probabilities.

        Returns:
            Validated NumPy array of shape (4,) summing to 1.0.

        Raises:
            ValueError: If array length != 4, contains non-finite values, negative values,
                        or sum deviates excessively from 1.0.
            TypeError: If input is None or cannot be parsed as a float array.
        """
        if vit_probabilities is None:
            raise TypeError("vit_probabilities cannot be None.")

        try:
            arr = np.asarray(vit_probabilities, dtype=np.float64)
        except (ValueError, TypeError) as err:
            raise TypeError(f"Could not convert vit_probabilities to numeric array: {err}") from err

        if arr.shape != (4,):
            raise ValueError(f"Expected ViT probability vector of shape (4,), got shape {arr.shape}")

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"ViT probabilities must be finite, got: {arr}")

        if np.any(arr < 0.0):
            raise ValueError(f"ViT probabilities cannot contain negative values, got: {arr}")

        prob_sum = float(np.sum(arr))
        if prob_sum <= 0.0:
            raise ValueError("Sum of ViT probabilities must be strictly positive.")

        if abs(prob_sum - 1.0) > 0.15:
            raise ValueError(
                f"Sum of ViT probabilities deviates excessively from 1.0 (sum={prob_sum:.4f})"
            )

        # Normalize to strictly sum to 1.0 to eliminate floating-point drift
        normalized = arr / prob_sum
        return normalized.astype(np.float32)

    def _extract_classical_evidence(
        self,
        ear: float,
        mar: float,
        perclos: float,
        pitch: float,
        yaw: float,
        roll: float
    ) -> ClassicalEvidence:
        """Normalize raw physical signals into bounded evidence scores in [0.0, 1.0].

        Args:
            ear: Eye Aspect Ratio float.
            mar: Mouth Aspect Ratio float.
            perclos: Rolling window PERCLOS percentage in [0.0, 1.0].
            pitch: Head pitch Euler angle in degrees (positive = nodding down).
            yaw: Head yaw Euler angle in degrees (lateral turn).
            roll: Head roll Euler angle in degrees.

        Returns:
            ClassicalEvidence container with normalized scores in [0.0, 1.0].

        Raises:
            ValueError: If inputs contain NaN or non-finite values.
        """
        raw_vals = [ear, mar, perclos, pitch, yaw, roll]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in raw_vals):
            raise ValueError(f"All classical signals must be finite numeric values, got: {raw_vals}")

        cfg = self.fusion_config

        # 1. EAR -> Eye Closure & Eye Openness Evidence
        ear_clamped = max(0.0, float(ear))
        if ear_clamped <= cfg.ear_closed_min:
            eye_closure = 1.0
        elif ear_clamped >= cfg.ear_open_ref:
            eye_closure = 0.0
        else:
            eye_closure = (cfg.ear_open_ref - ear_clamped) / (cfg.ear_open_ref - cfg.ear_closed_min)
        eye_closure = float(np.clip(eye_closure, 0.0, 1.0))
        eye_openness = float(1.0 - eye_closure)

        # 2. MAR -> Yawn Evidence
        mar_clamped = max(0.0, float(mar))
        if mar_clamped <= cfg.mar_normal_ref:
            yawn = 0.0
        elif mar_clamped >= cfg.mar_yawn_max:
            yawn = 1.0
        else:
            yawn = (mar_clamped - cfg.mar_normal_ref) / (cfg.mar_yawn_max - cfg.mar_normal_ref)
        yawn = float(np.clip(yawn, 0.0, 1.0))

        # 3. PERCLOS Evidence (proportional to saturation threshold)
        perclos_clamped = float(np.clip(perclos, 0.0, 1.0))
        perclos_evidence = float(np.clip(perclos_clamped / cfg.perclos_saturation, 0.0, 1.0))

        # 4. Head Pose -> Nod and Distraction Evidence
        # Head pitch nod is downward (positive); looking upward (negative) must not count as nodding
        downward_pitch = max(0.0, float(pitch))
        head_nod = float(np.clip(downward_pitch / cfg.pitch_nod_ref, 0.0, 1.0))

        abs_yaw = abs(float(yaw))
        head_distract = float(np.clip(abs_yaw / cfg.yaw_distract_ref, 0.0, 1.0))

        return ClassicalEvidence(
            eye_closure=eye_closure,
            eye_openness=eye_openness,
            yawn=yawn,
            perclos=perclos_evidence,
            head_nod=head_nod,
            head_distraction=head_distract,
        )

    def fuse(
        self,
        vit_probabilities: Union[np.ndarray, Sequence[float]],
        ear: float,
        mar: float,
        perclos: float,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        head_pose: Optional[Any] = None,
    ) -> FusionResult:
        """Fuse continuous ViT probabilities with classical signals for a single frame.

        Args:
            vit_probabilities: 4-class ViT softmax probability vector [ALERT, LOW, DROWSY, MICRO].
            ear: Current Eye Aspect Ratio float.
            mar: Current Mouth Aspect Ratio float.
            perclos: Current rolling-window PERCLOS ratio in [0.0, 1.0].
            pitch: Head pitch in degrees (used if head_pose is None).
            yaw: Head yaw in degrees (used if head_pose is None).
            roll: Head roll in degrees (used if head_pose is None).
            head_pose: Optional HeadPoseResult object. If provided and successful,
                       extracts pitch, yaw, and roll directly.

        Returns:
            FusionResult containing bounded state scores, dominant state, and explainability breakdown.
        """
        # 1. Validate ViT input
        probs = self._validate_and_normalize_vit_probs(vit_probabilities)

        # 2. Extract Head Pose values if HeadPoseResult provided
        if head_pose is not None:
            if getattr(head_pose, "success", False):
                pitch = float(getattr(head_pose, "pitch", 0.0))
                yaw = float(getattr(head_pose, "yaw", 0.0))
                roll = float(getattr(head_pose, "roll", 0.0))
            else:
                pitch, yaw, roll = 0.0, 0.0, 0.0

        # 3. Extract normalized classical modality evidence
        evidence = self._extract_classical_evidence(
            ear=ear, mar=mar, perclos=perclos, pitch=pitch, yaw=yaw, roll=roll
        )

        cfg = self.fusion_config
        w_vit = cfg.vit_weight
        w_classical = 1.0 - w_vit

        # 4. Compute State-Specific Classical Evidence Components
        # State: ALERT
        alert_pose_normal = max(0.0, 1.0 - max(evidence.head_nod, evidence.head_distraction))
        alert_yawn_normal = max(0.0, 1.0 - evidence.yawn)
        c_alert = (
            cfg.alert_ear_weight * evidence.eye_openness +
            cfg.alert_mar_weight * alert_yawn_normal +
            cfg.alert_pose_weight * alert_pose_normal
        ) / (cfg.alert_ear_weight + cfg.alert_mar_weight + cfg.alert_pose_weight)

        # State: LOW_VIGILANCE (Yawning dominant, supported by head distraction & mild closure)
        c_low = (
            cfg.low_vigilance_mar_weight * evidence.yawn +
            cfg.low_vigilance_yaw_weight * evidence.head_distraction +
            cfg.low_vigilance_ear_weight * evidence.eye_closure
        ) / (cfg.low_vigilance_mar_weight + cfg.low_vigilance_yaw_weight + cfg.low_vigilance_ear_weight)

        # State: DROWSY (PERCLOS & head nod primary, supported by eye closure & yawn)
        c_drowsy = (
            cfg.drowsy_perclos_weight * evidence.perclos +
            cfg.drowsy_nod_weight * evidence.head_nod +
            cfg.drowsy_ear_weight * evidence.eye_closure +
            cfg.drowsy_mar_weight * evidence.yawn
        ) / (
            cfg.drowsy_perclos_weight + cfg.drowsy_nod_weight +
            cfg.drowsy_ear_weight + cfg.drowsy_mar_weight
        )

        # State: MICROSLEEP (Eye closure heavily dominant, supported by PERCLOS & head nod)
        c_micro = (
            cfg.microsleep_ear_weight * evidence.eye_closure +
            cfg.microsleep_perclos_weight * evidence.perclos +
            cfg.microsleep_nod_weight * evidence.head_nod
        ) / (cfg.microsleep_ear_weight + cfg.microsleep_perclos_weight + cfg.microsleep_nod_weight)

        classical_map = {
            ProjectState.ALERT: float(np.clip(c_alert, 0.0, 1.0)),
            ProjectState.LOW_VIGILANCE: float(np.clip(c_low, 0.0, 1.0)),
            ProjectState.DROWSY: float(np.clip(c_drowsy, 0.0, 1.0)),
            ProjectState.MICROSLEEP: float(np.clip(c_micro, 0.0, 1.0)),
        }

        # 5. Fuse ViT probability + Classical evidence for each canonical state
        state_scores: Dict[ProjectState, float] = {}
        explanations: Dict[ProjectState, StateExplanation] = {}

        for state in ProjectState:
            state_idx = int(state)
            vit_prob = float(probs[state_idx])
            c_score = classical_map[state]

            vit_contrib = w_vit * vit_prob
            c_contrib = w_classical * c_score
            fused_score = float(np.clip(vit_contrib + c_contrib, 0.0, 1.0))
            state_scores[state] = fused_score

            # Detailed attribution breakdown for explainability
            if state == ProjectState.ALERT:
                w_alert_sum = cfg.alert_ear_weight + cfg.alert_mar_weight + cfg.alert_pose_weight
                alert_scale = w_classical / w_alert_sum if w_alert_sum > 0 else 0.0
                breakdown = {
                    "vit": vit_contrib,
                    "eye_openness": alert_scale * cfg.alert_ear_weight * evidence.eye_openness,
                    "mouth_normal": alert_scale * cfg.alert_mar_weight * alert_yawn_normal,
                    "head_normal": alert_scale * cfg.alert_pose_weight * alert_pose_normal,
                }
            elif state == ProjectState.LOW_VIGILANCE:
                w_low_sum = (
                    cfg.low_vigilance_mar_weight +
                    cfg.low_vigilance_yaw_weight +
                    cfg.low_vigilance_ear_weight
                )
                low_scale = w_classical / w_low_sum if w_low_sum > 0 else 0.0
                breakdown = {
                    "vit": vit_contrib,
                    "yawn": low_scale * cfg.low_vigilance_mar_weight * evidence.yawn,
                    "distraction": low_scale * cfg.low_vigilance_yaw_weight * evidence.head_distraction,
                    "eye_closure": low_scale * cfg.low_vigilance_ear_weight * evidence.eye_closure,
                }
            elif state == ProjectState.DROWSY:
                w_drowsy_sum = (
                    cfg.drowsy_perclos_weight +
                    cfg.drowsy_nod_weight +
                    cfg.drowsy_ear_weight +
                    cfg.drowsy_mar_weight
                )
                drowsy_scale = w_classical / w_drowsy_sum if w_drowsy_sum > 0 else 0.0
                breakdown = {
                    "vit": vit_contrib,
                    "perclos": drowsy_scale * cfg.drowsy_perclos_weight * evidence.perclos,
                    "head_nod": drowsy_scale * cfg.drowsy_nod_weight * evidence.head_nod,
                    "eye_closure": drowsy_scale * cfg.drowsy_ear_weight * evidence.eye_closure,
                    "yawn": drowsy_scale * cfg.drowsy_mar_weight * evidence.yawn,
                }
            else:  # MICROSLEEP
                w_micro_sum = (
                    cfg.microsleep_ear_weight +
                    cfg.microsleep_perclos_weight +
                    cfg.microsleep_nod_weight
                )
                micro_scale = w_classical / w_micro_sum if w_micro_sum > 0 else 0.0
                breakdown = {
                    "vit": vit_contrib,
                    "eye_closure": micro_scale * cfg.microsleep_ear_weight * evidence.eye_closure,
                    "perclos": micro_scale * cfg.microsleep_perclos_weight * evidence.perclos,
                    "head_nod": micro_scale * cfg.microsleep_nod_weight * evidence.head_nod,
                }

            explanations[state] = StateExplanation(
                state=state,
                total_score=fused_score,
                vit_contribution=vit_contrib,
                classical_contribution=c_contrib,
                component_breakdown=breakdown,
            )

        # Dominant state (instantaneous ranking; informational only.
        # MUST NOT bypass temporal persistence or trigger alerts directly;
        # temporal decision-making belongs exclusively to Step 3).
        dominant_state = max(state_scores, key=lambda s: state_scores[s])
        confidence = state_scores[dominant_state]

        return FusionResult(
            vit_probabilities=probs,
            classical_evidence=evidence,
            state_scores=state_scores,
            dominant_state=dominant_state,
            confidence=confidence,
            explanations=explanations,
        )
