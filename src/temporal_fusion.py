"""Temporal Decision Layer for Phase 3 Multimodal Drowsiness Detection.

Consumes frame-level FusionResult objects from FeatureFusion and manages:
    - Temporal persistence and evidence accumulation
    - State transition logic and priority arbitration
    - Hysteresis and gradual alert recovery
    - Microsleep, drowsiness, and low-vigilance temporal gating
    - Safe handling of missing face frames without false escalation
    - Optional non-blocking acoustic alert dispatch via AlertManager

Architectural Invariants:
    - Strictly deterministic and explainable.
    - Consumes pre-computed FusionResult objects; never re-runs landmark detection,
      biometric extraction, or neural network inference.
    - All persistence thresholds and score gates originate from utils.config.AppConfig.
    - Bounded memory usage via a fixed-capacity diagnostic history deque.
    - Zero blocking I/O or sleep calls.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from alerts.alert import AlertManager
from src.feature_fusion import FusionResult
from utils.config import (
    DEFAULT_CONFIG,
    AppConfig,
    ProjectState,
    TemporalDecisionConfig,
    TemporalLogicConfig,
)
from utils.logger import setup_logger

logger = setup_logger("temporal_fusion")


@dataclass(frozen=True)
class TemporalDecision:
    """Output container produced by the Temporal Decision Layer for a single time step.

    Attributes:
        current_state: Final resolved driver state after temporal logic.
        previous_state: State before processing the current time step.
        transition: True if state changed in this time step.
        transition_reason: Descriptive explanation of the transition or state maintenance.
        microsleep_progress: Ratio [0.0, 1.0] of progress toward microsleep persistence.
        drowsy_progress: Ratio [0.0, 1.0] of progress toward drowsy persistence.
        low_vigilance_progress: Ratio [0.0, 1.0] of progress toward low-vigilance persistence.
        alert_recovery_progress: Ratio [0.0, 1.0] of progress toward alert recovery persistence.
        continuous_closed_frames: Consecutive eye-closure frames.
        drowsy_persistence_counter: Accumulated fatigue persistence count.
        low_vigilance_counter: Accumulated low-vigilance persistence count.
        alert_reset_counter: Accumulated alert recovery frames.
        fusion_result: The single-frame FusionResult consumed, or None if no face.
    """
    current_state: ProjectState
    previous_state: ProjectState
    transition: bool
    transition_reason: str
    microsleep_progress: float
    drowsy_progress: float
    low_vigilance_progress: float
    alert_recovery_progress: float
    continuous_closed_frames: int
    drowsy_persistence_counter: int
    low_vigilance_counter: int
    alert_reset_counter: int
    fusion_result: Optional[FusionResult] = None


class TemporalDecisionLayer:
    """Stateful temporal evidence accumulator and decision layer for driver drowsiness."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        alert_manager: Optional[AlertManager] = None,
        enable_audio: bool = False,
    ) -> None:
        """Initialize the Temporal Decision Layer.

        Args:
            config: Master application configuration instance. Defaults to DEFAULT_CONFIG.
            alert_manager: Optional existing AlertManager instance. If None and enable_audio
                           is True, a new AlertManager will be initialized internally.
            enable_audio: Whether to initialize and trigger audio alerts on state transitions.
        """
        self.config = config or DEFAULT_CONFIG
        self.temporal_config: TemporalLogicConfig = self.config.temporal
        self.decision_config: TemporalDecisionConfig = getattr(
            self.config, "temporal_decision", TemporalDecisionConfig()
        )

        # Alert Manager integration
        if alert_manager is not None:
            self.alert_manager: Optional[AlertManager] = alert_manager
            self._owns_alert_manager: bool = False
        elif enable_audio:
            self.alert_manager = AlertManager(config=self.config)
            self._owns_alert_manager = True
        else:
            self.alert_manager = None
            self._owns_alert_manager = False

        # Active state
        self.current_state: ProjectState = ProjectState.ALERT

        # Temporal counters (strictly non-negative, bounded)
        self.continuous_closed_frames: int = 0
        self.drowsy_persistence_counter: int = 0
        self.low_vigilance_counter: int = 0
        self.alert_reset_counter: int = 0
        self.yawn_counter: int = 0
        self.nod_counter: int = 0

        # Bounded diagnostic history buffer (stores lightweight diagnostic tuples)
        self.history: Deque[Tuple[float, ProjectState, Optional[float]]] = deque(
            maxlen=self.decision_config.history_size
        )
        # Bounded buffer tracking recent head nod evidence for dynamic excursion detection
        self.nod_history: Deque[float] = deque(maxlen=30)

    def reset(self) -> None:
        """Reset all temporal counters, history buffers, and return to ALERT state."""
        self.current_state = ProjectState.ALERT
        self.continuous_closed_frames = 0
        self.drowsy_persistence_counter = 0
        self.low_vigilance_counter = 0
        self.alert_reset_counter = 0
        self.yawn_counter = 0
        self.nod_counter = 0
        self.history.clear()
        self.nod_history.clear()
        logger.info("TemporalDecisionLayer state and persistence counters reset to initial.")

    def update(self, fusion_result: Optional[FusionResult] = None) -> TemporalDecision:
        """Process a single frame's FusionResult and update temporal driver state.

        Args:
            fusion_result: Step 2 FusionResult containing multimodal state scores,
                           classical evidence, and ViT probabilities. If None,
                           indicates that no face was detected in this frame.

        Returns:
            TemporalDecision container detailing current state, transitions,
            progress meters, and causal reasons.
        """
        prev_state = self.current_state
        now_ts = time.time()

        # ---------------------------------------------------------
        # CASE 1: NO FACE DETECTED (Safety Critical Rule)
        # ---------------------------------------------------------
        # Absence of landmarks must NOT falsely escalate drowsiness,
        # trigger microsleep, or simulate closed eyes.
        if fusion_result is None:
            self.continuous_closed_frames = 0
            self.yawn_counter = 0
            self.nod_counter = 0
            # Gently decay or freeze fatigue counters without resetting alert recovery
            self.alert_reset_counter = 0

            reason = "No face detected: temporal state frozen safely without escalation"
            self.history.append((now_ts, self.current_state, None))

            return TemporalDecision(
                current_state=self.current_state,
                previous_state=prev_state,
                transition=False,
                transition_reason=reason,
                microsleep_progress=0.0,
                drowsy_progress=self._get_drowsy_progress(),
                low_vigilance_progress=self._get_low_vigilance_progress(),
                alert_recovery_progress=0.0,
                continuous_closed_frames=self.continuous_closed_frames,
                drowsy_persistence_counter=self.drowsy_persistence_counter,
                low_vigilance_counter=self.low_vigilance_counter,
                alert_reset_counter=self.alert_reset_counter,
                fusion_result=None,
            )

        # ---------------------------------------------------------
        # CASE 2: VALID FUSION RESULT (Extract Signals)
        # ---------------------------------------------------------
        ev = fusion_result.classical_evidence
        scores = fusion_result.state_scores
        cfg_t = self.temporal_config
        cfg_d = self.decision_config

        # 1. Eye Closure Tracking (Dominant Physical Cue for Microsleep)
        is_eye_closed = ev.eye_closure >= cfg_d.eye_closure_threshold
        if is_eye_closed:
            self.continuous_closed_frames += 1
        else:
            self.continuous_closed_frames = 0

        # 2. Yawning Persistence Tracking
        is_mouth_open = ev.yawn >= cfg_d.yawn_threshold
        if is_mouth_open:
            self.yawn_counter += 1
            is_yawning = self.yawn_counter >= cfg_t.yawn_persistence
        else:
            self.yawn_counter = 0
            is_yawning = False

        # 3. Head Nod Movement Tracking
        # Distinguish temporal head nod movement from static resting pitch angle.
        # A static posture (e.g. constant pitch +20 deg) must NOT accumulate nod persistence.
        self.nod_history.append(ev.head_nod)
        min_recent_nod = min(self.nod_history) if self.nod_history else ev.head_nod
        dynamic_nod_excursion = ev.head_nod - min_recent_nod
        # Nod movement requires dynamic downward excursion >= 0.25 OR severe slump >= 0.90
        is_nod_movement = (dynamic_nod_excursion >= 0.25) or (ev.head_nod >= 0.90)

        is_pitch_down = (ev.head_nod >= cfg_d.nod_threshold) and is_nod_movement
        if is_pitch_down:
            self.nod_counter += 1
            is_nodding = self.nod_counter >= cfg_t.head_nod_persistence
        else:
            self.nod_counter = max(0, self.nod_counter - 1)
            is_nodding = False

        # 4. Head Distraction Tracking
        is_distracted = ev.head_distraction >= cfg_d.distraction_threshold

        # 5. Drowsiness Cue Assessment
        # Cues include: continuous DROWSY score supported by classical physical cues,
        # or direct classical cues (prolonged eye closure, high PERCLOS, yawning, nodding).
        drowsy_score_active = scores.get(ProjectState.DROWSY, 0.0) >= cfg_d.drowsy_score_threshold
        prolonged_closure = self.continuous_closed_frames > cfg_t.max_normal_blink_frames
        perclos_active = ev.perclos >= 0.50

        # Physical classical fatigue presence check prevents false ViT probabilities
        # from accumulating persistence when all biometric signals indicate an alert driver.
        classical_fatigue_present = (
            prolonged_closure or
            perclos_active or
            is_yawning or
            is_nodding or
            ev.eye_closure >= cfg_d.eye_closure_threshold or
            ev.perclos >= 0.25 or
            ev.yawn >= cfg_d.yawn_threshold or
            (is_nod_movement and ev.head_nod >= cfg_d.nod_threshold)
        )

        drowsy_cues_active = (
            (drowsy_score_active and classical_fatigue_present) or
            prolonged_closure or
            perclos_active or
            is_yawning or
            is_nodding
        )

        if drowsy_cues_active:
            self.drowsy_persistence_counter = min(
                cfg_t.drowsy_persistence,
                self.drowsy_persistence_counter + 1
            )
            self.alert_reset_counter = 0
        else:
            self.drowsy_persistence_counter = max(0, self.drowsy_persistence_counter - 1)
            self.alert_reset_counter += 1

        # 6. Low Vigilance Cue Assessment
        # Cues include: continuous LOW_VIGILANCE score above gate, yawning, or distraction.
        # Conceptual decoupling: drowsy_persistence_counter never triggers low vigilance.
        low_vigilance_score_active = (
            scores.get(ProjectState.LOW_VIGILANCE, 0.0) >= cfg_d.low_vigilance_score_threshold
        )
        low_vigilance_cues_active = (
            low_vigilance_score_active or
            is_yawning or
            is_distracted
        )

        if low_vigilance_cues_active:
            self.low_vigilance_counter = min(
                cfg_t.low_vigilance_persistence,
                self.low_vigilance_counter + 1
            )
        else:
            self.low_vigilance_counter = max(0, self.low_vigilance_counter - 1)

        # ---------------------------------------------------------
        # STATE TRANSITION ARBITRATION (Priority Ordered)
        # ---------------------------------------------------------
        target_state = self.current_state
        reason = ""

        # RULE 1: MICROSLEEP (Highest Risk State)
        # Dominant trigger: eyes continuously closed >= microsleep_persistence.
        # Direct escalation from any state (including ALERT) is permitted for immediate safety.
        # Must NOT trigger solely from 1-frame ViT probability without physical closure persistence.
        if (
            self.continuous_closed_frames >= cfg_t.microsleep_persistence and
            ev.eye_closure >= cfg_d.eye_closure_threshold
        ):
            target_state = ProjectState.MICROSLEEP
            self.alert_reset_counter = 0
            reason = (
                f"Continuous eye closure ({self.continuous_closed_frames} frames >= "
                f"{cfg_t.microsleep_persistence}) with eye_closure evidence {ev.eye_closure:.2f}"
            )

        # RULE 2: DROWSY (Sustained Fatigue)
        # Triggered when accumulated drowsiness persistence satisfies requirement.
        elif self.drowsy_persistence_counter >= cfg_t.drowsy_persistence:
            target_state = ProjectState.DROWSY
            reason = (
                f"Drowsiness persisted for {self.drowsy_persistence_counter} frames >= "
                f"{cfg_t.drowsy_persistence}"
            )

        # RULE 3: LOW VIGILANCE (Warning Level)
        # Permitted to escalate from ALERT when warning persistence is satisfied.
        # Hysteresis rule: If current state is DROWSY or MICROSLEEP, do not downgrade
        # to LOW_VIGILANCE prematurely; require structured recovery.
        elif (
            self.low_vigilance_counter >= cfg_t.low_vigilance_persistence and
            self.current_state == ProjectState.ALERT
        ):
            target_state = ProjectState.LOW_VIGILANCE
            reason = (
                f"Low vigilance cues persisted for {self.low_vigilance_counter} frames >= "
                f"{cfg_t.low_vigilance_persistence}"
            )

        # RULE 4: ALERT RECOVERY (Hysteretic Gradual Recovery)
        # If the driver is in a fatigue state, sustained normal behavior is required
        # for alert_reset_persistence frames before recovering to ALERT.
        if self.alert_reset_counter >= cfg_t.alert_reset_persistence:
            if self.current_state != ProjectState.ALERT:
                target_state = ProjectState.ALERT
                reason = (
                    f"Driver recovered to ALERT after {self.alert_reset_counter} "
                    f"consecutive normal frames >= {cfg_t.alert_reset_persistence}"
                )
                self.drowsy_persistence_counter = 0
                self.low_vigilance_counter = 0

        # If no transition triggered, maintain state
        if not reason:
            reason = f"Maintaining current state {self.current_state.label}"

        # ---------------------------------------------------------
        # UPDATE STATE & TRIGGER ALERTS
        # ---------------------------------------------------------
        transition = target_state != self.current_state
        self.current_state = target_state

        # Log transition events
        if transition:
            logger.info(
                "Driver state transition: %s -> %s. Reason: %s",
                prev_state.label, self.current_state.label, reason
            )
            # Trigger audio alert non-blockingly on state escalation or transition
            if self.alert_manager is not None:
                self.alert_manager.trigger(self.current_state)

        # Record to history buffer
        dominant_score = scores.get(self.current_state, 0.0)
        self.history.append((now_ts, self.current_state, dominant_score))

        return TemporalDecision(
            current_state=self.current_state,
            previous_state=prev_state,
            transition=transition,
            transition_reason=reason,
            microsleep_progress=self._get_microsleep_progress(),
            drowsy_progress=self._get_drowsy_progress(),
            low_vigilance_progress=self._get_low_vigilance_progress(),
            alert_recovery_progress=self._get_alert_recovery_progress(),
            continuous_closed_frames=self.continuous_closed_frames,
            drowsy_persistence_counter=self.drowsy_persistence_counter,
            low_vigilance_counter=self.low_vigilance_counter,
            alert_reset_counter=self.alert_reset_counter,
            fusion_result=fusion_result,
        )

    def _get_microsleep_progress(self) -> float:
        """Calculate progress ratio in [0.0, 1.0] toward microsleep persistence."""
        target = self.temporal_config.microsleep_persistence
        if target <= 0:
            return 1.0
        return float(np.clip(self.continuous_closed_frames / target, 0.0, 1.0))

    def _get_drowsy_progress(self) -> float:
        """Calculate progress ratio in [0.0, 1.0] toward drowsy persistence."""
        target = self.temporal_config.drowsy_persistence
        if target <= 0:
            return 1.0
        return float(np.clip(self.drowsy_persistence_counter / target, 0.0, 1.0))

    def _get_low_vigilance_progress(self) -> float:
        """Calculate progress ratio in [0.0, 1.0] toward low vigilance persistence."""
        target = self.temporal_config.low_vigilance_persistence
        if target <= 0:
            return 1.0
        return float(np.clip(self.low_vigilance_counter / target, 0.0, 1.0))

    def _get_alert_recovery_progress(self) -> float:
        """Calculate progress ratio in [0.0, 1.0] toward alert recovery persistence."""
        target = self.temporal_config.alert_reset_persistence
        if target <= 0:
            return 1.0
        return float(np.clip(self.alert_reset_counter / target, 0.0, 1.0))

    def close(self) -> None:
        """Clean up resources, shutting down AlertManager if owned."""
        if self._owns_alert_manager and self.alert_manager is not None:
            self.alert_manager.close()
            self.alert_manager = None
        logger.info("TemporalDecisionLayer cleanly closed.")

    def __enter__(self) -> "TemporalDecisionLayer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
