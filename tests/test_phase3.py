"""Unit and integration tests for Phase 3 components.

Covers:
    - Step 1: ViT inference engine, probability vector contract, error handling,
      rejection of untrained fallback models, input preprocessing, and thread safety.
"""

import math
from pathlib import Path
import threading
from typing import Any, Optional
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from src.feature_fusion import (
    ClassicalEvidence,
    FeatureFusion,
    FusionResult,
    StateExplanation,
)
from src.temporal_fusion import TemporalDecision, TemporalDecisionLayer
from src.vit_inference import ViTInferenceEngine
from utils.config import (
    DEFAULT_CONFIG,
    AppConfig,
    FeatureFusionConfig,
    ProjectState,
    TemporalDecisionConfig,
    TemporalLogicConfig,
)


class MockViTModel(nn.Module):
    """Deterministic lightweight mock model simulating ViT 4-class output without loading 435MB weights."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        # Expand fixed logits to match batch size: (B, 4)
        return self.fixed_logits.unsqueeze(0).expand(batch_size, -1)


class TestStep1ViTInference:
    """Test suite for Phase 3 Step 1: ViTInferenceEngine."""

    def test_missing_checkpoint_raises_filenotfound(self, tmp_path: Path) -> None:
        """Verify that attempting to load a non-existent checkpoint raises FileNotFoundError and refuses fallback."""
        non_existent_ckpt = tmp_path / "non_existent_vit.pt"
        with pytest.raises(FileNotFoundError, match="untrained fallback models are strictly forbidden"):
            ViTInferenceEngine(checkpoint_path=non_existent_ckpt)

    def test_default_checkpoint_missing_fails_safely(self) -> None:
        """Verify that if the default production checkpoint does not exist, it fails safely without fallback."""
        default_path = Path("models/checkpoints/vit_best_production.pt")
        if not default_path.exists():
            with pytest.raises(FileNotFoundError, match="untrained fallback models are strictly forbidden"):
                ViTInferenceEngine(checkpoint_path=default_path)

    def test_mock_model_returns_valid_4class_probability_vector(self) -> None:
        """Verify that ViTInferenceEngine returns a valid (4,) continuous probability vector summing to 1.0."""
        # Logits: [0.0, 1.0, 4.0, 2.0]
        mock_logits = torch.tensor([0.0, 1.0, 4.0, 2.0], dtype=torch.float32)
        mock_model = MockViTModel(mock_logits)
        engine = ViTInferenceEngine(model=mock_model, device="cpu")

        assert engine.is_ready is True

        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        probs = engine.infer(dummy_img)

        # Check type, shape, and value constraints
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (4,)
        assert probs.dtype == np.float32
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert pytest.approx(float(np.sum(probs)), 1e-4) == 1.0

        # Verify probabilities are continuous evidence and NOT hard argmax
        assert not np.array_equal(probs, np.array([0.0, 0.0, 1.0, 0.0]))
        # Expected highest probability corresponds to index 2 (DROWSY)
        expected_highest_idx = int(ProjectState.DROWSY)
        assert np.argmax(probs) == expected_highest_idx
        # Probability ranking matches logits: DROWSY > MICROSLEEP > LOW_VIGILANCE > ALERT
        assert probs[2] > probs[3] > probs[1] > probs[0]

    def test_infer_handles_pil_and_numpy_bgr(self) -> None:
        """Verify inference accepts both PIL.Image and BGR/RGB NumPy arrays seamlessly."""
        mock_logits = torch.tensor([2.0, 1.0, 0.5, 0.1], dtype=torch.float32)
        mock_model = MockViTModel(mock_logits)
        engine = ViTInferenceEngine(model=mock_model, device="cpu")

        # 1. PIL Image input
        pil_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        probs_pil = engine.infer(pil_img)
        assert probs_pil.shape == (4,)

        # 2. NumPy array (RGB)
        np_rgb = np.full((224, 224, 3), fill_value=128, dtype=np.uint8)
        probs_rgb = engine.infer(np_rgb, is_bgr=False)
        assert probs_rgb.shape == (4,)

        # 3. NumPy array (BGR)
        np_bgr = np.full((224, 224, 3), fill_value=128, dtype=np.uint8)
        probs_bgr = engine.infer(np_bgr, is_bgr=True)
        assert probs_bgr.shape == (4,)

    def test_invalid_input_handling(self) -> None:
        """Verify inference rejects invalid image inputs with informative exceptions."""
        mock_logits = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        mock_model = MockViTModel(mock_logits)
        engine = ViTInferenceEngine(model=mock_model, device="cpu")

        with pytest.raises(ValueError, match="Input face_roi cannot be None"):
            engine.infer(None)

        with pytest.raises(ValueError, match="Input face_roi is an empty NumPy array"):
            engine.infer(np.array([], dtype=np.uint8))

        with pytest.raises(TypeError, match="Expected np.ndarray or PIL.Image"):
            engine.infer("not_an_image")

    def test_inference_thread_safety(self) -> None:
        """Verify that concurrent calls to infer from multiple threads succeed without race conditions."""
        mock_logits = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        mock_model = MockViTModel(mock_logits)
        engine = ViTInferenceEngine(model=mock_model, device="cpu")

        results = []
        errors = []

        def worker(thread_idx: int) -> None:
            try:
                img = np.full((224, 224, 3), fill_value=(thread_idx * 20) % 255, dtype=np.uint8)
                probs = engine.infer(img)
                results.append((thread_idx, probs))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors encountered: {errors}"
        assert len(results) == 10
        for _, probs in results:
            assert probs.shape == (4,)
            assert pytest.approx(float(np.sum(probs)), 1e-4) == 1.0


class DummyHeadPose:
    """Mock HeadPoseResult for testing head pose object integration."""

    def __init__(self, pitch: float, yaw: float, roll: float, success: bool = True) -> None:
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.success = success


class TestStep2FeatureFusion:
    """Test suite for Phase 3 Step 2: FeatureFusion module."""

    @pytest.fixture
    def fusion_engine(self) -> FeatureFusion:
        """Fixture providing initialized FeatureFusion instance with default config."""
        return FeatureFusion()

    def test_valid_fusion_result_structure(self, fusion_engine: FeatureFusion) -> None:
        """Verify that valid inputs produce a complete, structured FusionResult."""
        vit_probs = [0.70, 0.15, 0.10, 0.05]
        res = fusion_engine.fuse(
            vit_probabilities=vit_probs,
            ear=0.30,
            mar=0.25,
            perclos=0.05,
            pitch=2.0,
            yaw=1.0,
        )

        assert isinstance(res, FusionResult)
        assert res.vit_probabilities.shape == (4,)
        assert isinstance(res.classical_evidence, ClassicalEvidence)
        assert len(res.state_scores) == 4
        for st in ProjectState:
            assert st in res.state_scores
            score = res.state_scores[st]
            assert 0.0 <= score <= 1.0
            assert st in res.explanations
            expl = res.explanations[st]
            assert isinstance(expl, StateExplanation)
            assert expl.total_score == score

        assert res.dominant_state == ProjectState.ALERT
        assert res.confidence == res.state_scores[ProjectState.ALERT]

    def test_invalid_vit_vector_length(self, fusion_engine: FeatureFusion) -> None:
        """Verify that ViT probability vectors with length != 4 raise ValueError."""
        with pytest.raises(ValueError, match="Expected ViT probability vector of shape \\(4,\\)"):
            fusion_engine.fuse([0.5, 0.5], ear=0.3, mar=0.2, perclos=0.0)

        with pytest.raises(ValueError, match="Expected ViT probability vector of shape \\(4,\\)"):
            fusion_engine.fuse([0.2, 0.2, 0.2, 0.2, 0.2], ear=0.3, mar=0.2, perclos=0.0)

    def test_invalid_negative_probabilities(self, fusion_engine: FeatureFusion) -> None:
        """Verify that negative probabilities raise ValueError."""
        with pytest.raises(ValueError, match="cannot contain negative values"):
            fusion_engine.fuse([-0.1, 0.5, 0.3, 0.3], ear=0.3, mar=0.2, perclos=0.0)

    def test_invalid_non_finite_probabilities(self, fusion_engine: FeatureFusion) -> None:
        """Verify that NaN or Inf probabilities raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            fusion_engine.fuse([float("nan"), 0.5, 0.2, 0.3], ear=0.3, mar=0.2, perclos=0.0)

    def test_probability_normalization_behavior(self, fusion_engine: FeatureFusion) -> None:
        """Verify slight float drift in sum is normalized smoothly, and large drift raises ValueError."""
        # Minor drift (sum = 1.02)
        res = fusion_engine.fuse([0.27, 0.25, 0.25, 0.25], ear=0.3, mar=0.2, perclos=0.0)
        assert pytest.approx(float(np.sum(res.vit_probabilities)), 1e-4) == 1.0

        # Excessive drift (sum = 1.30)
        with pytest.raises(ValueError, match="deviates excessively from 1.0"):
            fusion_engine.fuse([0.4, 0.3, 0.3, 0.3], ear=0.3, mar=0.2, perclos=0.0)

    def test_ear_evidence_directionality(self, fusion_engine: FeatureFusion) -> None:
        """Verify lower EAR yields higher eye_closure and lower eye_openness."""
        ev_closed = fusion_engine._extract_classical_evidence(ear=0.05, mar=0.2, perclos=0.0, pitch=0.0, yaw=0.0, roll=0.0)
        ev_mid = fusion_engine._extract_classical_evidence(ear=0.20, mar=0.2, perclos=0.0, pitch=0.0, yaw=0.0, roll=0.0)
        ev_open = fusion_engine._extract_classical_evidence(ear=0.35, mar=0.2, perclos=0.0, pitch=0.0, yaw=0.0, roll=0.0)

        assert ev_closed.eye_closure == 1.0
        assert ev_closed.eye_openness == 0.0
        assert 0.0 < ev_mid.eye_closure < 1.0
        assert ev_open.eye_closure == 0.0
        assert ev_open.eye_openness == 1.0
        assert ev_closed.eye_closure > ev_mid.eye_closure > ev_open.eye_closure

    def test_mar_evidence_directionality(self, fusion_engine: FeatureFusion) -> None:
        """Verify higher MAR yields higher yawn evidence."""
        ev_normal = fusion_engine._extract_classical_evidence(ear=0.3, mar=0.25, perclos=0.0, pitch=0.0, yaw=0.0, roll=0.0)
        ev_opening = fusion_engine._extract_classical_evidence(ear=0.3, mar=0.55, perclos=0.0, pitch=0.0, yaw=0.0, roll=0.0)
        ev_wide = fusion_engine._extract_classical_evidence(ear=0.3, mar=0.90, perclos=0.0, pitch=0.0, yaw=0.0, roll=0.0)

        assert ev_normal.yawn == 0.0
        assert 0.0 < ev_opening.yawn < 1.0
        assert ev_wide.yawn == 1.0
        assert ev_wide.yawn > ev_opening.yawn > ev_normal.yawn

    def test_perclos_evidence_directionality(self, fusion_engine: FeatureFusion) -> None:
        """Verify higher PERCLOS yields higher perclos evidence."""
        ev_zero = fusion_engine._extract_classical_evidence(ear=0.3, mar=0.2, perclos=0.0, pitch=0.0, yaw=0.0, roll=0.0)
        ev_mid = fusion_engine._extract_classical_evidence(ear=0.3, mar=0.2, perclos=0.15, pitch=0.0, yaw=0.0, roll=0.0)
        ev_high = fusion_engine._extract_classical_evidence(ear=0.3, mar=0.2, perclos=0.35, pitch=0.0, yaw=0.0, roll=0.0)

        assert ev_zero.perclos == 0.0
        assert pytest.approx(ev_mid.perclos, 1e-2) == 0.50
        assert ev_high.perclos == 1.0
        assert ev_high.perclos > ev_mid.perclos > ev_zero.perclos

    def test_head_pose_evidence_directionality(self, fusion_engine: FeatureFusion) -> None:
        """Verify pitch increases head_nod and yaw increases head_distraction."""
        ev_neutral = fusion_engine._extract_classical_evidence(ear=0.3, mar=0.2, perclos=0.0, pitch=0.0, yaw=0.0, roll=0.0)
        ev_nod = fusion_engine._extract_classical_evidence(ear=0.3, mar=0.2, perclos=0.0, pitch=20.0, yaw=0.0, roll=0.0)
        ev_distract = fusion_engine._extract_classical_evidence(ear=0.3, mar=0.2, perclos=0.0, pitch=0.0, yaw=30.0, roll=0.0)

        assert ev_neutral.head_nod == 0.0
        assert ev_nod.head_nod > 0.0
        assert ev_distract.head_distraction > 0.0

    def test_weighted_fusion_calculation(self, fusion_engine: FeatureFusion) -> None:
        """Verify exact arithmetic combination of ViT probability and classical components."""
        cfg = fusion_engine.fusion_config
        vit_probs = [0.10, 0.20, 0.30, 0.40]
        res = fusion_engine.fuse(vit_probs, ear=0.05, mar=0.20, perclos=0.30, pitch=25.0)

        # In this input, eye_closure = 1.0, perclos = 1.0, head_nod = 1.0
        # Microsleep classical evidence = 1.0
        expected_micro = cfg.vit_weight * 0.40 + (1.0 - cfg.vit_weight) * 1.0
        assert pytest.approx(res.state_scores[ProjectState.MICROSLEEP], 1e-4) == expected_micro

        # Explainability breakdown matches
        expl_micro = res.explanations[ProjectState.MICROSLEEP]
        assert pytest.approx(expl_micro.vit_contribution, 1e-4) == cfg.vit_weight * 0.40
        assert pytest.approx(expl_micro.classical_contribution, 1e-4) == (1.0 - cfg.vit_weight) * 1.0

    def test_microsleep_evidence_increases_with_low_ear(self, fusion_engine: FeatureFusion) -> None:
        """Verify low EAR produces high MICROSLEEP score and isolated yawning does not trigger microsleep."""
        neutral_probs = [0.25, 0.25, 0.25, 0.25]
        # Closed eyes
        res_closed = fusion_engine.fuse(neutral_probs, ear=0.06, mar=0.20, perclos=0.0)
        # Open eyes but wide yawn
        res_yawn = fusion_engine.fuse(neutral_probs, ear=0.35, mar=0.85, perclos=0.0)

        # Closed eyes must produce substantially higher microsleep score than wide yawn
        assert res_closed.state_scores[ProjectState.MICROSLEEP] > res_yawn.state_scores[ProjectState.MICROSLEEP]
        # Wide yawn must NOT trigger MICROSLEEP as dominant
        assert res_yawn.dominant_state != ProjectState.MICROSLEEP

    def test_drowsy_evidence_increases_with_elevated_perclos(self, fusion_engine: FeatureFusion) -> None:
        """Verify elevated PERCLOS and head nodding substantially elevate DROWSY score."""
        neutral_probs = [0.25, 0.25, 0.25, 0.25]
        res_alert = fusion_engine.fuse(neutral_probs, ear=0.30, mar=0.20, perclos=0.0, pitch=0.0)
        res_drowsy = fusion_engine.fuse(neutral_probs, ear=0.22, mar=0.20, perclos=0.30, pitch=20.0)

        assert res_drowsy.state_scores[ProjectState.DROWSY] > res_alert.state_scores[ProjectState.DROWSY]

    def test_low_vigilance_evidence_increases_with_sustained_mar(self, fusion_engine: FeatureFusion) -> None:
        """Verify high MAR elevates LOW_VIGILANCE score and dominates when supported by ViT."""
        neutral_probs = [0.25, 0.25, 0.25, 0.25]
        res_normal = fusion_engine.fuse(neutral_probs, ear=0.30, mar=0.20, perclos=0.0)
        res_yawn = fusion_engine.fuse(neutral_probs, ear=0.30, mar=0.80, perclos=0.0)

        assert res_yawn.state_scores[ProjectState.LOW_VIGILANCE] > res_normal.state_scores[ProjectState.LOW_VIGILANCE]

        # With supportive ViT evidence (e.g. visual yawning/distraction detected), LOW_VIGILANCE is dominant
        yawn_probs = [0.15, 0.55, 0.15, 0.15]
        res_yawn_supported = fusion_engine.fuse(yawn_probs, ear=0.30, mar=0.80, perclos=0.0)
        assert res_yawn_supported.dominant_state == ProjectState.LOW_VIGILANCE

    def test_vit_probabilities_contribute_continuously_not_argmax(self, fusion_engine: FeatureFusion) -> None:
        """Verify ViT probabilities act as continuous evidence and are not collapsed to hard argmax."""
        # Hold classical signals constant
        ear, mar, perclos = 0.30, 0.20, 0.05
        drowsy_scores = []
        p_values = [0.10, 0.30, 0.50, 0.70, 0.90]

        for p in p_values:
            remaining = (1.0 - p) / 3.0
            probs = [remaining, remaining, p, remaining]
            res = fusion_engine.fuse(probs, ear=ear, mar=mar, perclos=perclos)
            drowsy_scores.append(res.state_scores[ProjectState.DROWSY])

        # Fused score must increase strictly monotonically with continuous ViT probability
        for i in range(len(drowsy_scores) - 1):
            assert drowsy_scores[i + 1] > drowsy_scores[i]

    def test_boundary_values_remain_bounded(self, fusion_engine: FeatureFusion) -> None:
        """Verify extreme boundary values produce finite, strictly bounded scores in [0.0, 1.0]."""
        extreme_cases = [
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (1.5, 2.5, 1.0, 90.0, 90.0),
            (0.0, 2.5, 1.0, -90.0, -90.0),
        ]
        vit_probs = [0.25, 0.25, 0.25, 0.25]

        for ear, mar, perclos, pitch, yaw in extreme_cases:
            res = fusion_engine.fuse(vit_probs, ear=ear, mar=mar, perclos=perclos, pitch=pitch, yaw=yaw)
            for st, score in res.state_scores.items():
                assert 0.0 <= score <= 1.0
                assert math.isfinite(score)

    def test_invalid_classical_inputs(self, fusion_engine: FeatureFusion) -> None:
        """Verify NaN or Inf inputs raise ValueError."""
        vit_probs = [0.25, 0.25, 0.25, 0.25]
        with pytest.raises(ValueError, match="must be finite numeric values"):
            fusion_engine.fuse(vit_probs, ear=float("nan"), mar=0.2, perclos=0.0)

        with pytest.raises(ValueError, match="must be finite numeric values"):
            fusion_engine.fuse(vit_probs, ear=0.3, mar=float("inf"), perclos=0.0)

    def test_output_structure_and_explainability(self, fusion_engine: FeatureFusion) -> None:
        """Verify explanations contain all component attributions and sum exactly."""
        res = fusion_engine.fuse([0.4, 0.3, 0.2, 0.1], ear=0.25, mar=0.35, perclos=0.10)
        for st in ProjectState:
            expl = res.explanations[st]
            assert "vit" in expl.component_breakdown
            assert expl.vit_contribution > 0.0
            assert expl.classical_contribution >= 0.0
            assert pytest.approx(expl.vit_contribution + expl.classical_contribution, 1e-4) == expl.total_score
            # Sum of all individual subcomponents in breakdown matches total_score exactly
            sum_breakdown = sum(expl.component_breakdown.values())
            assert pytest.approx(sum_breakdown, 1e-4) == expl.total_score

    def test_dominant_state_is_informational_and_stateless(self, fusion_engine: FeatureFusion) -> None:
        """Verify dominant_state is strictly instantaneous without persistence, hysteresis, or alert side-effects."""
        # 1. Verify engine has no temporal state attributes or alert managers
        assert not hasattr(fusion_engine, "counter")
        assert not hasattr(fusion_engine, "persistence")
        assert not hasattr(fusion_engine, "alert_manager")
        assert not hasattr(fusion_engine, "current_state")

        # 2. Alternating inputs produce instantaneous dominant_state without latching or memory
        alert_probs = [0.80, 0.10, 0.05, 0.05]
        drowsy_probs = [0.05, 0.05, 0.80, 0.10]

        r1 = fusion_engine.fuse(alert_probs, ear=0.32, mar=0.20, perclos=0.0)
        assert r1.dominant_state == ProjectState.ALERT

        r2 = fusion_engine.fuse(drowsy_probs, ear=0.12, mar=0.20, perclos=0.30)
        assert r2.dominant_state == ProjectState.DROWSY

        # Immediate return to alert frame instantly produces ALERT (no recovery delay or hysteresis in Step 2)
        r3 = fusion_engine.fuse(alert_probs, ear=0.32, mar=0.20, perclos=0.0)
        assert r3.dominant_state == ProjectState.ALERT

    def test_deterministic_repeated_calls(self, fusion_engine: FeatureFusion) -> None:
        """Verify identical inputs produce identical results."""
        vit_probs = [0.55, 0.20, 0.15, 0.10]
        res1 = fusion_engine.fuse(vit_probs, ear=0.28, mar=0.32, perclos=0.08, pitch=5.0, yaw=-10.0)
        res2 = fusion_engine.fuse(vit_probs, ear=0.28, mar=0.32, perclos=0.08, pitch=5.0, yaw=-10.0)

        for st in ProjectState:
            assert res1.state_scores[st] == res2.state_scores[st]
        assert res1.dominant_state == res2.dominant_state
        assert res1.confidence == res2.confidence

    def test_head_pose_object_support(self, fusion_engine: FeatureFusion) -> None:
        """Verify HeadPoseResult object support works seamlessly."""
        vit_probs = [0.25, 0.25, 0.25, 0.25]
        hp_obj = DummyHeadPose(pitch=18.0, yaw=-22.0, roll=5.0, success=True)
        res_obj = fusion_engine.fuse(vit_probs, ear=0.25, mar=0.25, perclos=0.05, head_pose=hp_obj)
        res_manual = fusion_engine.fuse(vit_probs, ear=0.25, mar=0.25, perclos=0.05, pitch=18.0, yaw=-22.0, roll=5.0)

        for st in ProjectState:
            assert pytest.approx(res_obj.state_scores[st], 1e-5) == res_manual.state_scores[st]

        # Failed head pose object safely falls back to neutral angles
        hp_failed = DummyHeadPose(pitch=50.0, yaw=50.0, roll=50.0, success=False)
        res_failed = fusion_engine.fuse(vit_probs, ear=0.25, mar=0.25, perclos=0.05, head_pose=hp_failed)
        res_neutral = fusion_engine.fuse(vit_probs, ear=0.25, mar=0.25, perclos=0.05, pitch=0.0, yaw=0.0, roll=0.0)

        for st in ProjectState:
            assert pytest.approx(res_failed.state_scores[st], 1e-5) == res_neutral.state_scores[st]


def make_mock_fusion_result(
    dominant: ProjectState = ProjectState.ALERT,
    eye_closure: float = 0.0,
    yawn: float = 0.0,
    perclos: Optional[float] = None,
    head_nod: float = 0.0,
    head_distraction: float = 0.0,
    drowsy_score: float = 0.05,
    microsleep_score: float = 0.05,
    low_vigilance_score: float = 0.05,
    alert_score: float = 0.85,
) -> FusionResult:
    """Helper creating a lightweight deterministic mock FusionResult for Step 3 testing."""
    if perclos is None:
        perclos = 0.60 if dominant == ProjectState.DROWSY else 0.0

    ev = ClassicalEvidence(
        eye_closure=eye_closure,
        eye_openness=float(max(0.0, 1.0 - eye_closure)),
        yawn=yawn,
        perclos=perclos,
        head_nod=head_nod,
        head_distraction=head_distraction,
    )
    scores = {
        ProjectState.ALERT: alert_score,
        ProjectState.LOW_VIGILANCE: low_vigilance_score,
        ProjectState.DROWSY: drowsy_score,
        ProjectState.MICROSLEEP: microsleep_score,
    }
    raw_probs = np.array([alert_score, low_vigilance_score, drowsy_score, microsleep_score], dtype=np.float32)
    p_sum = float(np.sum(raw_probs))
    probs = raw_probs / p_sum if p_sum > 0 else np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    explanations = {
        st: StateExplanation(
            state=st,
            total_score=scores[st],
            vit_contribution=scores[st] * 0.60,
            classical_contribution=scores[st] * 0.40,
            component_breakdown={"vit": scores[st] * 0.60},
        )
        for st in ProjectState
    }
    return FusionResult(
        vit_probabilities=probs,
        classical_evidence=ev,
        state_scores=scores,
        dominant_state=dominant,
        confidence=scores[dominant],
        explanations=explanations,
    )


class TestStep3TemporalDecisionLayer:
    """Test suite for Phase 3 Step 3: TemporalDecisionLayer."""

    @pytest.fixture
    def layer(self) -> TemporalDecisionLayer:
        """Fixture providing initialized TemporalDecisionLayer."""
        return TemporalDecisionLayer()

    def test_1_initial_state_is_alert(self, layer: TemporalDecisionLayer) -> None:
        """Verify initial state is ALERT and all counters start at zero."""
        assert layer.current_state == ProjectState.ALERT
        assert layer.continuous_closed_frames == 0
        assert layer.drowsy_persistence_counter == 0
        assert layer.low_vigilance_counter == 0
        assert layer.alert_reset_counter == 0

    def test_2_single_noisy_drowsy_frame_does_not_trigger_drowsy(self, layer: TemporalDecisionLayer) -> None:
        """Verify single noisy drowsy frame does not trigger state change."""
        drowsy_frame = make_mock_fusion_result(
            dominant=ProjectState.DROWSY,
            drowsy_score=0.85,
            perclos=0.8,
            alert_score=0.05
        )
        decision = layer.update(drowsy_frame)
        assert decision.current_state == ProjectState.ALERT
        assert decision.transition is False
        assert decision.drowsy_persistence_counter == 1
        assert decision.drowsy_progress < 1.0

    def test_3_drowsy_triggers_after_required_persistence(self, layer: TemporalDecisionLayer) -> None:
        """Verify DROWSY triggers exactly after required persistence frames (60 frames)."""
        drowsy_frame = make_mock_fusion_result(
            dominant=ProjectState.DROWSY,
            drowsy_score=0.85,
            perclos=0.8,
            alert_score=0.05
        )
        persistence = layer.temporal_config.drowsy_persistence  # 60
        low_vig_persistence = layer.temporal_config.low_vigilance_persistence  # 20

        # Frames 1 to 19: Not enough persistence even for LOW_VIGILANCE -> remains ALERT
        for _ in range(low_vig_persistence - 1):
            d = layer.update(drowsy_frame)
            assert d.current_state == ProjectState.ALERT
            assert d.transition is False

        # Frames 20 to 59: Progresses through LOW_VIGILANCE warning, but NOT yet DROWSY
        for _ in range(persistence - low_vig_persistence):
            d = layer.update(drowsy_frame)
            assert d.current_state != ProjectState.DROWSY

        # Frame 60: Drowsiness persistence fully satisfied -> escalates to DROWSY
        d = layer.update(drowsy_frame)
        assert d.current_state == ProjectState.DROWSY
        assert d.transition is True
        assert pytest.approx(d.drowsy_progress, 1e-4) == 1.0

    def test_4_drowsy_counter_decays_when_fatigue_cues_stop(self, layer: TemporalDecisionLayer) -> None:
        """Verify drowsy persistence counter decays gracefully when normal cues return."""
        drowsy_frame = make_mock_fusion_result(dominant=ProjectState.DROWSY, drowsy_score=0.85, alert_score=0.05)
        alert_frame = make_mock_fusion_result(dominant=ProjectState.ALERT, alert_score=0.90)

        for _ in range(15):
            layer.update(drowsy_frame)
        assert layer.drowsy_persistence_counter == 15

        for _ in range(5):
            layer.update(alert_frame)
        assert layer.drowsy_persistence_counter == 10

    def test_5_single_low_ear_frame_does_not_trigger_microsleep(self, layer: TemporalDecisionLayer) -> None:
        """Verify single low-EAR frame does not trigger MICROSLEEP."""
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        decision = layer.update(closed_frame)
        assert decision.current_state == ProjectState.ALERT
        assert decision.transition is False
        assert decision.continuous_closed_frames == 1
        assert decision.microsleep_progress < 1.0

    def test_6_microsleep_triggers_after_required_persistence(self, layer: TemporalDecisionLayer) -> None:
        """Verify MICROSLEEP triggers after continuous closed frames reach persistence."""
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        persistence = layer.temporal_config.microsleep_persistence  # 20
        for _ in range(persistence - 1):
            d = layer.update(closed_frame)
            assert d.current_state == ProjectState.ALERT

        d = layer.update(closed_frame)
        assert d.current_state == ProjectState.MICROSLEEP
        assert d.transition is True
        assert pytest.approx(d.microsleep_progress, 1e-4) == 1.0

    def test_7_microsleep_priority_over_lower_risk_states(self, layer: TemporalDecisionLayer) -> None:
        """Verify MICROSLEEP has higher priority and escalates from DROWSY."""
        drowsy_frame = make_mock_fusion_result(dominant=ProjectState.DROWSY, drowsy_score=0.85, alert_score=0.05)
        for _ in range(layer.temporal_config.drowsy_persistence):
            layer.update(drowsy_frame)
        assert layer.current_state == ProjectState.DROWSY

        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        for _ in range(layer.temporal_config.microsleep_persistence):
            d = layer.update(closed_frame)

        assert d.current_state == ProjectState.MICROSLEEP
        assert d.previous_state == ProjectState.DROWSY

    def test_8_low_vigilance_requires_persistence(self, layer: TemporalDecisionLayer) -> None:
        """Verify LOW_VIGILANCE requires sustained warning persistence."""
        yawn_frame = make_mock_fusion_result(
            dominant=ProjectState.LOW_VIGILANCE,
            yawn=0.80,
            low_vigilance_score=0.80,
            alert_score=0.10
        )
        persistence = layer.temporal_config.low_vigilance_persistence  # 20
        for _ in range(persistence - 1):
            d = layer.update(yawn_frame)
            assert d.current_state == ProjectState.ALERT

        d = layer.update(yawn_frame)
        assert d.current_state == ProjectState.LOW_VIGILANCE
        assert d.transition is True

    def test_9_single_yawn_frame_does_not_trigger_low_vigilance(self, layer: TemporalDecisionLayer) -> None:
        """Verify single isolated yawn does not cause LOW_VIGILANCE."""
        yawn_frame = make_mock_fusion_result(
            dominant=ProjectState.LOW_VIGILANCE,
            yawn=1.0,
            low_vigilance_score=0.80,
            alert_score=0.10
        )
        decision = layer.update(yawn_frame)
        assert decision.current_state == ProjectState.ALERT
        assert decision.transition is False

    def test_10_alert_recovery_requires_configured_normal_frames(self, layer: TemporalDecisionLayer) -> None:
        """Verify recovery to ALERT requires alert_reset_persistence frames."""
        drowsy_frame = make_mock_fusion_result(dominant=ProjectState.DROWSY, drowsy_score=0.85, alert_score=0.05)
        for _ in range(layer.temporal_config.drowsy_persistence):
            layer.update(drowsy_frame)
        assert layer.current_state == ProjectState.DROWSY

        alert_frame = make_mock_fusion_result(dominant=ProjectState.ALERT, alert_score=0.90)
        reset_req = layer.temporal_config.alert_reset_persistence  # 30

        for _ in range(reset_req - 1):
            d = layer.update(alert_frame)
            assert d.current_state == ProjectState.DROWSY
            assert d.alert_recovery_progress < 1.0

        d = layer.update(alert_frame)
        assert d.current_state == ProjectState.ALERT
        assert d.transition is True
        assert pytest.approx(d.alert_recovery_progress, 1e-4) == 1.0

    def test_11_fatigue_state_does_not_immediately_recover_from_one_normal_frame(
        self,
        layer: TemporalDecisionLayer
    ) -> None:
        """Verify single normal frame does not cause immediate recovery."""
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        for _ in range(layer.temporal_config.microsleep_persistence):
            layer.update(closed_frame)
        assert layer.current_state == ProjectState.MICROSLEEP

        alert_frame = make_mock_fusion_result(dominant=ProjectState.ALERT, alert_score=0.90)
        d = layer.update(alert_frame)
        assert d.current_state != ProjectState.ALERT

    def test_12_hysteresis_prevents_rapid_oscillation(self, layer: TemporalDecisionLayer) -> None:
        """Verify state does not flicker between ALERT and DROWSY on alternating frames."""
        drowsy_frame = make_mock_fusion_result(dominant=ProjectState.DROWSY, drowsy_score=0.85, alert_score=0.05)
        alert_frame = make_mock_fusion_result(dominant=ProjectState.ALERT, alert_score=0.90)

        for _ in range(layer.temporal_config.drowsy_persistence):
            layer.update(drowsy_frame)
        assert layer.current_state == ProjectState.DROWSY

        states = []
        for _ in range(5):
            states.append(layer.update(alert_frame).current_state)
            states.append(layer.update(alert_frame).current_state)
            states.append(layer.update(drowsy_frame).current_state)
            states.append(layer.update(drowsy_frame).current_state)

        # Must maintain steady DROWSY state throughout fluctuations
        assert all(st == ProjectState.DROWSY for st in states)

    def test_13_no_face_input_does_not_escalate_to_fatigue(self, layer: TemporalDecisionLayer) -> None:
        """Verify missing face frames do not falsely escalate fatigue or simulate closed eyes."""
        for _ in range(50):
            d = layer.update(None)
            assert d.current_state == ProjectState.ALERT
            assert d.continuous_closed_frames == 0
            assert d.drowsy_persistence_counter == 0

    def test_14_missing_fusion_result_handles_safely_when_in_fatigue(self, layer: TemporalDecisionLayer) -> None:
        """Verify missing face during fatigue safely freezes state without crashing or escalating."""
        drowsy_frame = make_mock_fusion_result(dominant=ProjectState.DROWSY, drowsy_score=0.85, alert_score=0.05)
        for _ in range(layer.temporal_config.drowsy_persistence):
            layer.update(drowsy_frame)
        assert layer.current_state == ProjectState.DROWSY

        for _ in range(10):
            d = layer.update(None)
            assert d.current_state == ProjectState.DROWSY
            assert d.transition is False
            assert "No face detected" in d.transition_reason

    def test_15_counters_reset_correctly_after_evidence_disappears(self, layer: TemporalDecisionLayer) -> None:
        """Verify reset() method clears all counters and returns to initial state."""
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        for _ in range(layer.temporal_config.microsleep_persistence):
            layer.update(closed_frame)
        assert layer.current_state == ProjectState.MICROSLEEP

        layer.reset()
        assert layer.current_state == ProjectState.ALERT
        assert layer.continuous_closed_frames == 0
        assert layer.drowsy_persistence_counter == 0
        assert layer.low_vigilance_counter == 0
        assert layer.alert_reset_counter == 0
        assert len(layer.history) == 0

    def test_16_counters_remain_bounded(self, layer: TemporalDecisionLayer) -> None:
        """Verify persistence counters are strictly bounded and do not grow indefinitely."""
        drowsy_frame = make_mock_fusion_result(dominant=ProjectState.DROWSY, drowsy_score=0.85, alert_score=0.05)
        for _ in range(150):
            layer.update(drowsy_frame)

        # Capped at drowsy_persistence
        assert layer.drowsy_persistence_counter == layer.temporal_config.drowsy_persistence

        alert_frame = make_mock_fusion_result(dominant=ProjectState.ALERT, alert_score=0.90)
        for _ in range(100):
            layer.update(alert_frame)

        # Bounded at 0 (never negative)
        assert layer.drowsy_persistence_counter == 0

    def test_17_state_transitions_are_deterministic(self) -> None:
        """Verify identical inputs to independent instances produce identical outputs."""
        layer1 = TemporalDecisionLayer()
        layer2 = TemporalDecisionLayer()

        drowsy_frame = make_mock_fusion_result(dominant=ProjectState.DROWSY, drowsy_score=0.85, alert_score=0.05)
        alert_frame = make_mock_fusion_result(dominant=ProjectState.ALERT, alert_score=0.90)
        seq = [drowsy_frame] * 35 + [alert_frame] * 5 + [drowsy_frame] * 35

        res1 = [layer1.update(f) for f in seq]
        res2 = [layer2.update(f) for f in seq]

        for d1, d2 in zip(res1, res2):
            assert d1.current_state == d2.current_state
            assert d1.transition == d2.transition
            assert d1.drowsy_persistence_counter == d2.drowsy_persistence_counter

    def test_18_previous_current_state_reporting(self, layer: TemporalDecisionLayer) -> None:
        """Verify previous_state and transition flags accurately reflect state change."""
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        for _ in range(layer.temporal_config.microsleep_persistence - 1):
            layer.update(closed_frame)

        d_trans = layer.update(closed_frame)
        assert d_trans.previous_state == ProjectState.ALERT
        assert d_trans.current_state == ProjectState.MICROSLEEP
        assert d_trans.transition is True

        d_steady = layer.update(closed_frame)
        assert d_steady.previous_state == ProjectState.MICROSLEEP
        assert d_steady.current_state == ProjectState.MICROSLEEP
        assert d_steady.transition is False

    def test_19_transition_reason_is_available(self, layer: TemporalDecisionLayer) -> None:
        """Verify transition_reason contains informative explanation."""
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        for _ in range(layer.temporal_config.microsleep_persistence - 1):
            layer.update(closed_frame)

        d = layer.update(closed_frame)
        assert "Continuous eye closure" in d.transition_reason
        assert str(layer.temporal_config.microsleep_persistence) in d.transition_reason

    def test_20_alert_integration_remains_non_blocking(self) -> None:
        """Verify alert dispatch is non-blocking and notifies mock alert manager."""
        class MockAlert:
            def __init__(self) -> None:
                self.calls = []

            def trigger(self, st: ProjectState) -> None:
                self.calls.append(st)

            def close(self) -> None:
                pass

        mock_alert = MockAlert()
        layer = TemporalDecisionLayer(alert_manager=mock_alert)

        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        for _ in range(layer.temporal_config.microsleep_persistence):
            layer.update(closed_frame)

        assert ProjectState.MICROSLEEP in mock_alert.calls

    def test_21_repeated_identical_input_deterministic_progression(self, layer: TemporalDecisionLayer) -> None:
        """Verify step-by-step progress accumulates monotonically."""
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        for step in range(1, layer.temporal_config.microsleep_persistence + 1):
            d = layer.update(closed_frame)
            assert d.continuous_closed_frames == step

    def test_22_direct_microsleep_escalation_from_alert(self, layer: TemporalDecisionLayer) -> None:
        """Verify direct escalation from ALERT to MICROSLEEP is permitted without intermediate states."""
        assert layer.current_state == ProjectState.ALERT
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05
        )
        for _ in range(layer.temporal_config.microsleep_persistence - 1):
            d = layer.update(closed_frame)
            assert d.current_state == ProjectState.ALERT

        d_final = layer.update(closed_frame)
        assert d_final.current_state == ProjectState.MICROSLEEP
        assert d_final.previous_state == ProjectState.ALERT

    def test_23_false_vit_drowsy_without_classical_fatigue_does_not_accumulate(
        self, layer: TemporalDecisionLayer
    ) -> None:
        """Verify high ViT DROWSY probability alone without classical fatigue cues does NOT accumulate."""
        # Simulated diagnostic HUD false positive case:
        # High ViT DROWSY (0.83) and high fused DROWSY (0.55), but all classical biometric signals normal
        false_drowsy_frame = make_mock_fusion_result(
            dominant=ProjectState.DROWSY,
            drowsy_score=0.85,
            eye_closure=0.23,
            yawn=0.46,
            perclos=0.02,
            head_nod=0.14,
            head_distraction=0.01,
            alert_score=0.31,
        )
        for _ in range(100):
            d = layer.update(false_drowsy_frame)
            assert d.current_state == ProjectState.ALERT
            assert d.drowsy_persistence_counter == 0
            assert d.low_vigilance_counter == 0

    def test_24_drowsy_reachable_when_sustained_classical_fatigue_exists(
        self, layer: TemporalDecisionLayer
    ) -> None:
        """Verify DROWSY state is reached when sustained classical fatigue exists."""
        # Genuine drowsiness with elevated PERCLOS
        true_drowsy_frame = make_mock_fusion_result(
            dominant=ProjectState.DROWSY,
            drowsy_score=0.85,
            perclos=0.60,
            eye_closure=0.30,
            alert_score=0.05,
        )
        for _ in range(layer.temporal_config.drowsy_persistence - 1):
            d = layer.update(true_drowsy_frame)
            assert d.current_state != ProjectState.DROWSY

        d = layer.update(true_drowsy_frame)
        assert d.current_state == ProjectState.DROWSY
        assert d.drowsy_persistence_counter == layer.temporal_config.drowsy_persistence

    def test_25_low_vigilance_not_triggered_by_drowsy_counter(
        self, layer: TemporalDecisionLayer
    ) -> None:
        """Verify LOW VIGILANCE is not triggered merely because drowsy counter reaches 20."""
        # Drowsy cues active (elevated PERCLOS), but low vigilance cues absent (low_vigilance_score low, no yawn/distract)
        drowsy_frame = make_mock_fusion_result(
            dominant=ProjectState.DROWSY,
            drowsy_score=0.85,
            perclos=0.60,
            low_vigilance_score=0.10,
            yawn=0.0,
            head_distraction=0.0,
            alert_score=0.05,
        )
        # Advance through 35 frames (past low_vigilance_persistence = 20)
        for _ in range(35):
            d = layer.update(drowsy_frame)
            assert d.low_vigilance_counter == 0
            assert d.current_state != ProjectState.LOW_VIGILANCE

    def test_26_microsleep_behavior_remains_unchanged(
        self, layer: TemporalDecisionLayer
    ) -> None:
        """Verify MICROSLEEP behavior is preserved with sustained physical eye closure."""
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.95,
            alert_score=0.05,
        )
        for _ in range(layer.temporal_config.microsleep_persistence - 1):
            d = layer.update(closed_frame)
            assert d.current_state == ProjectState.ALERT

        d = layer.update(closed_frame)
        assert d.current_state == ProjectState.MICROSLEEP
        assert d.transition is True

    def test_27_normal_blinking_remains_alert(
        self, layer: TemporalDecisionLayer
    ) -> None:
        """Verify normal blinks (<= max_normal_blink_frames) remain ALERT without accumulating fatigue."""
        alert_frame = make_mock_fusion_result(dominant=ProjectState.ALERT, alert_score=0.90)
        blink_frame = make_mock_fusion_result(
            dominant=ProjectState.ALERT,
            eye_closure=1.0,
            microsleep_score=0.10,
            alert_score=0.75,
        )
        # Simulate 10 normal blinks (6 frames closed, 20 frames open)
        for _ in range(10):
            for _ in range(6):
                d = layer.update(blink_frame)
                assert d.current_state == ProjectState.ALERT
            for _ in range(20):
                d = layer.update(alert_frame)
                assert d.current_state == ProjectState.ALERT

        assert layer.current_state == ProjectState.ALERT
        assert layer.drowsy_persistence_counter == 0
        assert layer.continuous_closed_frames == 0

    def test_28_normal_open_eyes_at_observed_ear_range_do_not_generate_perclos(self) -> None:
        """Verify normal open eyes (EAR ~0.22-0.24) do not get marked closed or inflate PERCLOS."""
        from models.baseline import BaselineDetector
        detector = BaselineDetector()
        mock_landmarks = np.zeros((468, 2), dtype=np.float32)
        # Configure eye landmarks to produce EAR = 0.23 (normal alert open eye)
        # EAR = (v1 + v2) / (2 * h) -> v1 = v2 = 9.2, h = 40 -> (9.2 + 9.2) / 80 = 0.23
        idx = detector.indices
        # Left eye: [33, 160, 158, 133, 153, 144]
        mock_landmarks[idx.LEFT_EYE[0]] = [100.0, 200.0]  # p1
        mock_landmarks[idx.LEFT_EYE[3]] = [140.0, 200.0]  # p4 (h = 40)
        mock_landmarks[idx.LEFT_EYE[1]] = [115.0, 195.4]  # p2
        mock_landmarks[idx.LEFT_EYE[5]] = [115.0, 204.6]  # p6 (v1 = 9.2)
        mock_landmarks[idx.LEFT_EYE[2]] = [125.0, 195.4]  # p3
        mock_landmarks[idx.LEFT_EYE[4]] = [125.0, 204.6]  # p5 (v2 = 9.2)
        # Right eye: symmetric
        mock_landmarks[idx.RIGHT_EYE[0]] = [200.0, 200.0]
        mock_landmarks[idx.RIGHT_EYE[3]] = [240.0, 200.0]
        mock_landmarks[idx.RIGHT_EYE[1]] = [215.0, 195.4]
        mock_landmarks[idx.RIGHT_EYE[5]] = [215.0, 204.6]
        mock_landmarks[idx.RIGHT_EYE[2]] = [225.0, 195.4]
        mock_landmarks[idx.RIGHT_EYE[4]] = [225.0, 204.6]

        # Feed 100 consecutive frames with EAR ~0.23
        for _ in range(100):
            res = detector.update(mock_landmarks)
            assert res.eyes_closed is False
            assert res.continuous_closed_frames == 0
            assert res.perclos == 0.0
            assert res.state == ProjectState.ALERT

    def test_29_static_pitch_does_not_continuously_count_as_head_nodding(
        self, layer: TemporalDecisionLayer
    ) -> None:
        """Verify stationary posture at +20 deg pitch does NOT count as head nodding or trigger DROWSY."""
        from models.baseline import BaselineDetector
        detector = BaselineDetector()
        mock_landmarks = np.zeros((468, 2), dtype=np.float32)
        # Normal open eye coordinates
        idx = detector.indices
        mock_landmarks[idx.LEFT_EYE[0]] = [100.0, 200.0]
        mock_landmarks[idx.LEFT_EYE[3]] = [140.0, 200.0]
        mock_landmarks[idx.LEFT_EYE[1]] = [115.0, 195.0]
        mock_landmarks[idx.LEFT_EYE[5]] = [115.0, 205.0]
        mock_landmarks[idx.LEFT_EYE[2]] = [125.0, 195.0]
        mock_landmarks[idx.LEFT_EYE[4]] = [125.0, 205.0]
        mock_landmarks[idx.RIGHT_EYE[0]] = [200.0, 200.0]
        mock_landmarks[idx.RIGHT_EYE[3]] = [240.0, 200.0]
        mock_landmarks[idx.RIGHT_EYE[1]] = [215.0, 195.0]
        mock_landmarks[idx.RIGHT_EYE[5]] = [215.0, 205.0]
        mock_landmarks[idx.RIGHT_EYE[2]] = [225.0, 195.0]
        mock_landmarks[idx.RIGHT_EYE[4]] = [225.0, 205.0]

        # 1. BaselineDetector check with static pitch = 20.0 deg
        for _ in range(100):
            b_res = detector.update(mock_landmarks, pitch=20.0)
            assert b_res.is_nodding is False
            assert detector.nod_counter == 0

        # 2. TemporalDecisionLayer check with static pitch evidence (constant head_nod ~0.80)
        static_pitch_frame = make_mock_fusion_result(
            dominant=ProjectState.ALERT,
            head_nod=0.80,  # 20.0 deg / 25.0 ref
            alert_score=0.85,
        )
        for _ in range(100):
            d = layer.update(static_pitch_frame)
            assert d.current_state == ProjectState.ALERT
            assert layer.nod_counter == 0
            assert layer.drowsy_persistence_counter == 0

    def test_30_genuine_sustained_eye_closure_still_works(
        self, layer: TemporalDecisionLayer
    ) -> None:
        """Verify genuine sustained eye closure (EAR < 0.18, eye_closure = 1.0) triggers MICROSLEEP."""
        from models.baseline import BaselineDetector
        detector = BaselineDetector()
        mock_landmarks = np.zeros((468, 2), dtype=np.float32)
        idx = detector.indices
        # Closed eye coordinates: v1 = v2 = 2.0, h = 40 -> EAR = 4/80 = 0.05 < 0.18
        mock_landmarks[idx.LEFT_EYE[0]] = [100.0, 200.0]
        mock_landmarks[idx.LEFT_EYE[3]] = [140.0, 200.0]
        mock_landmarks[idx.LEFT_EYE[1]] = [115.0, 199.0]
        mock_landmarks[idx.LEFT_EYE[5]] = [115.0, 201.0]
        mock_landmarks[idx.LEFT_EYE[2]] = [125.0, 199.0]
        mock_landmarks[idx.LEFT_EYE[4]] = [125.0, 201.0]
        mock_landmarks[idx.RIGHT_EYE[0]] = [200.0, 200.0]
        mock_landmarks[idx.RIGHT_EYE[3]] = [240.0, 200.0]
        mock_landmarks[idx.RIGHT_EYE[1]] = [215.0, 199.0]
        mock_landmarks[idx.RIGHT_EYE[5]] = [215.0, 201.0]
        mock_landmarks[idx.RIGHT_EYE[2]] = [225.0, 199.0]
        mock_landmarks[idx.RIGHT_EYE[4]] = [225.0, 201.0]

        for _ in range(20):
            b_res = detector.update(mock_landmarks)
            assert b_res.eyes_closed is True

        # Temporal layer: continuous closed frame reaches persistence -> MICROSLEEP
        closed_frame = make_mock_fusion_result(
            dominant=ProjectState.MICROSLEEP,
            eye_closure=1.0,
            microsleep_score=0.90,
            alert_score=0.05,
        )
        for _ in range(layer.temporal_config.microsleep_persistence - 1):
            d = layer.update(closed_frame)
            assert d.current_state == ProjectState.ALERT
        d = layer.update(closed_frame)
        assert d.current_state == ProjectState.MICROSLEEP

    def test_31_genuine_head_nod_movement_detected(
        self, layer: TemporalDecisionLayer
    ) -> None:
        """Verify dynamic downward head nod movement (downward excursion) is recognized."""
        from models.baseline import BaselineDetector
        detector = BaselineDetector()
        mock_landmarks = np.zeros((468, 2), dtype=np.float32)
        idx = detector.indices
        mock_landmarks[idx.LEFT_EYE[0]] = [100.0, 200.0]
        mock_landmarks[idx.LEFT_EYE[3]] = [140.0, 200.0]
        mock_landmarks[idx.LEFT_EYE[1]] = [115.0, 195.0]
        mock_landmarks[idx.LEFT_EYE[5]] = [115.0, 205.0]
        mock_landmarks[idx.LEFT_EYE[2]] = [125.0, 195.0]
        mock_landmarks[idx.LEFT_EYE[4]] = [125.0, 205.0]
        mock_landmarks[idx.RIGHT_EYE[0]] = [200.0, 200.0]
        mock_landmarks[idx.RIGHT_EYE[3]] = [240.0, 200.0]
        mock_landmarks[idx.RIGHT_EYE[1]] = [215.0, 195.0]
        mock_landmarks[idx.RIGHT_EYE[5]] = [215.0, 205.0]
        mock_landmarks[idx.RIGHT_EYE[2]] = [225.0, 195.0]
        mock_landmarks[idx.RIGHT_EYE[4]] = [225.0, 205.0]

        # Warmup resting pitch at 20.0 deg
        for _ in range(20):
            detector.update(mock_landmarks, pitch=20.0)

        # Dynamic downward drop from 20 deg to 35 deg (dip = +15 deg >= 12 deg threshold)
        for _ in range(detector.temporal.head_nod_persistence):
            res = detector.update(mock_landmarks, pitch=35.0)

        assert res.is_nodding is True

    def test_32_regression_microsleep_and_low_vigilance_and_vit_gating_preserved(
        self, layer: TemporalDecisionLayer
    ) -> None:
        """Verify MICROSLEEP, decoupled LOW-vigilance, and false-ViT gating remain robust."""
        # 1. False ViT DROWSY (0.85) with normal classical signals does not accumulate
        false_drowsy_frame = make_mock_fusion_result(
            dominant=ProjectState.DROWSY,
            drowsy_score=0.85,
            eye_closure=0.15,
            perclos=0.0,
            head_nod=0.20,
            yawn=0.10,
            alert_score=0.30,
        )
        for _ in range(50):
            d = layer.update(false_drowsy_frame)
            assert d.current_state == ProjectState.ALERT
            assert d.drowsy_persistence_counter == 0

        # 2. Drowsy counter does not trigger LOW VIGILANCE
        genuine_drowsy_frame = make_mock_fusion_result(
            dominant=ProjectState.DROWSY,
            drowsy_score=0.85,
            perclos=0.60,
            low_vigilance_score=0.10,
            alert_score=0.05,
        )
        for _ in range(25):
            d = layer.update(genuine_drowsy_frame)
            assert d.low_vigilance_counter == 0
            assert d.current_state != ProjectState.LOW_VIGILANCE



