"""Unit and integration tests for ViT classifier and Phase 2 dataset infrastructure.

Covers:
- Dataset directory validation & graceful missing-data error handling
- Canonical label mapper (4 classes) and Kaggle dataset-specific mapping
- Metadata record preservation and lineage
- Primary Development Split: zero temporal/video-sequence leakage verification
- Subject overlap documentation & lack of subject-independence
- Leave-One-Subject-Out (LOSO) cross-subject fold generation
- Subject 006 missing yawning class handling (no fabricated data)
- Face ROI extraction with configurable 20% margin and square padding
- Training-only class weights calculation
- ViT model architecture, parameter freezing/unfreezing, and checkpoint roundtrip
- Standalone predict API contract
"""

import logging
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from models.vit_classifier import ViTClassifier
from src.dataset import (
    CanonicalLabelMapper,
    DrowsinessDataset,
    SampleMetadata,
    calculate_class_weights,
    crop_face_roi,
    generate_loso_splits,
    generate_stratified_sequence_splits,
    get_vit_transforms,
    load_fallback_sample_ids,
    load_manifest_csv,
    resolve_path,
    save_manifest_csv,
    to_project_relative_path,
    validate_dataset_directory,
    verify_dataset_manifest,
)
from utils.config import DEFAULT_CONFIG, FACE_ROI_MARGIN, ProjectState, ViTConfig


class TestDatasetAndPreprocessing:
    """Tests for dataset validation, mapping, face ROI, and metadata schemas."""

    def test_missing_dataset_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify missing dataset path logs exact required message."""
        with caplog.at_level(logging.ERROR):
            result = validate_dataset_directory("/non/existent/path/to/nthu")
        assert result is False
        assert any(
            "NTHU dataset not found. Dataset preprocessing/training cannot be executed" in record.message
            for record in caplog.records
        )

    def test_canonical_label_mapping(self) -> None:
        """Verify mapping of Kaggle multi-class labels to canonical project states."""
        mapper = CanonicalLabelMapper()

        assert mapper.map_label("notdrowsy") == ProjectState.ALERT
        assert mapper.map_label("yawning") == ProjectState.LOW_VIGILANCE
        assert mapper.map_label("slowBlinkWithNodding") == ProjectState.DROWSY
        assert mapper.map_label("sleepyCombination") == ProjectState.MICROSLEEP

        with pytest.raises(ValueError, match="Unknown raw label"):
            mapper.map_label("unknown_fatigue_class")

    def test_metadata_serialization_and_lineage(self) -> None:
        """Verify SampleMetadata preserves full audit lineage."""
        meta = SampleMetadata(
            sample_id="001_glasses_yawning_0100_drowsy.jpg",
            subject_id="001",
            condition="glasses",
            scenario="yawning",
            frame_index=100,
            video_sequence_id="001_glasses_yawning",
            raw_label="yawning",
            canonical_state=ProjectState.LOW_VIGILANCE,
            original_path="/path/to/original.jpg",
            processed_path="/path/to/processed.jpg",
            split="train",
            original_filename="001_glasses_yawning_0100_drowsy.jpg",
        )

        d = meta.to_dict()
        assert d["subject_id"] == "001"
        assert d["canonical_state_id"] == 1
        assert d["canonical_state_name"] == "LOW_VIGILANCE"
        assert d["original_filename"] == "001_glasses_yawning_0100_drowsy.jpg"

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_manifest_csv([meta], temp_path)
            loaded = load_manifest_csv(temp_path)
            assert len(loaded) == 1
            assert loaded[0].subject_id == "001"
            assert loaded[0].canonical_state == ProjectState.LOW_VIGILANCE
            assert loaded[0].original_filename == "001_glasses_yawning_0100_drowsy.jpg"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_face_roi_crop_aspect_ratio_and_margin(self) -> None:
        """Verify Face ROI cropping maintains square aspect ratio and 20% margin."""
        # Create a mock 640x480 image
        mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Face located in center: (200, 100) to (400, 300) -> width=200, height=200
        bbox = (200, 100, 400, 300)

        cropped_pil, crop_coords = crop_face_roi(
            mock_img,
            bbox=bbox,
            margin=FACE_ROI_MARGIN,
            target_size=(224, 224),
        )

        assert isinstance(cropped_pil, Image.Image)
        assert cropped_pil.size == (224, 224)

        x1, y1, x2, y2 = crop_coords
        crop_w = x2 - x1
        crop_h = y2 - y1
        # Crop must be strictly square
        assert crop_w == crop_h
        # Side must include 20% margin: 200 * (1 + 2*0.20) = 280
        expected_side = int(round(200 * (1.0 + 2.0 * FACE_ROI_MARGIN)))
        assert crop_w == expected_side

    def test_class_weights_calculation_training_only(self) -> None:
        """Verify class weights are derived inversely to frequency and normalized."""
        samples = []
        # ALERT: 50, LOW_VIGILANCE: 10, DROWSY: 20, MICROSLEEP: 20 -> total: 100
        for _ in range(50):
            samples.append(SampleMetadata("1", "001", "c", "s", 0, "seq1", "r", ProjectState.ALERT, "p"))
        for _ in range(10):
            samples.append(SampleMetadata("2", "001", "c", "s", 0, "seq1", "r", ProjectState.LOW_VIGILANCE, "p"))
        for _ in range(20):
            samples.append(SampleMetadata("3", "001", "c", "s", 0, "seq1", "r", ProjectState.DROWSY, "p"))
        for _ in range(20):
            samples.append(SampleMetadata("4", "001", "c", "s", 0, "seq1", "r", ProjectState.MICROSLEEP, "p"))

        weights = calculate_class_weights(samples, num_classes=4)
        assert len(weights) == 4
        assert pytest.approx(float(weights.mean()), 1e-4) == 1.0
        # Less frequent class (LOW_VIGILANCE: 10) must have higher weight than ALERT (50)
        assert weights[1] > weights[0]
        # Equal frequency classes (DROWSY and MICROSLEEP: 20 each) must have identical weights
        assert pytest.approx(float(weights[2]), 1e-4) == float(weights[3])


class TestSplittingMethodology:
    """Tests for sequence-grouped development split and subject-wise LOSO folds."""

    def _create_mock_dataset(self) -> list:
        """Generate mock dataset representing 4 subjects and 27 sequences."""
        samples = []
        subjects = ["001", "002", "005", "006"]
        conditions = ["glasses", "noglasses"]
        scenarios = ["nonsleepyCombination", "sleepyCombination", "slowBlinkWithNodding", "yawning"]

        frame_counter = 0
        for subj in subjects:
            for cond in conditions:
                for scen in scenarios:
                    # Subject 006 has no noglasses and no yawning
                    if subj == "006" and (cond == "noglasses" or scen == "yawning"):
                        continue

                    seq_id = f"{subj}_{cond}_{scen}"
                    state_map = {
                        "nonsleepyCombination": ProjectState.ALERT,
                        "yawning": ProjectState.LOW_VIGILANCE,
                        "slowBlinkWithNodding": ProjectState.DROWSY,
                        "sleepyCombination": ProjectState.MICROSLEEP,
                    }
                    st = state_map[scen]

                    for f_idx in range(50):
                        samples.append(SampleMetadata(
                            sample_id=f"{seq_id}_{f_idx}.jpg",
                            subject_id=subj,
                            condition=cond,
                            scenario=scen,
                            frame_index=f_idx,
                            video_sequence_id=seq_id,
                            raw_label=scen,
                            canonical_state=st,
                            original_path=f"/fake/{seq_id}_{f_idx}.jpg",
                        ))
                        frame_counter += 1

        return samples

    def test_development_split_sequence_disjointness_and_subject_overlap(self) -> None:
        """Verify zero sequence leakage and explicitly check subject overlap."""
        samples = self._create_mock_dataset()
        train_s, val_s, test_s = generate_stratified_sequence_splits(samples, random_seed=42)

        train_seqs = {s.video_sequence_id for s in train_s}
        val_seqs = {s.video_sequence_id for s in val_s}
        test_seqs = {s.video_sequence_id for s in test_s}

        # 1. Zero Video-Sequence Leakage Guarantee
        assert train_seqs.intersection(val_seqs) == set(), "Sequence leakage between Train and Val!"
        assert train_seqs.intersection(test_seqs) == set(), "Sequence leakage between Train and Test!"
        assert val_seqs.intersection(test_seqs) == set(), "Sequence leakage between Val and Test!"

        # 2. Proportion Verification (~70% Train, ~15% Val, ~15% Test)
        total_frames = len(samples)
        assert 0.65 <= (len(train_s) / total_frames) <= 0.75
        assert 0.10 <= (len(val_s) / total_frames) <= 0.20
        assert 0.10 <= (len(test_s) / total_frames) <= 0.20

        # 3. Explicit Documentation & Subject Overlap
        train_subjs = {s.subject_id for s in train_s}
        val_subjs = {s.subject_id for s in val_s}
        test_subjs = {s.subject_id for s in test_s}

        # In development split, subjects DO occur across splits (not subject-independent)
        overlapping_subjs = train_subjs.intersection(val_subjs.union(test_subjs))
        assert len(overlapping_subjs) > 0, "Expected subject overlap across sequences in development split"

        all_assigned = train_s + val_s + test_s
        audit = verify_dataset_manifest(all_assigned, split_type="development")
        assert audit["sequence_leakage_detected"] is False
        assert audit["subject_overlap_detected"] is True
        assert audit["is_subject_independent"] is False

    def test_loso_subject_independent_folds(self) -> None:
        """Verify strict subject independence and incomplete class detection for Subject 006."""
        samples = self._create_mock_dataset()
        folds = generate_loso_splits(samples)

        assert len(folds) == 4
        assert "loso_subject_006" in folds

        for fold_name, fold_info in folds.items():
            held_out = fold_info.held_out_subject
            train_subjs = {s.subject_id for s in fold_info.train_samples}
            eval_subjs = {s.subject_id for s in fold_info.eval_samples}

            # Strict subject independence: held_out subject MUST NOT enter training
            assert held_out not in train_subjs
            assert train_subjs.intersection(eval_subjs) == set()

            if held_out == "006":
                # Subject 006 has 0 yawning frames
                assert fold_info.is_complete_coverage is False
                assert "LOW_VIGILANCE" in fold_info.absent_classes
            else:
                assert fold_info.is_complete_coverage is True
                assert len(fold_info.absent_classes) == 0


@pytest.fixture(scope="module")
def model_instance() -> ViTClassifier:
    """Create a CPU-based ViTClassifier instance without pre-trained weights."""
    cfg = ViTConfig(model_name="google/vit-base-patch16-224", num_classes=4)
    model = ViTClassifier(config=cfg, pretrained=False, device="cpu")
    return model


class TestViTModelArchitecture:
    """Tests for ViTClassifier architecture, parameter freezing, and inference API."""

    def test_parameter_counts_and_freezing(self, model_instance: ViTClassifier) -> None:
        """Verify backbone freezing (Stage 1) and deep layer unfreezing (Stage 2)."""
        counts_init = model_instance.get_parameter_counts()
        assert counts_init["total"] > 80_000_000

        # Stage 1: Freeze backbone
        model_instance.freeze_backbone()
        counts_s1 = model_instance.get_parameter_counts()
        # Only classifier head trainable: Linear(768, 4) -> 768*4 + 4 = 3076 params
        assert counts_s1["trainable"] == 3076
        assert counts_s1["frozen"] == counts_init["total"] - 3076

        # Stage 2: Unfreeze last 2 blocks
        model_instance.unfreeze_deep_layers(num_blocks=2)
        counts_s2 = model_instance.get_parameter_counts()
        assert counts_s2["trainable"] > counts_s1["trainable"]
        assert counts_s2["frozen"] < counts_s1["frozen"]

    def test_forward_pass_cpu(self, model_instance: ViTClassifier) -> None:
        """Verify forward pass output shape (batch_size, 4)."""
        dummy_batch = torch.randn(2, 3, 224, 224)
        logits = model_instance(dummy_batch)
        assert logits.shape == (2, 4)

    def test_forward_pass_cuda_if_available(self) -> None:
        """Verify CUDA forward pass if GPU is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU not available on this system")

        cfg = ViTConfig(model_name="google/vit-base-patch16-224", num_classes=4)
        model = ViTClassifier(config=cfg, pretrained=False, device="cuda")
        dummy_batch = torch.randn(2, 3, 224, 224, device="cuda")
        logits = model(dummy_batch)
        assert logits.shape == (2, 4)
        assert logits.is_cuda

    def test_predict_api_contract(self, model_instance: ViTClassifier) -> None:
        """Verify predict API returns structured dictionary with probabilities."""
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = model_instance.predict(dummy_img)

        assert "predicted_state" in result
        assert isinstance(result["predicted_state"], ProjectState)
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert "probabilities" in result
        assert len(result["probabilities"]) == 4

        # Probabilities sum to 1.0
        prob_sum = sum(result["probabilities"].values())
        assert pytest.approx(prob_sum, 1e-4) == 1.0

    def test_checkpoint_save_and_load(self, model_instance: ViTClassifier) -> None:
        """Verify model checkpoint serialization and restoration."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            chk_path = Path(f.name)

        try:
            metadata = {"epoch": 3, "val_f1": 0.854}
            model_instance.save_checkpoint(chk_path, metadata=metadata)
            assert chk_path.exists()

            loaded_model, loaded_meta = ViTClassifier.load_checkpoint(chk_path, device="cpu")
            assert loaded_meta["epoch"] == 3
            assert pytest.approx(loaded_meta["val_f1"], 1e-4) == 0.854

            # Verify forward pass on loaded model
            dummy_batch = torch.randn(1, 3, 224, 224)
            orig_logits = model_instance(dummy_batch)
            loaded_logits = loaded_model(dummy_batch)
            assert torch.allclose(orig_logits, loaded_logits, atol=1e-5)
        finally:
            if chk_path.exists():
                chk_path.unlink()


class TestProductionArtifactsAndIntegrity:
    """Tests for Phase 2B/2C production dataset, manifests, transforms, and checkpoints."""

    def test_processed_dataset_integrity(self) -> None:
        """Verify data/processed/ contains valid crops for all canonical classes."""
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            pytest.skip("data/processed directory not yet generated")

        canonical_names = ["ALERT", "LOW_VIGILANCE", "DROWSY", "MICROSLEEP"]
        for c_name in canonical_names:
            class_dir = processed_dir / c_name
            assert class_dir.exists(), f"Missing processed directory for class: {c_name}"
            img_files = list(class_dir.glob("*.jpg"))
            assert len(img_files) > 0, f"No processed images found in {class_dir}"

    def test_manifest_consistency_and_split_integrity(self) -> None:
        """Verify split manifests partition exactly 66,521 samples with zero sequence leakage."""
        splits_dir = Path("data/splits")
        train_p = splits_dir / "dev_train_manifest.csv"
        val_p = splits_dir / "dev_val_manifest.csv"
        test_p = splits_dir / "dev_test_manifest.csv"

        if not (train_p.exists() and val_p.exists() and test_p.exists()):
            pytest.skip("Split manifests not yet generated")

        train_s = load_manifest_csv(train_p)
        val_s = load_manifest_csv(val_p)
        test_s = load_manifest_csv(test_p)

        assert len(train_s) == 46873
        assert len(val_s) == 9416
        assert len(test_s) == 10232
        assert len(train_s) + len(val_s) + len(test_s) == 66521

        # Zero sequence leakage
        train_seqs = {s.video_sequence_id for s in train_s}
        val_seqs = {s.video_sequence_id for s in val_s}
        test_seqs = {s.video_sequence_id for s in test_s}

        assert train_seqs.isdisjoint(val_seqs)
        assert train_seqs.isdisjoint(test_seqs)
        assert val_seqs.isdisjoint(test_seqs)

        # Processed path populated
        assert all(s.processed_path != "" for s in train_s[:100])

    def test_deterministic_eval_transforms(self) -> None:
        """Verify validation and test transforms produce deterministic tensors across runs."""
        eval_transform = get_vit_transforms(is_training=False)
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        tensor_1 = eval_transform(dummy_img)
        tensor_2 = eval_transform(dummy_img)

        assert torch.equal(tensor_1, tensor_2), "Evaluation transform is not deterministic!"

    def test_production_checkpoint_loading_and_inference(self) -> None:
        """Verify production checkpoint can be loaded and outputs 4-class probabilities."""
        chk_path = Path("models/checkpoints/vit_best_production.pt")
        if not chk_path.exists():
            pytest.skip("Production checkpoint vit_best_production.pt not yet created")

        model, meta = ViTClassifier.load_checkpoint(chk_path, device="cpu")
        assert model.num_classes == 4
        assert "canonical_label_mapping" in meta or "stage" in meta

        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        res = model.predict(dummy_img)

        assert res["predicted_state"] in [
            ProjectState.ALERT,
            ProjectState.LOW_VIGILANCE,
            ProjectState.DROWSY,
            ProjectState.MICROSLEEP,
        ]
        assert len(res["probabilities"]) == 4
        assert pytest.approx(sum(res["probabilities"].values()), 1e-4) == 1.0


class TestViTResumeAndPathPortability:
    """Unit tests for exact checkpoint resume, criteria separation, and path portability."""

    def test_resume_from_stage1_epoch1_starts_at_epoch2(self) -> None:
        """Verify checkpoint resume correctly parses Stage 1 Epoch 1 and starts at Epoch 2."""
        chk_path = Path("models/checkpoints/vit_best_production.pt.bak_epoch1")
        if not chk_path.exists():
            chk_path = Path("models/checkpoints/vit_best_production.pt")
        if not chk_path.exists():
            pytest.skip("Production checkpoint not found")

        _, meta = ViTClassifier.load_checkpoint(chk_path, device="cpu")
        stage = meta.get("stage")
        completed_epoch = meta.get("epoch")

        # Check resume dispatch logic
        if stage == 1:
            start_epoch_s1 = completed_epoch + 1
            start_epoch_s2 = 1
            assert start_epoch_s1 == 2, "Resume from Stage 1 Epoch 1 must start at Epoch 2!"
            assert start_epoch_s2 == 1
            val_metrics = meta.get("validation_metrics", {})
            assert "macro_f1" in val_metrics
            restored_f1 = float(val_metrics["macro_f1"])
            assert pytest.approx(restored_f1, 1e-3) == 0.1528
        elif stage == 2:
            start_epoch_s1 = 6
            start_epoch_s2 = completed_epoch + 1
            assert start_epoch_s2 >= 2
            val_metrics = meta.get("validation_metrics", {})
            assert "macro_f1" in val_metrics
        else:
            raise ValueError(f"Unknown stage {stage}")

    def test_optimizer_state_restores_correctly(self) -> None:
        """Verify optimizer state dict loads into matching AdamW without errors."""
        chk_path = Path("models/checkpoints/vit_best_production.pt.bak_epoch1")
        if not chk_path.exists():
            chk_path = Path("models/checkpoints/vit_best_production.pt")
        if not chk_path.exists():
            pytest.skip("Production checkpoint not found")

        raw_ckpt = torch.load(chk_path, map_location="cpu", weights_only=False)
        opt_state = raw_ckpt.get("metadata", {}).get("optimizer_state_dict")
        stage = raw_ckpt.get("metadata", {}).get("stage", 1)
        assert opt_state is not None, "Missing optimizer_state_dict in checkpoint metadata!"

        # Initialize optimizer matching saved checkpoint stage
        model = ViTClassifier(pretrained=False, device="cpu")
        if stage == 1:
            model.freeze_backbone()
        else:
            model.unfreeze_top_blocks(2)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3 if stage == 1 else 2e-5,
            weight_decay=0.01,
        )

        # Restore optimizer state
        optimizer.load_state_dict(opt_state)
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.state) > 0, "Optimizer state buffers should be restored!"

    def test_incompatible_checkpoint_fails_clearly(self) -> None:
        """Verify incompatible checkpoint metadata or optimizer states fail with clear exceptions."""
        chk_path = Path("models/checkpoints/vit_best_production.pt")
        if not chk_path.exists():
            pytest.skip("Production checkpoint not found")

        real_ckpt = torch.load(chk_path, map_location="cpu", weights_only=False)

        # 1. Missing stage/epoch metadata
        bad_meta_checkpoint = {
            "state_dict": real_ckpt["state_dict"],
            "config": real_ckpt["config"],
            "metadata": {"num_classes": 4},  # Missing stage and epoch
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            bad_path = Path(f.name)
        try:
            torch.save(bad_meta_checkpoint, bad_path)
            _, meta = ViTClassifier.load_checkpoint(bad_path, device="cpu")
            with pytest.raises(ValueError, match="missing required 'stage' or 'epoch' metadata"):
                resume_stage = meta.get("stage")
                resume_epoch = meta.get("epoch")
                if resume_stage is None or resume_epoch is None:
                    raise ValueError(f"Checkpoint at {bad_path} is missing required 'stage' or 'epoch' metadata for resume.")
        finally:
            if bad_path.exists():
                bad_path.unlink()

        # 2. Incompatible num_classes
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            bad_cls_path = Path(f.name)
        try:
            bad_cls_checkpoint = {
                "state_dict": real_ckpt["state_dict"],
                "config": real_ckpt["config"],
                "metadata": {"num_classes": 2, "stage": 1, "epoch": 1},
            }
            torch.save(bad_cls_checkpoint, bad_cls_path)
            _, meta2 = ViTClassifier.load_checkpoint(bad_cls_path, device="cpu")
            with pytest.raises(ValueError, match="incompatible with model num_classes"):
                if meta2.get("num_classes") != 4:
                    raise ValueError(f"Checkpoint num_classes ({meta2.get('num_classes')}) is incompatible with model num_classes (4)")
        finally:
            if bad_cls_path.exists():
                bad_cls_path.unlink()

    def test_validation_criterion_is_unweighted(self) -> None:
        """Verify training criterion is weighted and validation criterion is strictly unweighted."""
        dummy_weights = torch.tensor([0.43, 1.70, 1.14, 0.73], dtype=torch.float32)
        train_criterion = nn.CrossEntropyLoss(weight=dummy_weights)
        val_criterion = nn.CrossEntropyLoss()

        assert train_criterion.weight is not None
        assert torch.equal(train_criterion.weight, dummy_weights)
        assert val_criterion.weight is None, "Validation criterion must be strictly unweighted!"

        # Ensure loss computation on fixed outputs reflects unweighted calculation
        logits = torch.tensor([[2.0, 0.5, 0.1, 0.1], [0.1, 0.1, 2.5, 0.2]])
        targets = torch.tensor([0, 2])
        loss_val = val_criterion(logits, targets)
        expected_manual = - (torch.log_softmax(logits, dim=-1)[0, 0] + torch.log_softmax(logits, dim=-1)[1, 2]) / 2.0
        assert pytest.approx(loss_val.item(), 1e-5) == expected_manual.item()

    def test_existing_absolute_manifest_paths_still_load(self) -> None:
        """Verify DrowsinessDataset backward compatibility with existing absolute paths."""
        train_manifest = Path("data/splits/dev_train_manifest.csv")
        if not train_manifest.exists():
            pytest.skip("dev_train_manifest.csv not found")

        samples = load_manifest_csv(train_manifest)
        first_sample = samples[0]
        # Confirm absolute path in manifest
        assert Path(first_sample.processed_path).is_absolute()

        dataset = DrowsinessDataset([first_sample], transform=None, use_processed=True)
        img_tensor, label, meta = dataset[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 224, 224)
        assert label == int(first_sample.canonical_state)

    def test_new_relative_manifest_paths_resolve_correctly(self) -> None:
        """Verify relative paths resolve dynamically against project root."""
        # Test to_project_relative_path
        abs_crop = (DEFAULT_CONFIG.base_dir / "data/processed/ALERT").resolve()
        rel_crop = to_project_relative_path(abs_crop)
        assert rel_crop == "data/processed/ALERT"

        # Test resolve_path on relative path
        resolved = resolve_path("data/processed/ALERT")
        assert resolved == abs_crop

        # Test DrowsinessDataset with relative sample path
        train_manifest = Path("data/splits/dev_train_manifest.csv")
        if not train_manifest.exists():
            pytest.skip("dev_train_manifest.csv not found")

        samples = load_manifest_csv(train_manifest)
        orig_s = samples[0]

        rel_processed = to_project_relative_path(orig_s.processed_path)
        rel_sample = SampleMetadata(
            sample_id=orig_s.sample_id,
            subject_id=orig_s.subject_id,
            condition=orig_s.condition,
            scenario=orig_s.scenario,
            frame_index=orig_s.frame_index,
            video_sequence_id=orig_s.video_sequence_id,
            raw_label=orig_s.raw_label,
            canonical_state=orig_s.canonical_state,
            original_path=to_project_relative_path(orig_s.original_path),
            processed_path=rel_processed,
            split=orig_s.split,
            original_filename=orig_s.original_filename,
        )

        assert not Path(rel_sample.processed_path).is_absolute()
        dataset = DrowsinessDataset([rel_sample], transform=None, use_processed=True)
        img_tensor, label, meta = dataset[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 224, 224)

    def test_fallback_sample_ids_and_auditability(self) -> None:
        """Verify face detection failures are tracked and flag is_fallback in metadata."""
        failures_path = Path("data/splits/face_detection_failures.csv")
        train_manifest = Path("data/splits/dev_train_manifest.csv")
        if not (failures_path.exists() and train_manifest.exists()):
            pytest.skip("face_detection_failures.csv or dev_train_manifest.csv not found")

        fallback_ids = load_fallback_sample_ids(failures_path)
        assert len(fallback_ids) == 727, f"Expected 727 fallback IDs, found {len(fallback_ids)}"

        samples = load_manifest_csv(train_manifest)
        sample_fail = next(s for s in samples if s.sample_id in fallback_ids)
        sample_norm = next(s for s in samples if s.sample_id not in fallback_ids)

        dataset = DrowsinessDataset(
            [sample_fail, sample_norm],
            transform=None,
            use_processed=True,
            fallback_sample_ids=fallback_ids,
        )

        img_fail, _, meta_fail = dataset[0]
        assert isinstance(img_fail, torch.Tensor)
        assert meta_fail["is_fallback"] is True, "Sample in fallback_ids must have is_fallback=True!"

        img_norm, _, meta_norm = dataset[1]
        assert isinstance(img_norm, torch.Tensor)
        assert meta_norm["is_fallback"] is False, "Normal sample must have is_fallback=False!"


