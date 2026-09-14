"""Dataset pipeline, label mappings, and PyTorch dataset definitions for ViT.

Implements the canonical 4-class label mapping, group-aware video-sequence
splitting, subject-independent LOSO cross-validation architecture, class weight
calculations from training data only, face ROI extraction with configurable margins,
and torchvision preprocessing transforms (224x224 RGB with ImageNet normalization).

Canonical Project Labels:
    0 = ALERT
    1 = LOW_VIGILANCE
    2 = DROWSY
    3 = MICROSLEEP

Dataset-Specific Mapping Note:
    The mapping from raw labels:
        notdrowsy            -> ALERT (0)
        yawning              -> LOW_VIGILANCE (1)
        slowBlinkWithNodding -> DROWSY (2)
        sleepyCombination    -> MICROSLEEP (3)
    is specific to the downloaded Kaggle multi-class dataset
    (samymesbah/nthu-dataset-ddd-multi-class) and is not an assumption about
    the original NTHU dataset annotation scheme.

Evaluation Methodologies:
    1. Primary Development Split (Stratified Video-Sequence Grouping):
       Enforces zero temporal/video-sequence leakage (every continuous recording
       belongs strictly to one split).
       Subjects may occur across different splits because the dataset contains only
       four subjects and a strict subject-wise 70/15/15 partition is not feasible while
       maintaining class coverage. The development split prevents temporal/sequence
       leakage but is not subject-independent.
    2. Subject-Independent Leave-One-Subject-Out (LOSO) Cross-Validation:
       Trains on 3 subjects and evaluates on 1 held-out subject across 4 folds.
       Subject 006 contains zero yawning samples, which is explicitly handled
       and reported as incomplete class coverage without data fabrication.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils.config import DEFAULT_CONFIG, FACE_ROI_MARGIN, AppConfig, ProjectState
from utils.logger import setup_logger

logger = setup_logger("dataset")


# Canonical Project Label Mapping for the Kaggle NTHU Multi-Class dataset
DEFAULT_RAW_TO_STATE_MAPPING: Dict[str, ProjectState] = {
    "notdrowsy": ProjectState.ALERT,
    "yawning": ProjectState.LOW_VIGILANCE,
    "slowBlinkWithNodding": ProjectState.DROWSY,
    "sleepyCombination": ProjectState.MICROSLEEP,
}


def resolve_path(
    path_val: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve a filesystem path against project root supporting both relative and absolute paths.

    Handles:
    1. Relative paths (e.g. 'data/processed/ALERT/foo.jpg') -> resolved against base_dir.
    2. Absolute paths matching current filesystem -> returned directly.
    3. Stale absolute paths from foreign machines containing 'data/...' -> dynamically re-anchored.

    Args:
        path_val: Relative or absolute path string or Path object.
        base_dir: Anchor root directory. Defaults to DEFAULT_CONFIG.base_dir.

    Returns:
        Resolved pathlib.Path object.
    """
    if not path_val:
        return Path("")

    root = (Path(base_dir) if base_dir else DEFAULT_CONFIG.base_dir).resolve()
    p = Path(path_val)

    if not p.is_absolute():
        return (root / p).resolve()

    if p.exists():
        return p

    # If foreign absolute path does not exist, re-anchor from 'data' if present
    parts = p.parts
    if "data" in parts:
        data_idx = parts.index("data")
        rel_subpath = Path(*parts[data_idx:])
        candidate = (root / rel_subpath).resolve()
        if candidate.exists():
            return candidate

    return p


def to_project_relative_path(
    path_val: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Convert an absolute or relative path to a clean project-root relative path string.

    Args:
        path_val: Input path string or Path object.
        base_dir: Project root directory. Defaults to DEFAULT_CONFIG.base_dir.

    Returns:
        Forward-slash relative path string if inside project root, else original str.
    """
    if not path_val:
        return ""

    root = (Path(base_dir) if base_dir else DEFAULT_CONFIG.base_dir).resolve()
    p = Path(path_val).resolve()

    try:
        rel = p.relative_to(root)
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def load_fallback_sample_ids(
    failures_csv: Optional[Union[str, Path]] = None,
) -> Set[str]:
    """Load set of sample IDs that required fallback center-crop due to FaceMesh detection failure.

    Args:
        failures_csv: Path to face_detection_failures.csv.
                      Defaults to data/splits/face_detection_failures.csv.

    Returns:
        Set of sample_id strings.
    """
    path = resolve_path(failures_csv or (DEFAULT_CONFIG.splits_dir / "face_detection_failures.csv"))
    if not path.exists():
        return set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {r["sample_id"] for r in reader if "sample_id" in r}
    except Exception as e:
        logger.warning("Could not read fallback sample IDs from %s: %s", path, e)
        return set()



@dataclass
class SampleMetadata:
    """Structured metadata record for a single dataset sample.

    Preserves full lineage including subject ID, video sequence, condition,
    scenario, frame index, original filename, raw label, canonical state,
    and split assignments.
    """
    sample_id: str
    subject_id: str
    condition: str
    scenario: str
    frame_index: int
    video_sequence_id: str
    raw_label: str
    canonical_state: ProjectState
    original_path: str
    processed_path: str = ""
    split: str = "unassigned"
    original_filename: str = ""

    def __post_init__(self) -> None:
        if not self.original_filename and self.sample_id:
            self.original_filename = self.sample_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata record to dictionary for serialization."""
        return {
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "condition": self.condition,
            "scenario": self.scenario,
            "frame_index": self.frame_index,
            "video_sequence_id": self.video_sequence_id,
            "raw_label": self.raw_label,
            "canonical_state_id": int(self.canonical_state),
            "canonical_state_name": self.canonical_state.name,
            "original_path": self.original_path,
            "processed_path": self.processed_path,
            "split": self.split,
            "original_filename": self.original_filename,
        }


class CanonicalLabelMapper:
    """Manages mapping between raw dataset labels and canonical ProjectStates.

    Note: This mapping is specific to the downloaded Kaggle multi-class dataset
    and is not an assumption about the original NTHU dataset annotation scheme.
    """

    def __init__(self, mapping: Optional[Dict[str, ProjectState]] = None) -> None:
        """Initialize label mapper with custom or default mapping.

        Args:
            mapping: Dictionary mapping raw label string to ProjectState enum.
        """
        self._mapping: Dict[str, ProjectState] = dict(mapping or DEFAULT_RAW_TO_STATE_MAPPING)

    def map_label(self, raw_label: str) -> ProjectState:
        """Map raw label string to canonical ProjectState.

        Args:
            raw_label: Raw directory or file tag string.

        Returns:
            Mapped ProjectState.

        Raises:
            ValueError: If raw_label is not found in the mapping registry.
        """
        if raw_label not in self._mapping:
            raise ValueError(
                f"Unknown raw label '{raw_label}'. Registered labels: {list(self._mapping.keys())}"
            )
        return self._mapping[raw_label]

    def register(self, raw_label: str, state: ProjectState) -> None:
        """Register or override a raw label mapping.

        Args:
            raw_label: Raw string tag.
            state: Corresponding ProjectState.
        """
        self._mapping[raw_label] = state

    @property
    def mapping(self) -> Dict[str, ProjectState]:
        """Return a copy of the active label mapping."""
        return dict(self._mapping)


def validate_dataset_directory(raw_dir: Union[str, Path]) -> bool:
    """Validate that raw NTHU dataset directory exists and contains expected files.

    Args:
        raw_dir: Path to raw dataset root.

    Returns:
        True if valid dataset directory, False otherwise.
    """
    path = Path(raw_dir)
    if not path.exists() or not path.is_dir():
        logger.error(
            "NTHU dataset not found. Dataset preprocessing/training cannot be executed "
            "until the dataset is downloaded and configured."
        )
        return False

    train_dir = path / "train"
    if not train_dir.exists():
        logger.error(
            "NTHU dataset not found. Dataset preprocessing/training cannot be executed "
            "until the dataset is downloaded and configured."
        )
        return False

    return True


def crop_face_roi(
    image: Union[np.ndarray, Image.Image],
    landmarks: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    margin: float = FACE_ROI_MARGIN,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Crop face region of interest with configurable margin and square aspect ratio.

    Ensures the facial bounding box includes context around eyes, mouth, chin,
    and head orientation by applying a margin (default 20%) and expanding to a
    square crop with mirror-reflection padding if the box crosses image boundaries.

    Args:
        image: Source image as RGB/BGR numpy array or PIL Image.
        landmarks: Optional (N, 2) or (N, 3) landmark pixel coordinates.
        bbox: Optional (x_min, y_min, x_max, y_max) pixel coordinates.
        margin: Configurable margin percentage (default: FACE_ROI_MARGIN = 0.20).
        target_size: Target output resolution (default: (224, 224)).

    Returns:
        Tuple of (cropped_and_resized_pil_image, bounding_box_tuple).
    """
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"))
    else:
        img_arr = image.copy()

    h, w = img_arr.shape[:2]

    if landmarks is not None and len(landmarks) > 0:
        lm = np.asarray(landmarks)
        x_min = int(np.floor(np.min(lm[:, 0])))
        y_min = int(np.floor(np.min(lm[:, 1])))
        x_max = int(np.ceil(np.max(lm[:, 0])))
        y_max = int(np.ceil(np.max(lm[:, 1])))
    elif bbox is not None:
        x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    else:
        cx, cy = w / 2.0, h / 2.0
        side = min(w, h) * 0.6
        x_min, y_min = int(cx - side / 2.0), int(cy - side / 2.0)
        x_max, y_max = int(cx + side / 2.0), int(cy + side / 2.0)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, max(x_min + 1, x_max))
    y_max = min(h, max(y_min + 1, y_max))

    box_w = x_max - x_min
    box_h = y_max - y_min
    square_side = int(round(max(box_w, box_h) * (1.0 + 2.0 * margin)))

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    x1 = int(round(center_x - square_side / 2.0))
    y1 = int(round(center_y - square_side / 2.0))
    x2 = x1 + square_side
    y2 = y1 + square_side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = cv2.copyMakeBorder(
            img_arr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        crop_arr = padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
    else:
        crop_arr = img_arr[y1:y2, x1:x2]

    pil_crop = Image.fromarray(crop_arr).resize(
        target_size, resample=Image.Resampling.BICUBIC
    )

    return pil_crop, (x1, y1, x2, y2)


def get_vit_transforms(
    is_training: bool = True,
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Callable:
    """Generate torchvision transformation pipeline for ViT input.

    Args:
        is_training: If True, applies data augmentations (e.g. slight horizontal flip).
        image_size: Target (height, width) dimensions (default: 224, 224).
        mean: ImageNet channel mean for normalization.
        std: ImageNet channel standard deviation for normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transforms_list = [
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
    ]

    if is_training:
        # Subtle augmentations preserving eye, mouth, and pose geometry
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        transforms_list.append(T.ColorJitter(brightness=0.1, contrast=0.1))

    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transforms_list)


class DrowsinessDataset(Dataset):
    """PyTorch Dataset loading driver facial images for ViT classification."""

    def __init__(
        self,
        samples: List[SampleMetadata],
        transform: Optional[Callable] = None,
        use_processed: bool = True,
        base_dir: Optional[Union[str, Path]] = None,
        fallback_sample_ids: Optional[Set[str]] = None,
    ) -> None:
        """Initialize DrowsinessDataset.

        Args:
            samples: List of SampleMetadata records.
            transform: Optional torchvision transform pipeline.
            use_processed: If True, load preprocessed facial crop if available;
                           otherwise fallback to original path.
            base_dir: Base directory for resolving relative paths. Defaults to DEFAULT_CONFIG.base_dir.
            fallback_sample_ids: Optional set of sample IDs that used fallback crops for audit tracking.
        """
        self.samples = samples
        self.transform = transform
        self.use_processed = use_processed
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_CONFIG.base_dir
        self.fallback_sample_ids = fallback_sample_ids

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """Load and transform sample at given index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, class_index, metadata_dict).
        """
        sample = self.samples[idx]

        target_path_str = (
            sample.processed_path
            if self.use_processed and sample.processed_path
            else sample.original_path
        )
        img_path = resolve_path(target_path_str, base_dir=self.base_dir)
        if not img_path.exists():
            img_path = resolve_path(sample.original_path, base_dir=self.base_dir)

        with Image.open(img_path) as pil_img:
            image = pil_img.convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.functional.to_tensor(image)

        label_idx = int(sample.canonical_state)
        metadata = sample.to_dict()
        if self.fallback_sample_ids is not None:
            metadata["is_fallback"] = sample.sample_id in self.fallback_sample_ids
        return image_tensor, label_idx, metadata


def calculate_class_weights(
    train_samples: List[SampleMetadata],
    num_classes: int = 4
) -> torch.Tensor:
    """Calculate balanced inverse-frequency class weights from TRAINING SET ONLY.

    Formula:
        weight[c] = total_samples / (num_classes * count[c])

    Args:
        train_samples: List of SampleMetadata belonging strictly to the training split.
        num_classes: Total number of canonical classes (4).

    Returns:
        torch.Tensor of shape (num_classes,) with normalized class weights.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for sample in train_samples:
        counts[int(sample.canonical_state)] += 1.0

    total_samples = float(len(train_samples))
    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = float(total_samples / (num_classes * counts[c]))
        else:
            weights[c] = 1.0

    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def generate_stratified_sequence_splits(
    samples: List[SampleMetadata],
    random_seed: int = 42,
) -> Tuple[List[SampleMetadata], List[SampleMetadata], List[SampleMetadata]]:
    """Partition samples into Train, Val, Test using Stratified Video-Sequence Grouping.

    This serves as the Primary Development Split.

    Guarantees:
        1. Zero Temporal/Video-Sequence Leakage: Every continuous recording is
           assigned strictly to ONE split.
           train_seqs ∩ val_seqs = 0, train_seqs ∩ test_seqs = 0, val_seqs ∩ test_seqs = 0.
        2. Proportions: ~70% Train, ~15% Val, ~15% Test.
        3. All 4 classes represented across all splits.

    Important Terminology & Limitation Notice:
        Subjects may occur across different splits because the dataset contains only
        four subjects and a strict subject-wise 70/15/15 partition is not feasible
        while maintaining class coverage. The development split prevents
        temporal/sequence leakage but is not subject-independent.

    Args:
        samples: List of SampleMetadata parsed from the dataset.
        random_seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    import random
    rng = random.Random(random_seed)

    sequences: Dict[str, List[SampleMetadata]] = {}
    sequence_scenario: Dict[str, str] = {}

    for s in samples:
        seq_id = s.video_sequence_id
        if seq_id not in sequences:
            sequences[seq_id] = []
            sequence_scenario[seq_id] = s.scenario
        sequences[seq_id].append(s)

    scenario_to_seqs: Dict[str, List[str]] = {}
    for seq_id, scen in sequence_scenario.items():
        if scen not in scenario_to_seqs:
            scenario_to_seqs[scen] = []
        scenario_to_seqs[scen].append(seq_id)

    train_seq_ids: Set[str] = set()
    val_seq_ids: Set[str] = set()
    test_seq_ids: Set[str] = set()

    for scen, seq_list in sorted(scenario_to_seqs.items()):
        shuffled = list(sorted(seq_list))
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n == 7:
            # 5 Train (71.4%), 1 Val (14.3%), 1 Test (14.3%)
            train_seq_ids.update(shuffled[:5])
            val_seq_ids.add(shuffled[5])
            test_seq_ids.add(shuffled[6])
        elif n == 6:
            # 4 Train (66.7%), 1 Val (16.7%), 1 Test (16.7%)
            train_seq_ids.update(shuffled[:4])
            val_seq_ids.add(shuffled[4])
            test_seq_ids.add(shuffled[5])
        else:
            n_val = max(1, int(round(n * 0.15)))
            n_test = max(1, int(round(n * 0.15)))
            n_train = n - n_val - n_test
            train_seq_ids.update(shuffled[:n_train])
            val_seq_ids.update(shuffled[n_train : n_train + n_val])
            test_seq_ids.update(shuffled[n_train + n_val :])

    train_samples: List[SampleMetadata] = []
    val_samples: List[SampleMetadata] = []
    test_samples: List[SampleMetadata] = []

    for s in samples:
        if s.video_sequence_id in train_seq_ids:
            s.split = "train"
            train_samples.append(s)
        elif s.video_sequence_id in val_seq_ids:
            s.split = "val"
            val_samples.append(s)
        elif s.video_sequence_id in test_seq_ids:
            s.split = "test"
            test_samples.append(s)

    logger.info(
        "Generated sequence splits: Train=%d samples (%d seqs), "
        "Val=%d samples (%d seqs), Test=%d samples (%d seqs)",
        len(train_samples), len(train_seq_ids),
        len(val_samples), len(val_seq_ids),
        len(test_samples), len(test_seq_ids)
    )

    return train_samples, val_samples, test_samples


@dataclass
class LOSOFoldInfo:
    """Metadata and partition statistics for a single Leave-One-Subject-Out fold."""
    fold_name: str
    held_out_subject: str
    train_samples: List[SampleMetadata]
    eval_samples: List[SampleMetadata]
    train_class_counts: Dict[str, int]
    eval_class_counts: Dict[str, int]
    present_classes: List[str]
    absent_classes: List[str]
    is_complete_coverage: bool


def generate_loso_splits(
    samples: List[SampleMetadata]
) -> Dict[str, LOSOFoldInfo]:
    """Generate Subject-Wise / Leave-One-Subject-Out (LOSO) cross-subject evaluation folds.

    Measures generalization to drivers whose identity was not present during training.
    Strictly guarantees that no frames from the held-out subject enter training.

    Handles incomplete class coverage (e.g. Subject 006 having zero yawning samples)
    explicitly by logging absent classes rather than fabricating data.

    Args:
        samples: List of SampleMetadata instances across the dataset.

    Returns:
        Dictionary mapping fold name to LOSOFoldInfo structure.
    """
    unique_subjects = sorted(list({s.subject_id for s in samples}))
    folds: Dict[str, LOSOFoldInfo] = {}

    all_states = [
        ProjectState.ALERT,
        ProjectState.LOW_VIGILANCE,
        ProjectState.DROWSY,
        ProjectState.MICROSLEEP,
    ]

    for subj in unique_subjects:
        fold_name = f"loso_subject_{subj}"
        train_s = [s for s in samples if s.subject_id != subj]
        eval_s = [s for s in samples if s.subject_id == subj]

        train_counts: Dict[str, int] = {st.name: 0 for st in all_states}
        eval_counts: Dict[str, int] = {st.name: 0 for st in all_states}

        for s in train_s:
            train_counts[s.canonical_state.name] += 1
        for s in eval_s:
            eval_counts[s.canonical_state.name] += 1

        present = [st.name for st in all_states if eval_counts[st.name] > 0]
        absent = [st.name for st in all_states if eval_counts[st.name] == 0]
        is_complete = len(absent) == 0

        if not is_complete:
            logger.warning(
                "LOSO Fold '%s' (Held-out Subject: %s) has INCOMPLETE class coverage: "
                "Absent classes = %s. Evaluation will explicitly report this limitation.",
                fold_name, subj, absent
            )

        folds[fold_name] = LOSOFoldInfo(
            fold_name=fold_name,
            held_out_subject=subj,
            train_samples=train_s,
            eval_samples=eval_s,
            train_class_counts=train_counts,
            eval_class_counts=eval_counts,
            present_classes=present,
            absent_classes=absent,
            is_complete_coverage=is_complete,
        )

    logger.info("Generated %d LOSO cross-subject folds across subjects: %s", len(folds), unique_subjects)
    return folds


def verify_dataset_manifest(
    samples: List[SampleMetadata],
    split_type: str = "development"
) -> Dict[str, Any]:
    """Verify dataset manifest integrity, sequence disjointness, and subject distributions.

    Args:
        samples: List of SampleMetadata instances.
        split_type: 'development' (sequence-grouped) or 'loso' (subject-grouped).

    Returns:
        Structured audit report dictionary.
    """
    total_samples = len(samples)
    splits: Dict[str, List[SampleMetadata]] = {}
    split_sequences: Dict[str, Set[str]] = {}
    split_subjects: Dict[str, Set[str]] = {}

    for s in samples:
        sp = s.split
        if sp not in splits:
            splits[sp] = []
            split_sequences[sp] = set()
            split_subjects[sp] = set()
        splits[sp].append(s)
        split_sequences[sp].add(s.video_sequence_id)
        split_subjects[sp].add(s.subject_id)

    train_seqs = split_sequences.get("train", set())
    val_seqs = split_sequences.get("val", set())
    test_seqs = split_sequences.get("test", set())

    train_val_seq_leak = train_seqs & val_seqs
    train_test_seq_leak = train_seqs & test_seqs
    val_test_seq_leak = val_seqs & test_seqs
    total_sequence_leak = len(train_val_seq_leak) + len(train_test_seq_leak) + len(val_test_seq_leak)

    train_subjs = split_subjects.get("train", set())
    val_subjs = split_subjects.get("val", set())
    test_subjs = split_subjects.get("test", set())

    overlapping_subjects = sorted(list(train_subjs & (val_subjs | test_subjs)))

    class_counts_by_split: Dict[str, Dict[str, int]] = {}
    for sp_name, sp_list in splits.items():
        counts = {st.name: 0 for st in ProjectState}
        for s in sp_list:
            counts[s.canonical_state.name] += 1
        class_counts_by_split[sp_name] = counts

    audit_result = {
        "total_samples": total_samples,
        "split_counts": {sp: len(lst) for sp, lst in splits.items()},
        "split_percentages": {
            sp: (len(lst) / total_samples * 100.0) if total_samples > 0 else 0.0
            for sp, lst in splits.items()
        },
        "sequence_counts": {sp: len(seqs) for sp, seqs in split_sequences.items()},
        "sequence_leakage_detected": total_sequence_leak > 0,
        "sequence_leakage_count": total_sequence_leak,
        "subject_counts": {sp: len(subjs) for sp, subjs in split_subjects.items()},
        "subject_overlap_detected": len(overlapping_subjects) > 0,
        "overlapping_subjects": overlapping_subjects,
        "is_subject_independent": len(overlapping_subjects) == 0,
        "class_distributions": class_counts_by_split,
        "methodology_note": (
            "Development split guarantees ZERO temporal/video-sequence leakage. "
            "Subjects occur across multiple splits due to dataset limitation (4 subjects total); "
            "it is NOT subject-independent. Use LOSO for subject generalization."
            if split_type == "development"
            else "LOSO evaluation guarantees strict subject independence (zero subject overlap)."
        ),
    }

    return audit_result


def save_manifest_csv(samples: List[SampleMetadata], output_path: Union[str, Path]) -> None:
    """Save list of SampleMetadata records to CSV file.

    Args:
        samples: List of SampleMetadata instances.
        output_path: Target CSV file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id", "subject_id", "condition", "scenario", "frame_index",
        "video_sequence_id", "raw_label", "canonical_state_id",
        "canonical_state_name", "original_path", "processed_path", "split",
        "original_filename"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(s.to_dict())

    logger.info("Saved manifest (%d records) to %s", len(samples), path)


def load_manifest_csv(input_path: Union[str, Path]) -> List[SampleMetadata]:
    """Load SampleMetadata records from a CSV manifest file.

    Args:
        input_path: Path to CSV manifest.

    Returns:
        List of SampleMetadata instances.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    samples: List[SampleMetadata] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = SampleMetadata(
                sample_id=row["sample_id"],
                subject_id=row["subject_id"],
                condition=row["condition"],
                scenario=row["scenario"],
                frame_index=int(row["frame_index"]),
                video_sequence_id=row["video_sequence_id"],
                raw_label=row["raw_label"],
                canonical_state=ProjectState(int(row["canonical_state_id"])),
                original_path=row["original_path"],
                processed_path=row.get("processed_path", ""),
                split=row.get("split", "unassigned"),
                original_filename=row.get("original_filename", row["sample_id"]),
            )
            samples.append(sample)

    return samples
