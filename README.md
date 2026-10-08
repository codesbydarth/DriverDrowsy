# Real-Time Driver Drowsiness Detection System with Vision Transformer & Multimodal Biometric Fusion

An end-to-end, non-intrusive driver drowsiness monitoring system developed as an academic engineering project for **VIT Bhopal University**.

The system integrates **Google MediaPipe FaceMesh** landmark tracking, classical ocular and facial biometrics (EAR, MAR, PERCLOS, 3D Head Pose), a fine-tuned **Vision Transformer (ViT-Base/16)**, **Multimodal Feature Fusion with Classical Fatigue Gating**, and a **Temporal Decision Layer** state machine to deliver accurate, real-time detection of driver fatigue, distraction, and microsleep with non-blocking acoustic alerts.

---

## System Architecture

```text
               +--------------------------------------------+
               |          Live Video / Camera Input         |
               +--------------------------------------------+
                                     |
                                     v
               +--------------------------------------------+
               |        MediaPipe FaceMesh (468 Lms)        |
               +--------------------------------------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
+---------------------------------+   +---------------------------------+
|  Classical Biometric Extraction |   |       Face ROI Extraction       |
|  - Eye Aspect Ratio (EAR)       |   |  - Bounding Box with Margin     |
|  - Mouth Aspect Ratio (MAR)     |   |  - Aspect-Preserving Resize     |
|  - Rolling PERCLOS (90 frames)  |   |  - Normalized Tensor (224x224)  |
|  - 3D Head Pose (solvePnP)      |   +---------------------------------+
|  - Dynamic Head Nod Rate        |                      |
+---------------------------------+                      v
                 |                    +---------------------------------+
                 |                    | Vision Transformer (ViT-Base)   |
                 |                    | - 4-Class Softmax Probabilities |
                 |                    +---------------------------------+
                 |                                       |
                 +-------------------+-------------------+
                                     |
                                     v
               +--------------------------------------------+
               |     Multimodal Feature Fusion Layer        |
               |  - Weighted Classical & ViT Evidence       |
               |  - Classical Fatigue Gating for ViT DROWSY |
               +--------------------------------------------+
                                     |
                                     v
               +--------------------------------------------+
               |         Temporal Decision Layer            |
               |  - State Persistence Counters              |
               |  - Normal Blink Safety Filter              |
               |  - Decoupled State Escalation              |
               +--------------------------------------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
+---------------------------------+   +---------------------------------+
|     Production OpenCV HUD       |   |    Non-Blocking Audio Alerts    |
|  - Telemetry Banner             |   |  - Pygame Daemon Thread         |
|  - Biometric Metric Bars        |   |  - Synthesized Alert Tones      |
|  - Head Pose Vector Indicator   |   |  - Cooldown Management          |
+---------------------------------+   +---------------------------------+
```

---

## Key Features

- **Hybrid Multimodal Detection**: Unifies deep spatial representation learning (ViT) with geometrically explainable classical facial biometrics.
- **Four-Class State Machine**: Resolves driver alertness into `ALERT`, `LOW_VIGILANCE`, `DROWSY`, and `MICROSLEEP`.
- **Classical Fatigue Gating**: ViT DROWSY classifications are mathematically gated against classical physical fatigue cues to eliminate spurious deep-learning false positives during alert states.
- **Decoupled State Escalation**: Prevents cumulative drowsiness counter increments from falsely triggering low vigilance alerts.
- **Calibrated MediaPipe Biometrics**: EAR and PERCLOS thresholds are tailored for MediaPipe FaceMesh geometry (`ear_threshold = 0.18`), preventing normal eye opening from being flagged as closed.
- **Dynamic Head Nod Detection**: Evaluates pitch change rate ($\Delta\text{pitch} / \Delta t < -8.0^\circ/\text{s}$) with downward-pitch gating and resting-pitch adaptation, distinguishing nodding off from static tilted head positions.
- **Natural Blink Resilience**: Temporal persistence filters out normal blinks (1–4 frames) to prevent alert fatigue.
- **Non-Blocking Acoustic Alerts**: Synthesized multi-frequency audio warnings managed via a separate daemon thread to ensure zero frame-processing latency overhead.
- **Real-Time Telemetry HUD**: Clean production overlay featuring live state banners, EAR/MAR/PERCLOS progress bars, head pose directional arrows, and measured FPS metrics.

---

## Detection Methodology & Biometrics

### 1. Eye Aspect Ratio (EAR)
Measures vertical eyelid separation relative to horizontal eye width:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
- **Threshold**: $\text{EAR} < 0.18$ indicates eyelid closure.
- **Left Eye Indices**: `[33, 160, 158, 133, 153, 144]`
- **Right Eye Indices**: `[362, 385, 387, 263, 373, 380]`

### 2. Mouth Aspect Ratio (MAR)
Detects wide mouth opening characteristic of yawning:
$$\text{MAR} = \frac{\|p_{39} - p_{181}\| + \|p_0 - p_{17}\| + \|p_{269} - p_{405}\|}{2 \cdot \|p_{61} - p_{291}\|}$$
- **Threshold**: $\text{MAR} > 0.60$ indicates yawning.
- **Mouth Indices**: `[61, 291, 39, 181, 0, 17, 269, 405]`

### 3. PERCLOS (Percentage of Eye Closure)
Evaluated as a rolling-window metric over 90 frames (~3.0 seconds at 30 FPS):
$$\text{PERCLOS} = \frac{1}{N} \sum_{k=1}^{N} \mathbb{I}(\text{EAR}_k < 0.18)$$
- **Threshold**: $\text{PERCLOS} > 0.15$ indicates sustained ocular fatigue.

### 4. 3D Head Pose & Dynamic Nodding
Uses 6 key 3D facial landmarks mapped to a canonical 3D model via OpenCV `solvePnP` with Levenberg-Marquardt refinement:
- **Landmarks**: Nose tip (`1`), Chin (`152`), Left eye corner (`33`), Right eye corner (`263`), Left mouth corner (`61`), Right mouth corner (`291`).
- **Yaw**: $|\text{yaw}| > 20.0^\circ$ indicates lateral distraction.
- **Pitch**: Evaluated relative to an adaptive resting pitch baseline.
- **Dynamic Nodding**: Triggered when downward pitch velocity is significant ($\Delta\text{pitch} / \Delta t < -8.0^\circ/\text{s}$) and head pitch is below resting baseline. Upward head motion is excluded.

### 5. Multimodal Feature Fusion
Combines ViT softmax probabilities $\mathbf{P}_{\text{ViT}}$ with normalized classical evidence $\mathbf{E}_{\text{classical}}$:
$$\mathbf{S} = w_{\text{vit}} \cdot \mathbf{P}_{\text{ViT}} + w_{\text{classical}} \cdot \mathbf{E}_{\text{classical}}$$
- **Fatigue Gating Rule**: If classical physical fatigue evidence is absent ($\text{Closure} < 0.35$ and $\text{Yawn} < 0.50$ and $\text{PERCLOS} < 0.30$ and $\text{Nod} < 0.40$), ViT DROWSY evidence is suppressed to zero, preserving high alert reliability.

---

## Driver Classification States

| State Index | State Name | Description | Trigger Conditions |
|---|---|---|---|
| **0** | `ALERT` | Fully attentive driver | Normal blink duration, eye open, head upright. |
| **1** | `LOW_VIGILANCE` | Early attention decline / distraction | Sustained head yaw ($> 20^\circ$) for $\ge 20$ frames or persistent mild inattention cues. |
| **2** | `DROWSY` | Dangerous cognitive fatigue | Combined fatigue evidence (yawning, low EAR, PERCLOS $> 0.15$, nodding) persisting for $\ge 60$ frames. |
| **3** | `MICROSLEEP` | Critical emergency: eyes shut | Continuous eye closure ($\text{EAR} < 0.18$) persisting for $\ge 10$ frames (~330 ms). Triggers immediate emergency alert. |

---

## Technology Stack

- **Computer Vision & Face Mesh**: MediaPipe 0.10.21, OpenCV 4.11.0, NumPy 1.26.4
- **Deep Learning**: PyTorch 2.14.0+cu130, Torchvision 0.29.0+cu130, Hugging Face Transformers 5.17.0
- **Audio Alert System**: Pygame 2.6.1
- **Testing & Verification**: Pytest 9.1.1
- **Hardware Platform Used for Validation**:
  - GPU: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 13.0)
  - CPU: Intel64 Family 6 Model 183
  - OS: Windows 11 (64-bit)

---

## Installation & Setup

### 1. Prerequisites
- Python 3.11 (tested on 3.11.9)
- Webcam (built-in or USB)
- Optional: CUDA-compatible NVIDIA GPU for real-time ViT acceleration

### 2. Clone and Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd DriverDrowsy

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### 3. GPU PyTorch Acceleration (Recommended)
If using an NVIDIA GPU, ensure CUDA-accelerated PyTorch is installed:
```bash
# For CUDA 12.1 / 12.4 / 13.0 environments
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Model Setup

Due to GitHub repository file size limits, pre-trained model checkpoint weights are not stored directly in this repository and are excluded via `.gitignore`.

### Checkpoint Options:
1. **Inference-Only Checkpoint (`vit_inference_only.pt`) [Recommended]**:
   - Size: **~327.38 MB**
   - Contains strictly the trained ViT model weights (optimizer and scheduler training states stripped).
   - Produces bit-for-bit identical inference output to the full checkpoint.
2. **Full Production Checkpoint (`vit_best_production.pt`)**:
   - Size: **~435.60 MB**
   - Contains full AdamW optimizer and training state.

### Setup Instructions:
1. Download the checkpoint:
   - **Download link**: *To be provided*
2. Create the target directory and place the checkpoint file:
   ```text
   models/
   └── checkpoints/
       └── vit_inference_only.pt
   ```
*(Note: If using `vit_best_production.pt`, place it in the same directory. The pipeline automatically searches for `vit_inference_only.pt` first and falls back to `vit_best_production.pt`.)*

---

## How to Run

### 1. Live Webcam Feed
```bash
.venv\Scripts\python main.py --source 0
```
- Press **`q`** to quit.
- Press **`r`** to reset state persistence counters.

### 2. Video File Input
```bash
.venv\Scripts\python main.py --source /path/to/driver_video.mp4
```

### 3. Headless Synthetic Smoke Test
Verifies end-to-end pipeline execution without requiring camera hardware:
```bash
.venv\Scripts\python main.py --synthetic --frames 100 --headless
```

### 4. Physical Webcam Benchmark
Executes a 500-frame hardware-synchronized latency benchmark on webcam index 0:
```bash
.venv\Scripts\python scripts/benchmark_phase3_webcam.py --source 0 --frames 500
```

---

## Automated Test Suite

The test suite validates EAR/MAR calculations, 3D head pose estimation, solvePnP decomposition, normal blink safety, ViT inference forward pass, multimodal feature fusion gating, and temporal state persistence:

```bash
.venv\Scripts\pytest tests/test_phase3.py tests/test_pipeline.py -v
```

**Status**: 63/63 tests passing.

---

## Performance Benchmarks

Measured on physical webcam index 0 (640x480 resolution) using an NVIDIA GeForce RTX 4050 Laptop GPU with explicit CUDA synchronization (`torch.cuda.synchronize()`):

### 1. Latency & Throughput Metrics (500 Post-Warmup Frames)

| Metric | Measured Value |
|---|---|
| **Face Detection Success Rate** | **99.80%** (499 / 500 frames) |
| **Mean Synchronized Latency** | **30.03 ms** |
| **Median Synchronized Latency** | **30.06 ms** |
| **95th Percentile (P95) Latency** | **44.68 ms** |
| **Maximum Pipeline Latency** | **47.26 ms** |
| **Synchronized Computational Throughput** | **38.05 FPS** (Mean) / **33.27 FPS** (Median) |
| **Application / HUD Loop Frame Rate** | **27.74 FPS** (Mean) / **24.75 FPS** (Median) |

> **Note on Frame Rates**:
> - **Computational Throughput (38.05 FPS)** measures the exact GPU/CPU processing execution time per frame.
> - **Application / HUD Loop FPS (27.74 FPS)** reflects end-to-end wall-clock display time (mean 39.38 ms), which includes physical camera sensor frame-acquisition blocking and OpenCV GUI rendering.

### 2. Component Latency Breakdown

| Pipeline Stage | Mean Execution Time | Percentage |
|---|---|---|
| **MediaPipe FaceMesh Landmark Detection** | 6.43 ms | 21.4% |
| **Face ROI Extraction & Normalization** | 2.07 ms | 6.9% |
| **ViT Inference (CUDA Synchronized)** | 20.12 ms | 67.0% |
| **Multimodal Feature Fusion** | 0.30 ms | 1.0% |
| **Temporal Decision Layer** | 0.06 ms | 0.2% |
| **Overhead, Head Pose & Pipeline IO** | 1.05 ms | 3.5% |
| **Total Mean Latency** | **30.03 ms** | **100.0%** |

Benchmark data and charts are preserved in `data/performance/`.

---

## Dataset & Training Information

The Vision Transformer was trained and evaluated on the **National Tsing Hua University (NTHU) Driver Drowsiness Dataset**:
- **Classes**: Mapped into 4 canonical states (`ALERT`, `LOW_VIGILANCE`, `DROWSY`, `MICROSLEEP`).
- **Data Splitting**: Stratified video-sequence grouping (~70% Train, ~15% Val, ~15% Test) ensuring zero temporal frame leakage across sequences.
- **Evaluation Characteristics**: Video sequence grouping ensures frames from the same video sequence do not cross splits. Full cross-subject independence remains an area for future work due to subject-specific recording conditions and non-uniform class distribution in the dataset.

### Standalone ViT Test Set Metrics
Evaluated on 10,232 test frames:
- **Accuracy**: 40.14%
- **Balanced Accuracy**: 21.58%
- **Macro-F1**: 19.75%
- **Weighted-F1**: 36.55%

> **Important Architectural Context**:
> Standalone ViT single-frame predictions exhibit class confusion when evaluated in isolation on static images (e.g., distinguishing a momentary voluntary blink from fatigue). This directly underscores the architectural necessity of **Phase 3 Multimodal Feature Fusion** and the **Temporal Decision Layer**, which gate ViT outputs with continuous physical biometric signals (EAR, PERCLOS, head nod velocity) across time.

---

## Project Structure

```text
DriverDrowsy/
├── README.md                              # Project documentation
├── requirements.txt                      # Project dependency specification
├── .gitignore                            # Exclusions for models, venvs, datasets, caches
├── main.py                               # Application entry point (webcam / video / synthetic)
│
├── src/                                  # Core detection pipeline
│   ├── detector.py                       # Orchestrator (Frame -> landmarks -> ROI -> ViT -> Fusion -> Temporal)
│   ├── feature_fusion.py                 # Multimodal fusion with classical fatigue gating
│   ├── temporal_fusion.py                # Temporal decision state machine & persistence counters
│   ├── vit_inference.py                  # Real-time ViT engine with CUDA synchronization
│   ├── landmark_detector.py              # MediaPipe FaceMesh wrapper (468 landmarks)
│   ├── head_pose.py                      # 3D Head Pose (solvePnP) & dynamic downward nod detection
│   └── dataset.py                        # Dataset parsers and ROI preprocessing utilities
│
├── models/                               # Models and weights
│   ├── baseline.py                       # Classical EAR, MAR, PERCLOS calculation routines
│   ├── vit_classifier.py                 # HuggingFace ViT architecture with 4-class classifier
│   └── checkpoints/                      # Model weights directory (git-ignored)
│
├── alerts/                               # Acoustic alert system
│   └── alert.py                          # Multi-frequency audio alert manager (daemon thread)
│
├── utils/                                # Application utilities
│   ├── config.py                         # Centralized configuration (thresholds, indices, persistence)
│   ├── visualizer.py                     # Production OpenCV HUD renderer
│   └── logger.py                         # Structured logging utility
│
├── scripts/                              # Benchmarking and training utilities
│   ├── benchmark_phase3_webcam.py        # Physical webcam 500-frame benchmark runner
│   ├── benchmark_phase3.py               # Pipeline runtime benchmark script
│   ├── plot_phase3_benchmark.py          # Benchmark graph generator
│   ├── preprocess_nthu.py                # Dataset extraction and sequence splitting script
│   └── train_vit.py                      # Vision Transformer training pipeline
│
├── tests/                                # Automated test suite (63 tests)
│   ├── test_phase3.py                    # Comprehensive tests for ViT, fusion, gating, temporal logic
│   ├── test_pipeline.py                  # End-to-end detector integration tests
│   ├── test_baseline.py                  # Unit tests for EAR, MAR, PERCLOS, head pose
│   └── test_vit.py                       # Unit tests for ViT forward pass and loss
│
└── data/                                 # Evaluation and benchmark artifacts
    ├── performance/                      # Authoritative benchmark results
    │   ├── phase3_webcam_benchmark.csv
    │   ├── phase3_webcam_benchmark_report.txt
    │   ├── webcam_component_latency.png
    │   ├── webcam_fps_comparison.png
    │   └── webcam_latency_over_frames.png
    └── production_evaluation_summary.json # ViT evaluation summary metrics
```

---

## Limitations

1. **Extreme Head Pose Angles**: Large lateral head rotations ($|\text{yaw}| > 45^\circ$) cause partial landmark occlusion, temporarily reducing facial biometric accuracy.
2. **Adverse Lighting Conditions**: Standard RGB webcams suffer under severe low-light conditions; production deployment requires an active Near-Infrared (NIR) camera.
3. **Eyewear Obstruction**: Polarized sunglasses or heavy dark frames can impede accurate eye landmark localization and iris tracking.
4. **Static Frame Representation**: Single-frame ViT inference does not intrinsically model temporal dynamics, necessitating the downstream temporal state machine.

---

## Future Improvements

- **Sequential Deep Modeling**: Incorporating temporal architectures (Video-MAE, Temporal Transformers, or Bi-LSTM) to model frame-to-frame motion directly in latent space.
- **Edge Acceleration**: Compiling the ViT backbone to TensorRT or ONNX Runtime with INT8 quantization for embedded deployment (e.g., NVIDIA Jetson, Raspberry Pi 5).
- **Driver Auto-Calibration**: Introducing a 10-second initial calibration routine at startup to establish personalized baseline EAR and resting head pitch for individual drivers.
- **NIR Sensor Integration**: Supporting Near-Infrared illumination pipelines for robust night-driving operation.

---

## Academic Context

Developed as an academic engineering project for **VIT Bhopal University**.
