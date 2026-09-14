# Driver Drowsiness Detection System

Real-time, non-intrusive driver drowsiness detection system developed as an academic engineering project for **VIT Bhopal University**.

The system monitors a driver's facial biometric signals via a standard camera and detects fatigue, inattention, and sleep onset using MediaPipe FaceMesh landmarks, classical computer vision indicators, temporal persistence logic, and non-blocking acoustic alerts.

---

## System Architecture

```text
drowsiness-detection/
├── alerts/
│   ├── __init__.py
│   └── alert.py               # Non-blocking audio alert manager with synthesized tones
├── data/
│   ├── raw/                   # Raw dataset storage
│   ├── processed/             # Preprocessed facial ROIs / frames
│   └── splits/                # Train / Val / Test subject-wise split metadata
├── models/
│   ├── __init__.py
│   ├── checkpoints/           # Trained model weights
│   └── baseline.py            # EAR, MAR, PERCLOS calculations and temporal state machine
├── src/
│   ├── __init__.py
│   ├── landmark_detector.py   # MediaPipe FaceMesh wrapper (468 landmarks, NO dlib)
│   ├── head_pose.py           # 3D pose estimation (solvePnP: pitch, yaw, roll)
│   └── detector.py            # Main pipeline orchestrator (decoupled from GUI)
├── ui/
│   └── __init__.py            # Future Tkinter GUI application (Phase 4)
├── utils/
│   ├── __init__.py
│   ├── config.py              # Central configuration (all parameters, indices, thresholds)
│   ├── logger.py              # Structured logging (zero raw print statements)
│   └── visualizer.py          # OpenCV HUD telemetry visualizer
├── tests/
│   ├── __init__.py
│   ├── test_baseline.py       # Unit tests for EAR, MAR, PERCLOS, head pose, state logic
│   └── test_pipeline.py      # Integration tests for pipeline execution and decoupling
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── main.py                    # Application CLI entry point
```

---

## Core Signals & Classical Baseline (Phase 1)

### 1. Eye Aspect Ratio (EAR)
Measures eyelid opening distance relative to horizontal eye width:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
* **Left eye indices**: `[33, 160, 158, 133, 153, 144]`
* **Right eye indices**: `[362, 385, 387, 263, 373, 380]`
* **Threshold**: `EAR < 0.25` (indicates eye closure)

### 2. Mouth Aspect Ratio (MAR)
Detects yawning and mouth opening:
$$\text{MAR} = \frac{\|p_{39} - p_{181}\| + \|p_0 - p_{17}\| + \|p_{269} - p_{405}\|}{2 \cdot \|p_{61} - p_{291}\|}$$
* **Mouth indices**: `[61, 291, 39, 181, 0, 17, 269, 405]`
* **Threshold**: `MAR > 0.60` (indicates wide open mouth / yawning)

### 3. PERCLOS (Percentage of Eye Closure)
* Evaluated as a **rolling-window metric** over $N$ frames (`config.temporal.perclos_window_size = 90`, ~3 seconds at 30 FPS).
* Represents the proportion of frames in the window where `EAR < 0.25`.
* **Threshold**: `PERCLOS > 0.15` (fatigue indicator).

### 4. 3D Head Pose Estimation
* Uses 6 key facial landmarks mapped to a canonical 3D facial model via OpenCV `solvePnP` and `RQDecomp3x3`:
  * Nose tip (`1`), Chin (`152`), Left eye corner (`33`), Right eye corner (`263`), Left mouth corner (`61`), Right mouth corner (`291`).
* **Yaw indicator**: `|yaw| > 20.0°` (lateral distraction / looking away).
* **Pitch indicator**: `|pitch| > 15.0°` (head dropping / nodding off).

### 5. Project States & Temporal Logic
```text
0 = ALERT
1 = LOW_VIGILANCE
2 = DROWSY
3 = MICROSLEEP
```
* **Normal Blinking Safety**: Normal blinks (1–4 frames) do NOT trigger `LOW_VIGILANCE` or `DROWSY`.
* **Microsleep Trigger**: Sustained abnormal eye closure where continuous eye closure $\ge 10$ frames (`MICROSLEEP_PERSISTENCE`).
* **Drowsy Trigger**: Cumulative fatigue cues (sustained yawning, prolonged low EAR, head nodding) persisting for $\ge 30$ frames (`DROWSY_PERSISTENCE`).
* **Alert Recovery**: 15 consecutive frames of normal signals restores the driver state to `ALERT`.

---

## Installation & Setup

### Requirements
* Linux / Windows / macOS
* Python 3.10+ (tested on Python 3.12)
* Webcam or video source

```bash
# Clone / navigate to project directory
cd /path/to/DriverDrowsy

# Create virtual environment using uv or venv
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

---

## Running the System

### 1. Live Webcam Feed
```bash
.venv/bin/python main.py --source 0
```
* Press `q` to quit.
* Press `r` to reset counters.

### 2. Video File Input
```bash
.venv/bin/python main.py --source /path/to/video.mp4
```

### 3. Headless Synthetic Benchmark (Offline / CI Verification)
Runs the complete pipeline on synthetic frames without requiring a webcam or display window:
```bash
.venv/bin/python main.py --synthetic --frames 100 --headless
```

### 4. Headless Webcam Mode
```bash
.venv/bin/python main.py --source 0 --frames 60 --headless
```

---

## Running the Automated Test Suite

The test suite validates EAR, MAR, PERCLOS rolling window calculations, head pose estimation, false positive safety (normal blinks remaining ALERT), temporal state transitions, detector/visualizer decoupling, and verifies the absolute absence of dlib.

```bash
.venv/bin/pytest tests/ -v
```

---

## Engineering Standards
* **No dlib**: Strictly uses Google MediaPipe FaceMesh.
* **No Magic Numbers**: All constants, indices, persistence counts, and thresholds are centralized in `utils/config.py`.
* **Structured Logging**: Uses standard library `logging` with timestamps and log levels. No raw `print()` statements.
* **Pipeline Decoupling**: Detector operates independently of visualizer (`Frame -> Detector -> DetectionResult -> Visualizer -> Annotated Frame`).
* **Non-Blocking Alerts**: Audio playback runs on a background daemon worker thread with cooldown management.
* **Actual FPS**: Frame rates are measured from actual microsecond frame timestamps.
