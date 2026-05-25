# ExamGuard — Motion Detection Proctoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-orange)
![License](https://img.shields.io/badge/License-MIT-purple)
![Tests](https://img.shields.io/badge/Tests-39%2F39%20Passing-brightgreen)

**A lightweight, privacy-preserving exam proctoring prototype using computer vision.**  
Runs entirely locally — no cloud, no data upload, no internet required during exams.

</div>

---

## 📸 Screenshots

> Launch the app and click **▶ Start Session** to activate the live webcam feed with overlays.

| Dashboard (idle) | Live Detection |
|:---:|:---:|
| Name dialog → dark dashboard with risk panel | Green face box, motion contours, live alert log |

---

## ✨ Features

| Feature | Technology | Description |
|---------|-----------|-------------|
| **Motion Detection** | MOG2 (Mixture of Gaussians) | Detects hand/body movements using adaptive background subtraction |
| **Face Detection** | Haar Cascade (OpenCV) | Counts faces, detects gaze direction, monitors eye visibility |
| **Zone Analysis** | Contour + position mapping | Classifies motion into Head / Body / Seat zones |
| **Risk Scoring** | Custom alert engine | Weighted violation scoring with per-type cooldown timers |
| **Alert Logging** | CSV + screenshots | Every flagged event is timestamped and saved |
| **Session Reports** | Plain text summary | End-of-exam report with total alerts, duration, and final risk level |
| **Dark GUI** | Tkinter (premium theme) | Live camera feed embedded in a styled dashboard |

---

## 🧠 How It Works

```
Webcam Frame
    │
    ├──► MOG2 Background Subtractor ──► Motion regions (zone + magnitude)
    │
    ├──► Haar Cascade Face Detector ──► Face count, eye count, gaze
    │
    └──► Alert Engine ─────────────────► Risk score + violation events
                │
                ├──► Tkinter Dashboard (live HUD overlay)
                ├──► CSV Session Log
                └──► Screenshot of flagged frames
```

### Violation Types & Weights

| Violation | Score | Cooldown |
|-----------|-------|----------|
| No face detected | +3 | 2 sec |
| Multiple persons | +10 | 5 sec |
| Looking away (no eyes) | +2 | 1 sec |
| Suspicious hand movement | +2 | 2 sec |
| Excessive body movement | +5 | 3 sec |
| Student left seat | +8 | 5 sec |

### Risk Levels

| Score | Level | Color |
|-------|-------|-------|
| 0–10 | 🟢 LOW | Green |
| 11–30 | 🟡 MEDIUM | Yellow |
| 31–60 | 🟠 HIGH | Orange |
| 61+ | 🔴 CRITICAL | Red |

> Score **decays by 1 point every 5 seconds** of clean behaviour.

---

## 🗂️ Project Structure

```
examguard-proctoring/
│
├── main.py                  # Entry point — run this
├── requirements.txt         # Python dependencies
├── test_system.py           # Test suite (39 tests)
│
├── config/
│   └── settings.py          # All tunable parameters
│
├── core/
│   ├── camera.py            # Thread-safe webcam manager
│   ├── motion_detector.py   # MOG2 background subtraction
│   ├── face_detector.py     # Haar cascade face + eye + gaze
│   └── alert_engine.py      # Scoring, cooldowns, violations
│
└── ui/
    ├── dashboard.py         # Tkinter dark-theme GUI
    └── report_generator.py  # CSV log + screenshot saver
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- A working webcam

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/examguard-proctoring.git
cd examguard-proctoring
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python main.py
```

### 4. (Optional) Run tests
```bash
python test_system.py
```

---

## ⚙️ Configuration

All parameters are in [`config/settings.py`](config/settings.py). Key settings:

```python
# Camera
CAMERA_INDEX = 0          # Change if you have multiple cameras

# Motion sensitivity (lower = more sensitive)
MOG2_VAR_THRESHOLD = 50

# Minimum contour area to count as motion
MOTION_MIN_AREA_SMALL  = 500
MOTION_MIN_AREA_MEDIUM = 3000
MOTION_MIN_AREA_LARGE  = 12000

# Screenshot saving
SAVE_SCREENSHOTS     = True
SCREENSHOT_MIN_SCORE = 10    # Only save when risk score >= this

# Score decay
SCORE_DECAY_INTERVAL = 5     # seconds
SCORE_DECAY_AMOUNT   = 1
```

---

## 🖥️ GUI Controls

| Button | Action |
|--------|--------|
| **▶ Start Session** | Opens webcam and begins proctoring |
| **⏸ Pause** | Freezes detection (for break time) |
| **🔄 Reset Score** | Clears the risk score to 0 |
| **💾 Save Report** | Writes session summary to `logs/` |
| **✖ Quit** | Auto-saves report and closes |

---

## 📁 Session Output

All session data is written to the `logs/` directory (git-ignored):

```
logs/
├── session_20260525_143000.csv       # Timestamped alert log
├── summary_20260525_143000.txt       # Human-readable report
└── screenshots/
    └── 20260525_143000/
        ├── no_face_143012_000001.jpg
        └── multiple_faces_143045_000002.jpg
```

---

## 🔬 Test Suite

```bash
python test_system.py
```

Covers 39 tests across 8 modules:

| Module | Tests |
|--------|-------|
| Imports & dependencies | 10 |
| Dashboard source analysis | 3 |
| Motion detector (MOG2) | 5 |
| Face detector (Haar) | 5 |
| Alert engine | 8 |
| Report generator | 3 |
| Integrated pipeline | 3 |
| Camera lifecycle | 2 |

---

## 🗺️ Roadmap

- [ ] **YOLOv8 phone/book detection** (object detection module)
- [ ] **Sound alerts** for CRITICAL risk level
- [ ] **Live risk score graph** over session timeline
- [ ] **PDF report** with charts
- [ ] **Multi-student mode** (multiple camera feeds)
- [ ] **Face recognition** (verify student identity at start)

---

## 🛠️ Tech Stack

| Library | Purpose | Version |
|---------|---------|---------|
| `opencv-python` | Core CV, camera, Haar cascades | ≥ 4.8 |
| `numpy` | Frame array operations | ≥ 1.24 |
| `Pillow` | Frame → Tkinter image conversion | ≥ 10.0 |
| `pandas` | Session data handling | ≥ 2.0 |
| `fpdf2` | PDF report generation (future) | ≥ 2.7 |
| `tkinter` | GUI framework (stdlib) | Built-in |

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built as a prototype for exam proctoring research using lightweight, privacy-preserving computer vision.
