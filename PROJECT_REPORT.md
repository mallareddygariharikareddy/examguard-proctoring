# ExamGuard Proctoring System - Project Report

## 1. Introduction

ExamGuard is a lightweight desktop-based exam proctoring prototype designed to monitor a student during an online examination using a local webcam. The system runs entirely on the user's machine and does not require cloud processing during the exam. This makes it suitable for privacy-conscious environments where video data should not be uploaded to external servers.

The project focuses on three main monitoring tasks: detecting whether a face is present, identifying if multiple people are visible, and observing suspicious motion patterns. The system combines computer vision, rule-based alerting, and local session logging to provide a practical proctoring workflow. A Tkinter dashboard displays the live camera feed and compact status indicators, while session events are saved locally for later review.

The latest version of the system improves reliability by replacing the older Haar-only face detection pipeline with MediaPipe's BlazeFace model. Haar cascades are still kept as a fallback, but MediaPipe is now the primary detector. This improves face detection accuracy, especially for normal webcam conditions where lighting, head angle, and frame quality can vary.

## 2. Objectives

The main objectives of this project are:

- To build a local proctoring application that can monitor webcam activity in real time.
- To detect key exam violations such as no face, multiple faces, looking away, and excessive movement.
- To reduce false alerts caused by small movements or short detection glitches.
- To maintain a clean and distraction-free user interface.
- To generate local logs and reports for post-exam review.
- To keep the system modular so that future detection models such as YOLO can be added easily.

## 3. System Architecture

ExamGuard follows a modular architecture. Each major responsibility is separated into its own component:

```text
main.py
  -> ui.dashboard.ProctoringDashboard
      -> core.camera.CameraManager
      -> core.face_detector.FaceDetector
      -> core.motion_detector.MotionDetector
      -> core.alert_engine.AlertEngine
      -> ui.report_generator.ReportGenerator
```

The application starts from `main.py`, which creates the Tkinter root window and initializes the dashboard. The dashboard controls the camera, starts the processing loop, updates the interface, and coordinates logging.

The runtime pipeline works as follows:

```text
Webcam frame
  -> frame resize
  -> face detection using MediaPipe
  -> motion detection using OpenCV MOG2
  -> alert evaluation
  -> UI status update
  -> CSV/report/screenshot logging
```

The camera module continuously captures frames in a background thread. The dashboard processing loop reads the latest frame, runs motion detection every frame, and runs face detection periodically to reduce CPU load. The results are passed to the alert engine, which decides whether a violation should be recorded.

## 4. Core Modules

### Camera Manager

`core/camera.py` handles webcam access using OpenCV. It opens the configured camera index, starts a background capture thread, resizes frames to the configured resolution, and provides thread-safe frame reads. The latest version also waits briefly for the first valid frame, improving startup reliability.

### Face Detector

`core/face_detector.py` is responsible for face analysis. The detector now uses MediaPipe BlazeFace as the primary backend. The model file is stored locally at:

```text
models/blaze_face_short_range.tflite
```

If MediaPipe is unavailable or the model cannot be loaded, the system can fall back to OpenCV Haar cascades. The face detector returns a `FaceAnalysis` object containing face count, primary face, eye visibility, gaze estimate, and status.

This module helps detect:

- no face visible
- one face visible
- multiple faces visible
- possible looking-away behavior

### Motion Detector

`core/motion_detector.py` uses OpenCV's MOG2 background subtraction to identify movement in the frame. It filters noise, classifies motion by size, and maps movement into head, body, and seat zones. Motion is classified as `none`, `low`, `medium`, or `high`.

The motion thresholds were tuned to reduce false alerts from tiny movements. This prevents normal behavior such as slight hand movement or posture adjustment from immediately increasing the risk score.

### Alert Engine

`core/alert_engine.py` converts detection results into proctoring events. The alert engine uses violation scores, cooldowns, and sustained-frame confirmation. This means an alert is only fired when a suspicious condition remains true for several frames.

This design reduces false positives. For example, if another face appears for only one frame due to detection noise, the system will not immediately log a high-severity violation. The reset score feature clears the current score, cooldowns, and confirmation streaks.

### Dashboard

`ui/dashboard.py` provides the Tkinter interface. The interface was simplified to avoid distracting visual elements. The live camera feed is now clean, with no face boxes, motion rectangles, risk score card, or alert log section. Instead, the dashboard shows compact detection indicators and session statistics.

The interface includes:

- Start Session
- Pause / Resume
- Reset Score
- Save Report
- Quit
- face, eye, gaze, and motion indicators
- session alert counters
- one-line status messages

### Report Generator

`ui/report_generator.py` saves session data locally. It creates CSV logs, optional screenshots, and a plain-text summary report. These artifacts are stored under the `logs/` directory, which is ignored by Git to avoid committing private session data.

## 5. Technologies Used

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Tkinter | Desktop GUI |
| OpenCV | Camera access and motion detection |
| MediaPipe | Face detection |
| NumPy | Frame and image array handling |
| Pillow | Converting frames for Tkinter display |
| Pandas | Session data handling |
| fpdf2 | PDF reporting support for future extension |

## 6. Testing and Validation

The project includes a test suite in `test_system.py`. It validates imports, dashboard source structure, motion detection behavior, face detection behavior, alert engine logic, report generation, integrated pipeline execution, and camera lifecycle.

The latest test result is:

```text
PASSED : 39
FAILED : 0
TOTAL  : 39
```

This confirms that the current implementation is stable after the MediaPipe integration, UI cleanup, motion sensitivity tuning, reset score verification, and report/architecture updates.

## 7. Conclusion

ExamGuard demonstrates a practical local proctoring workflow using computer vision and rule-based analysis. The system captures webcam frames, detects faces and motion, evaluates suspicious activity, and stores evidence locally. The latest updates make the application more reliable and user-friendly by integrating MediaPipe face detection, reducing false motion alerts, simplifying the interface, and improving reset/report behavior.

Overall, the project provides a strong foundation for a privacy-preserving exam monitoring tool. With future additions such as object detection and identity verification, it can be extended into a more complete proctoring solution.
