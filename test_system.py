"""
test_system.py — Comprehensive test suite for ExamGuard Proctoring System.
Tests every module independently then as an integrated pipeline.
Run: python test_system.py
"""

import sys
import os
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS  = "  [PASS]"
FAIL  = "  [FAIL]"
WARN  = "  [WARN]"
INFO  = "  [INFO]"
SEP   = "-" * 60

results = []

def test(name, fn):
    print(f"\n  Testing: {name}")
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, "PASS", ""))
    except AssertionError as e:
        msg = str(e)
        print(f"{FAIL} {name}: {msg}")
        results.append((name, "FAIL", msg))
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"{FAIL} {name}: {msg}")
        traceback.print_exc()
        results.append((name, "FAIL", msg))

# ─── helpers ─────────────────────────────────────────────────────────────────

def blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)

def face_frame():
    """Frame with a rough skin-tone rectangle to help Haar cascade."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 40
    # skin-tone patch in centre — not a real face but good for testing contours
    frame[160:320, 250:390] = [150, 120, 100]
    return frame

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  EXAMGUARD — SYSTEM TEST SUITE")
print("=" * 60)

# ─── 1. IMPORTS ───────────────────────────────────────────────
print(f"\n{SEP}")
print("  MODULE 1: IMPORTS")
print(SEP)

def test_import_cv2():
    import cv2
    assert cv2.__version__, "cv2 version missing"

def test_import_numpy():
    import numpy as np
    assert np.__version__

def test_import_pil():
    from PIL import Image
    assert Image.__version__

def test_import_config():
    from config import settings
    assert settings.FRAME_WIDTH == 640
    assert settings.FRAME_HEIGHT == 480
    assert len(settings.VIOLATIONS) > 0
    assert len(settings.RISK_LEVELS) > 0

def test_import_camera():
    from core.camera import CameraManager
    cam = CameraManager()
    assert cam is not None

def test_import_motion():
    from core.motion_detector import MotionDetector, MotionRegion
    md = MotionDetector()
    assert md is not None

def test_import_face():
    from core.face_detector import FaceDetector, FaceAnalysis, FaceResult
    fd = FaceDetector()
    assert fd is not None
    # Test FaceAnalysis default construction
    fa = FaceAnalysis()
    assert fa.face_count == 0
    assert fa.status == "ok"

def test_import_alert():
    from core.alert_engine import AlertEngine, AlertEvent
    ae = AlertEngine()
    assert ae.score == 0

def test_import_report():
    from ui.report_generator import ReportGenerator
    rg = ReportGenerator("Test Student")
    assert rg is not None

def test_import_dashboard():
    # Test that all names used in dashboard are importable
    from core.face_detector import FaceAnalysis  # this MUST work for dashboard._processing_loop
    fa = FaceAnalysis()
    assert fa is not None

test("cv2 importable",           test_import_cv2)
test("numpy importable",         test_import_numpy)
test("Pillow importable",        test_import_pil)
test("config.settings valid",    test_import_config)
test("CameraManager importable", test_import_camera)
test("MotionDetector importable",test_import_motion)
test("FaceDetector + FaceAnalysis importable", test_import_face)
test("AlertEngine importable",   test_import_alert)
test("ReportGenerator importable", test_import_report)
test("Dashboard FaceAnalysis dependency", test_import_dashboard)

# ─── 2. DASHBOARD IMPORTS CHECK ───────────────────────────────
print(f"\n{SEP}")
print("  MODULE 2: DASHBOARD SOURCE ANALYSIS")
print(SEP)

def test_dashboard_has_faceanalysis_import():
    """FaceAnalysis must be imported in dashboard.py for the processing loop."""
    with open("ui/dashboard.py", "r", encoding="utf-8") as f:
        src = f.read()
    assert "from core.face_detector import" in src and "FaceAnalysis" in src, \
        "FaceAnalysis NOT imported in dashboard.py -- will crash at runtime!"

def test_dashboard_no_double_ctrl():
    """ctrl frame should only be defined once."""
    with open("ui/dashboard.py", "r", encoding="utf-8") as f:
        src = f.read()
    count = src.count("ctrl = tk.Frame(root")
    assert count == 1, f"ctrl Frame defined {count} times (should be 1)"

def test_alert_engine_no_bad_import():
    with open("core/alert_engine.py", "r", encoding="utf-8") as f:
        src = f.read()
    assert "from core.motion_detector import FaceAnalysis" not in src, \
        "Bad import: FaceAnalysis from motion_detector in alert_engine"

test("dashboard.py has FaceAnalysis import",    test_dashboard_has_faceanalysis_import)
test("dashboard.py ctrl frame defined once",    test_dashboard_no_double_ctrl)
test("alert_engine.py no bad FaceAnalysis import", test_alert_engine_no_bad_import)

# ─── 3. MOTION DETECTOR ───────────────────────────────────────
print(f"\n{SEP}")
print("  MODULE 3: MOTION DETECTOR (MOG2)")
print(SEP)

def test_motion_blank_frame():
    from core.motion_detector import MotionDetector
    md = MotionDetector()
    frame = blank_frame()
    regions, mask, level = md.process(frame)
    assert isinstance(regions, list)
    assert mask is not None
    assert level in ("none", "low", "medium", "high")

def test_motion_detects_change():
    from core.motion_detector import MotionDetector
    md = MotionDetector()
    # Feed 30 blank frames to build background model
    blank = blank_frame()
    for _ in range(30):
        md.process(blank)
    # Now inject a bright white rectangle — should be detected as motion
    bright = blank.copy()
    bright[100:300, 100:400] = 255
    regions, mask, level = md.process(bright)
    assert level in ("medium", "high"), \
        f"Expected motion after big pixel change, got level='{level}'"

def test_motion_zone_classification():
    from core.motion_detector import MotionDetector
    md = MotionDetector(frame_height=480)
    # cy at top third = head zone
    zone = md._classify_zone(cy=100, frame_h=480)
    assert zone == "head", f"Expected 'head', got '{zone}'"
    zone = md._classify_zone(cy=280, frame_h=480)
    assert zone == "body", f"Expected 'body', got '{zone}'"
    zone = md._classify_zone(cy=420, frame_h=480)
    assert zone == "seat", f"Expected 'seat', got '{zone}'"

def test_motion_magnitude():
    from core.motion_detector import MotionDetector
    md = MotionDetector()
    assert md._classify_magnitude(100)   == "small"
    assert md._classify_magnitude(5000)  == "medium"
    assert md._classify_magnitude(15000) == "large"

def test_motion_reset():
    from core.motion_detector import MotionDetector
    md = MotionDetector()
    for _ in range(10):
        md.process(blank_frame())
    md.reset()  # should not raise
    # After reset, MOG2 needs a few warm-up frames before stabilising
    for _ in range(5):
        md.process(blank_frame())
    regions, mask, level = md.process(blank_frame())
    assert level in ("none", "low"), \
        f"After warm-up post-reset, blank frame should be none/low, got: {level}"

test("MotionDetector processes blank frame",   test_motion_blank_frame)
test("MotionDetector detects large change",    test_motion_detects_change)
test("MotionDetector zone classification",     test_motion_zone_classification)
test("MotionDetector magnitude thresholds",    test_motion_magnitude)
test("MotionDetector reset works",             test_motion_reset)

# ─── 4. FACE DETECTOR ─────────────────────────────────────────
print(f"\n{SEP}")
print("  MODULE 4: FACE DETECTOR (Haar Cascade)")
print(SEP)

def test_face_no_face():
    from core.face_detector import FaceDetector
    fd = FaceDetector()
    fa = fd.process(blank_frame())
    assert fa.face_count == 0
    assert fa.status == "no_face"

def test_face_analysis_fields():
    from core.face_detector import FaceDetector, FaceAnalysis
    fd = FaceDetector()
    fa = fd.process(blank_frame())
    assert hasattr(fa, "faces")
    assert hasattr(fa, "face_count")
    assert hasattr(fa, "primary")
    assert hasattr(fa, "eyes_visible")
    assert hasattr(fa, "gaze")
    assert hasattr(fa, "status")

def test_face_default_construction():
    from core.face_detector import FaceAnalysis
    fa = FaceAnalysis()
    assert fa.face_count == 0
    assert fa.faces == []
    assert fa.primary is None
    assert fa.eyes_visible == False
    assert fa.gaze == "unknown"
    assert fa.status == "ok"

def test_face_gaze_estimate():
    from core.face_detector import FaceDetector
    fd = FaceDetector()
    # face_x=10, face_w=100, frame_w=640 → face centre = 60/640 = 0.09 < 0.30 → "right"
    gaze = fd._estimate_gaze(face_x=10, face_w=100, frame_w=640, eye_count=2)
    assert gaze == "right", f"Expected 'right', got '{gaze}'"
    # face centred
    gaze = fd._estimate_gaze(face_x=250, face_w=140, frame_w=640, eye_count=2)
    assert gaze == "centre", f"Expected 'centre', got '{gaze}'"
    # no eyes → away
    gaze = fd._estimate_gaze(face_x=250, face_w=140, frame_w=640, eye_count=0)
    assert gaze == "away"

def test_face_streak_counter():
    from core.face_detector import FaceDetector
    fd = FaceDetector()
    for _ in range(5):
        fd.process(blank_frame())
    assert fd.no_face_streak == 5, \
        f"Expected streak=5, got {fd.no_face_streak}"

test("FaceDetector no face on blank frame",  test_face_no_face)
test("FaceAnalysis has all fields",          test_face_analysis_fields)
test("FaceAnalysis default construction",    test_face_default_construction)
test("Gaze estimation logic",               test_face_gaze_estimate)
test("No-face streak counter",              test_face_streak_counter)

# ─── 5. ALERT ENGINE ──────────────────────────────────────────
print(f"\n{SEP}")
print("  MODULE 5: ALERT ENGINE")
print(SEP)

def test_alert_initial_state():
    from core.alert_engine import AlertEngine
    ae = AlertEngine()
    assert ae.score == 0
    assert ae.all_events == []

def test_alert_no_face_fires():
    from core.alert_engine import AlertEngine
    from core.face_detector import FaceAnalysis
    ae = AlertEngine()
    fa = FaceAnalysis(face_count=0, status="no_face")
    events = ae.evaluate(fa, "none", [])
    assert len(events) >= 1
    assert any(e.violation_key == "no_face" for e in events)
    assert ae.score > 0

def test_alert_multiple_faces():
    from core.alert_engine import AlertEngine
    from core.face_detector import FaceAnalysis, FaceResult
    ae = AlertEngine()
    r1 = FaceResult(bbox=(0,0,100,100), eye_count=2, gaze="centre", area=10000)
    r2 = FaceResult(bbox=(200,0,100,100), eye_count=2, gaze="centre", area=10000)
    fa = FaceAnalysis(faces=[r1, r2], face_count=2,
                      primary=r1, eyes_visible=True, gaze="centre",
                      status="multiple")
    events = ae.evaluate(fa, "none", [])
    assert any(e.violation_key == "multiple_faces" for e in events), \
        "multiple_faces violation not fired"

def test_alert_cooldown():
    from core.alert_engine import AlertEngine
    from core.face_detector import FaceAnalysis
    ae = AlertEngine()
    fa = FaceAnalysis(face_count=0, status="no_face")
    e1 = ae.evaluate(fa, "none", [])
    e2 = ae.evaluate(fa, "none", [])
    # Second call should be empty (cooldown active)
    assert len(e2) == 0, "Cooldown not working — duplicate events fired"

def test_alert_score_accumulates():
    from core.alert_engine import AlertEngine
    from core.face_detector import FaceAnalysis
    ae = AlertEngine()
    fa = FaceAnalysis(face_count=0, status="no_face")
    ae.evaluate(fa, "none", [])
    score_after_1 = ae.score
    # Wait for cooldown to expire (no_face cooldown = 2s)
    time.sleep(2.1)
    ae.evaluate(fa, "none", [])
    assert ae.score > score_after_1, "Score should increase after cooldown expires"

def test_alert_reset_score():
    from core.alert_engine import AlertEngine
    from core.face_detector import FaceAnalysis
    ae = AlertEngine()
    fa = FaceAnalysis(face_count=0, status="no_face")
    ae.evaluate(fa, "none", [])
    assert ae.score > 0
    ae.reset_score()
    assert ae.score == 0

def test_alert_risk_levels():
    from core.alert_engine import AlertEngine
    ae = AlertEngine()
    ae.score = 5
    assert ae.get_risk_level().label == "LOW"
    ae.score = 20
    assert ae.get_risk_level().label == "MEDIUM"
    ae.score = 50
    assert ae.get_risk_level().label == "HIGH"
    ae.score = 70
    assert ae.get_risk_level().label == "CRITICAL"

def test_alert_high_motion_fires():
    from core.alert_engine import AlertEngine
    from core.face_detector import FaceAnalysis
    ae = AlertEngine()
    fa = FaceAnalysis(face_count=1, status="ok", eyes_visible=True)
    events = ae.evaluate(fa, "high", [])
    assert any(e.violation_key == "excess_motion" for e in events), \
        "excess_motion not fired for high motion"

test("AlertEngine initial state",         test_alert_initial_state)
test("AlertEngine fires no_face",         test_alert_no_face_fires)
test("AlertEngine fires multiple_faces",  test_alert_multiple_faces)
test("AlertEngine cooldown works",        test_alert_cooldown)
test("AlertEngine score accumulates",     test_alert_score_accumulates)
test("AlertEngine reset_score works",     test_alert_reset_score)
test("AlertEngine risk level thresholds", test_alert_risk_levels)
test("AlertEngine fires excess_motion",   test_alert_high_motion_fires)

# ─── 6. REPORT GENERATOR ──────────────────────────────────────
print(f"\n{SEP}")
print("  MODULE 6: REPORT GENERATOR")
print(SEP)

def test_report_open_close():
    from ui.report_generator import ReportGenerator
    from core.alert_engine import AlertEvent
    import tempfile, shutil
    rg = ReportGenerator("Test Student")
    rg.open_session()
    path = rg.close_session(final_score=25, duration_secs=120)
    assert os.path.exists(path), f"Summary file not created: {path}"
    assert os.path.exists(rg.csv_path), f"CSV not created: {rg.csv_path}"

def test_report_log_event():
    from ui.report_generator import ReportGenerator
    from core.alert_engine import AlertEvent
    import time
    rg = ReportGenerator("Test Student 2")
    rg.open_session()
    ev = AlertEvent(
        timestamp=time.time(),
        violation_key="no_face",
        label="No face detected",
        score_delta=3,
        severity="medium",
    )
    rg.log_event(ev, "")
    rg.close_session(final_score=3, duration_secs=10)
    with open(rg.csv_path) as f:
        content = f.read()
    assert "no_face" in content, "Event not written to CSV"

def test_report_screenshot_dir_created():
    from ui.report_generator import ReportGenerator
    rg = ReportGenerator("Test Student 3")
    rg.open_session()
    rg.close_session(final_score=0, duration_secs=5)
    assert os.path.exists(rg.screenshot_dir), "Screenshot dir not created"

test("ReportGenerator open/close session",    test_report_open_close)
test("ReportGenerator log_event to CSV",      test_report_log_event)
test("ReportGenerator screenshot dir created",test_report_screenshot_dir_created)

# ─── 7. INTEGRATED PIPELINE (no camera, no GUI) ───────────────
print(f"\n{SEP}")
print("  MODULE 7: INTEGRATED PIPELINE")
print(SEP)

def test_pipeline_full():
    """Run all detectors on synthetic frames and verify no crashes."""
    from core.motion_detector import MotionDetector
    from core.face_detector import FaceDetector, FaceAnalysis
    from core.alert_engine import AlertEngine

    md = MotionDetector()
    fd = FaceDetector()
    ae = AlertEngine()

    all_events = []
    for i in range(60):
        frame = blank_frame()
        if i > 30:
            frame[150:300, 200:400] = 200  # simulate motion
        regions, mask, level = md.process(frame)
        fa = fd.process(frame)
        events = ae.evaluate(fa, level, regions, frame)
        all_events.extend(events)

    assert ae.score >= 0, "Score went negative"
    print(f"    Pipeline: {len(all_events)} events fired, final score={ae.score}")

def test_pipeline_event_structure():
    """Every AlertEvent must have all required fields."""
    from core.alert_engine import AlertEngine
    from core.face_detector import FaceAnalysis
    ae = AlertEngine()
    fa = FaceAnalysis(face_count=0, status="no_face")
    events = ae.evaluate(fa, "none", [])
    for e in events:
        assert hasattr(e, "timestamp")
        assert hasattr(e, "violation_key")
        assert hasattr(e, "label")
        assert hasattr(e, "score_delta")
        assert hasattr(e, "severity")
        assert e.severity in ("low", "medium", "high")

def test_pipeline_recent_events_cap():
    """recent_events should never return more than 50 items."""
    from core.alert_engine import AlertEngine
    from core.face_detector import FaceAnalysis
    ae = AlertEngine()
    fa = FaceAnalysis(face_count=0, status="no_face")
    # Fire many events by bypassing cooldown manually
    for _ in range(100):
        ae._cooldowns.clear()
        ae.evaluate(fa, "none", [])
    assert len(ae.recent_events) <= 50

test("Integrated pipeline (60 frames, no crash)", test_pipeline_full)
test("AlertEvent structure validation",           test_pipeline_event_structure)
test("recent_events capped at 50",               test_pipeline_recent_events_cap)

# ─── 8. CAMERA ────────────────────────────────────────────────
print(f"\n{SEP}")
print("  MODULE 8: CAMERA")
print(SEP)

def test_camera_open():
    import cv2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    opened = cap.isOpened()
    cap.release()
    assert opened, "Webcam index 0 could not be opened — check camera connection"

def test_camera_manager_lifecycle():
    from core.camera import CameraManager
    from config import settings as _settings
    cam = CameraManager()
    ok = cam.start()
    assert ok, "CameraManager.start() returned False -- camera not available"
    time.sleep(0.5)
    frame = cam.read()
    assert frame is not None, "Camera opened but read() returned None"
    # After fix, frame must be resized to target resolution
    assert frame.shape[0] == _settings.FRAME_HEIGHT, \
        f"Frame height {frame.shape[0]} != expected {_settings.FRAME_HEIGHT}"
    assert frame.shape[1] == _settings.FRAME_WIDTH, \
        f"Frame width {frame.shape[1]} != expected {_settings.FRAME_WIDTH}"
    assert frame.shape[2] == 3, "Frame must be 3-channel BGR"
    cam.stop()

test("Webcam opens via cv2.VideoCapture",      test_camera_open)
test("CameraManager full lifecycle",           test_camera_manager_lifecycle)

# ─── SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST SUMMARY")
print("=" * 60)

passed = [r for r in results if r[1] == "PASS"]
failed = [r for r in results if r[1] == "FAIL"]

print(f"\n  PASSED : {len(passed)}")
print(f"  FAILED : {len(failed)}")
print(f"  TOTAL  : {len(results)}")

if failed:
    print("\n  FAILURES:")
    for name, status, msg in failed:
        print(f"    FAIL: {name}")
        print(f"      -> {msg}")
else:
    print("\n  All tests passed!")

print()
sys.exit(0 if not failed else 1)
