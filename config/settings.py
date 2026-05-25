"""
settings.py — Central configuration for the Exam Proctoring System.
All tunable parameters live here so you never need to hunt through source files.
"""

# ─────────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────────
CAMERA_INDEX        = 0        # 0 = default webcam
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
TARGET_FPS          = 30

# ─────────────────────────────────────────────
#  MOTION DETECTION (MOG2)
# ─────────────────────────────────────────────
MOG2_HISTORY        = 500      # frames kept in background model
MOG2_VAR_THRESHOLD  = 80       # sensitivity (lower = more sensitive)
MOG2_DETECT_SHADOWS = True
MOTION_BLUR_KERNEL  = (21, 21)
MORPH_KERNEL_SIZE   = (5, 5)

# Minimum contour area (pixels²) to count as real motion
MOTION_MIN_AREA_SMALL  = 1200   # noise floor
MOTION_MIN_AREA_MEDIUM = 7000   # deliberate hand / object movement
MOTION_MIN_AREA_LARGE  = 22000  # body / full-frame movement

# Frame zones (fractions of frame height)
ZONE_HEAD_TOP     = 0.0
ZONE_HEAD_BOTTOM  = 0.40
ZONE_BODY_TOP     = 0.40
ZONE_BODY_BOTTOM  = 0.75
ZONE_SEAT_TOP     = 0.75
ZONE_SEAT_BOTTOM  = 1.0

# ─────────────────────────────────────────────
#  FACE DETECTION (Haar Cascade)
# ─────────────────────────────────────────────
FACE_DETECTOR_BACKEND = "mediapipe"  # "mediapipe" or "haar"
MEDIAPIPE_FACE_MODEL_PATH = "models/blaze_face_short_range.tflite"
MEDIAPIPE_FACE_MIN_CONFIDENCE = 0.55

FACE_SCALE_FACTOR   = 1.1
FACE_MIN_NEIGHBORS  = 7
FACE_PROFILE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE       = (45, 45)
FACE_NMS_IOU_THRESHOLD = 0.30
FACE_SECONDARY_MIN_AREA_RATIO = 0.20

EYE_SCALE_FACTOR    = 1.1
EYE_MIN_NEIGHBORS   = 10
EYE_MIN_SIZE        = (20, 20)

# How many consecutive frames of no-face before alerting
NO_FACE_FRAME_THRESHOLD = 15    # ~0.5 sec at 30fps
MULTIPLE_FACE_FRAME_THRESHOLD = 4
LOOKING_AWAY_FRAME_THRESHOLD = 12
HIGH_MOTION_FRAME_THRESHOLD = 8
BODY_MOTION_FRAME_THRESHOLD = 12

# ─────────────────────────────────────────────
#  ALERT ENGINE — VIOLATION SCORES & COOLDOWNS
# ─────────────────────────────────────────────
VIOLATIONS = {
    # key: (score_delta, cooldown_seconds, display_label, severity)
    "no_face":        (3,  2,  "No face detected",          "medium"),
    "multiple_faces": (10, 5,  "Multiple persons detected",  "high"),
    "looking_away":   (2,  1,  "Looking away from screen",  "low"),
    "large_motion":   (2,  2,  "Suspicious hand movement",  "low"),
    "excess_motion":  (5,  3,  "Excessive body movement",   "medium"),
    "seat_empty":     (8,  5,  "Student left their seat",   "high"),
}

# Risk level thresholds
RISK_LEVELS = [
    (0,  10, "LOW",      "#2ecc71", "🟢"),
    (11, 30, "MEDIUM",   "#f39c12", "🟡"),
    (31, 60, "HIGH",     "#e67e22", "🟠"),
    (61, 999,"CRITICAL", "#e74c3c", "🔴"),
]

# Score decay: subtract 1 point every N seconds of clean behaviour
SCORE_DECAY_INTERVAL = 5      # seconds
SCORE_DECAY_AMOUNT   = 1

# ─────────────────────────────────────────────
#  SCREENSHOT SAVING
# ─────────────────────────────────────────────
SAVE_SCREENSHOTS    = True
SCREENSHOT_MIN_SCORE = 10      # only save when risk score >= this

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_DIR             = "logs"
SCREENSHOTS_SUBDIR  = "screenshots"

# ─────────────────────────────────────────────
#  UI COLOURS (Tkinter dark theme)
# ─────────────────────────────────────────────
BG_DARK     = "#0d1117"
BG_PANEL    = "#161b22"
BG_CARD     = "#21262d"
ACCENT_BLUE = "#58a6ff"
TEXT_PRIMARY   = "#f0f6fc"
TEXT_SECONDARY = "#8b949e"
BORDER_COLOR   = "#30363d"

FONT_FAMILY = "Segoe UI"
