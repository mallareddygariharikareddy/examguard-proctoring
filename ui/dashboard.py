"""
dashboard.py — Premium Tkinter GUI for the Exam Proctoring System.

Layout (dark theme):
  ┌───────────────── TOP BAR ─────────────────┐
  │ Logo  |  Student name  |  Timer  |  REC   │
  ├──────────────┬────────────────────────────┤
  │              │   RISK SCORE CARD           │
  │  CAMERA FEED │   STATUS PANEL              │
  │  (live HUD)  │   DETECTION INDICATORS      │
  │              │   RECENT ALERTS LOG         │
  └──────────────┴────────────────────────────┘
  │            BOTTOM CONTROLS                 │
  └───────────────────────────────────────────┘
"""

from __future__ import annotations
import tkinter as tk
from tkinter import font as tkfont
import threading
import queue
import time
import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional

from config import settings
from core.camera import CameraManager
from core.motion_detector import MotionDetector
from core.face_detector import FaceDetector, FaceAnalysis
from core.alert_engine import AlertEngine, AlertEvent
from ui.report_generator import ReportGenerator


# ─── Colour palette ──────────────────────────────────────────────────────────
C = {
    "bg":          "#0d1117",
    "panel":       "#161b22",
    "card":        "#21262d",
    "border":      "#30363d",
    "text":        "#f0f6fc",
    "text_dim":    "#8b949e",
    "accent":      "#58a6ff",
    "green":       "#3fb950",
    "yellow":      "#d29922",
    "orange":      "#e67e22",
    "red":         "#f85149",
    "purple":      "#bc8cff",
    "rec_red":     "#ff4444",
    "btn_bg":      "#21262d",
    "btn_hover":   "#30363d",
}

RISK_COLOURS = {
    "LOW":      C["green"],
    "MEDIUM":   C["yellow"],
    "HIGH":     C["orange"],
    "CRITICAL": C["red"],
}


class ProctoringDashboard:
    """Main Tkinter window."""

    # ─── Initialisation ───────────────────────────────────────────────────────

    def __init__(self, root: tk.Tk, student_name: str = "Student"):
        self._root         = root
        self._student_name = student_name
        self._start_time   = time.time()
        self._running      = False
        self._paused       = False

        # Processing queue (camera thread → main thread)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Core modules
        self._camera   = CameraManager()
        self._motion   = MotionDetector()
        self._face     = FaceDetector()
        self._alert    = AlertEngine()
        self._reporter = ReportGenerator(student_name)

        # State
        self._current_frame    = None
        self._face_count       = 0
        self._eyes_visible     = False
        self._gaze             = "unknown"
        self._motion_level     = "none"
        self._alert_list: list = []   # recent alert strings for the log box
        self._rec_blink        = True
        self._photo_ref        = None  # keep reference so GC doesn't kill it
        self._frame_counter    = 0    # used to throttle heavy detectors
        self._destroyed        = False  # guard against double-destroy

        self._build_ui()

    # ─── UI Construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        root = self._root
        root.title("ExamGuard — Proctoring System")
        root.configure(bg=C["bg"])
        root.resizable(False, False)

        # ── Fonts ──────────────────────────────────────────────────────────────
        self._f_title   = tkfont.Font(family=settings.FONT_FAMILY, size=13, weight="bold")
        self._f_body    = tkfont.Font(family=settings.FONT_FAMILY, size=10)
        self._f_small   = tkfont.Font(family=settings.FONT_FAMILY, size=9)
        self._f_score   = tkfont.Font(family=settings.FONT_FAMILY, size=36, weight="bold")
        self._f_risk    = tkfont.Font(family=settings.FONT_FAMILY, size=14, weight="bold")
        self._f_mono    = tkfont.Font(family="Consolas", size=9)
        self._f_icon    = tkfont.Font(family=settings.FONT_FAMILY, size=18)
        self._f_label   = tkfont.Font(family=settings.FONT_FAMILY, size=9, weight="bold")

        # ── Top Bar ────────────────────────────────────────────────────────────
        topbar = tk.Frame(root, bg=C["panel"], height=52)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="🎓 ExamGuard", font=self._f_title,
                 bg=C["panel"], fg=C["accent"]).pack(side="left", padx=18, pady=12)

        sep = tk.Frame(topbar, bg=C["border"], width=1)
        sep.pack(side="left", fill="y", pady=8)

        tk.Label(topbar, text=f"  👤 {self._student_name}",
                 font=self._f_body, bg=C["panel"], fg=C["text_dim"]).pack(
                     side="left", padx=14)

        # Timer (right-aligned)
        self._timer_var = tk.StringVar(value="00:00:00")
        tk.Label(topbar, textvariable=self._timer_var,
                 font=tkfont.Font(family="Consolas", size=12, weight="bold"),
                 bg=C["panel"], fg=C["text"]).pack(side="right", padx=18)

        # REC indicator
        self._rec_var = tk.StringVar(value="⏺ REC")
        self._rec_lbl = tk.Label(topbar, textvariable=self._rec_var,
                                  font=self._f_label, bg=C["panel"],
                                  fg=C["rec_red"])
        self._rec_lbl.pack(side="right", padx=10)

        # ── Bottom Controls (pack FIRST so it always anchors to bottom) ────────
        ctrl = tk.Frame(root, bg=C["panel"], height=56)
        ctrl.pack(side="bottom", fill="x")
        ctrl.pack_propagate(False)

        btn_cfg = dict(bg=C["btn_bg"], fg=C["text"],
                       font=self._f_body, relief="flat",
                       activebackground=C["btn_hover"],
                       activeforeground=C["text"],
                       cursor="hand2", padx=16, pady=10)

        self._start_btn = tk.Button(ctrl, text="▶  Start Session",
                                     command=self._on_start, **btn_cfg)
        self._start_btn.pack(side="left", padx=(14, 6), pady=8)

        self._pause_btn = tk.Button(ctrl, text="⏸  Pause",
                                     command=self._on_pause, state="disabled",
                                     **btn_cfg)
        self._pause_btn.pack(side="left", padx=6, pady=8)

        tk.Button(ctrl, text="🔄  Reset Score",
                  command=self._on_reset_score, **btn_cfg).pack(
                      side="left", padx=6, pady=8)

        tk.Button(ctrl, text="💾  Save Report",
                  command=self._on_save_report, **btn_cfg).pack(
                      side="left", padx=6, pady=8)

        tk.Button(ctrl, text="✖  Quit",
                  command=self._on_quit,
                  bg="#3d1a1a", fg=C["red"],
                  font=self._f_body, relief="flat",
                  activebackground="#5c2a2a", activeforeground=C["red"],
                  cursor="hand2", padx=16, pady=10).pack(side="right", padx=14, pady=8)

        # ── Main content area (packed AFTER bottom bar) ────────────────────────
        content = tk.Frame(root, bg=C["bg"])
        content.pack(fill="both", expand=True, padx=12, pady=(8, 4))

        # Left: camera feed
        cam_frame = tk.Frame(content, bg=C["panel"],
                             bd=0, highlightthickness=1,
                             highlightbackground=C["border"])
        cam_frame.pack(side="left", fill="both")

        self._canvas = tk.Canvas(cam_frame, width=640, height=480,
                                 bg="#000", highlightthickness=0)
        self._canvas.pack()

        # Right: status panel
        right = tk.Frame(content, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # ── Risk Score Card ────────────────────────────────────────────────────
        score_card = tk.Frame(right, bg=C["card"], bd=0,
                              highlightthickness=1,
                              highlightbackground=C["border"])
        score_card.pack(fill="x", pady=(0, 8))

        tk.Label(score_card, text="RISK SCORE", font=self._f_label,
                 bg=C["card"], fg=C["text_dim"]).pack(pady=(10, 0))

        self._score_var = tk.StringVar(value="0")
        self._score_lbl = tk.Label(score_card, textvariable=self._score_var,
                                    font=self._f_score, bg=C["card"], fg=C["green"])
        self._score_lbl.pack()

        self._risk_var = tk.StringVar(value="🟢 LOW")
        self._risk_lbl = tk.Label(score_card, textvariable=self._risk_var,
                                   font=self._f_risk, bg=C["card"], fg=C["green"])
        self._risk_lbl.pack(pady=(0, 10))

        # ── Status indicators ──────────────────────────────────────────────────
        status_card = tk.Frame(right, bg=C["card"], bd=0,
                               highlightthickness=1,
                               highlightbackground=C["border"])
        status_card.pack(fill="x", pady=(0, 8))

        tk.Label(status_card, text="DETECTION STATUS", font=self._f_label,
                 bg=C["card"], fg=C["text_dim"]).pack(anchor="w", padx=14, pady=(10, 6))

        indicators = [
            ("👤 Faces",   "_ind_face"),
            ("👁 Eyes",    "_ind_eyes"),
            ("👀 Gaze",    "_ind_gaze"),
            ("🌊 Motion",  "_ind_motion"),
        ]
        for label, attr in indicators:
            row = tk.Frame(status_card, bg=C["card"])
            row.pack(fill="x", padx=14, pady=3)
            tk.Label(row, text=label, font=self._f_body,
                     bg=C["card"], fg=C["text_dim"], width=12, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            lbl = tk.Label(row, textvariable=var, font=self._f_body,
                           bg=C["card"], fg=C["text"], anchor="w")
            lbl.pack(side="left")
            setattr(self, attr + "_var", var)
            setattr(self, attr + "_lbl", lbl)

        tk.Frame(status_card, bg=C["card"], height=8).pack()

        # ── Session stats strip ────────────────────────────────────────────────
        stats_card = tk.Frame(right, bg=C["card"], bd=0,
                              highlightthickness=1,
                              highlightbackground=C["border"])
        stats_card.pack(fill="x", pady=(0, 8))

        tk.Label(stats_card, text="SESSION STATS", font=self._f_label,
                 bg=C["card"], fg=C["text_dim"]).pack(anchor="w", padx=14, pady=(10, 6))

        stats_row = tk.Frame(stats_card, bg=C["card"])
        stats_row.pack(fill="x", padx=14, pady=(0, 10))

        self._total_alerts_var  = tk.StringVar(value="0")
        self._high_alerts_var   = tk.StringVar(value="0")

        for label, var, color in [
            ("Total alerts", self._total_alerts_var, C["text"]),
            ("High severity", self._high_alerts_var, C["red"]),
        ]:
            col = tk.Frame(stats_row, bg=C["card"])
            col.pack(side="left", expand=True)
            tk.Label(col, textvariable=var,
                     font=tkfont.Font(family=settings.FONT_FAMILY, size=20, weight="bold"),
                     bg=C["card"], fg=color).pack()
            tk.Label(col, text=label, font=self._f_small,
                     bg=C["card"], fg=C["text_dim"]).pack()

        # ── Alert log ──────────────────────────────────────────────────────────
        log_card = tk.Frame(right, bg=C["card"], bd=0,
                            highlightthickness=1,
                            highlightbackground=C["border"])
        log_card.pack(fill="both", expand=True)

        tk.Label(log_card, text="ALERT LOG", font=self._f_label,
                 bg=C["card"], fg=C["text_dim"]).pack(anchor="w", padx=14, pady=(10, 4))

        self._log_text = tk.Text(
            log_card, bg=C["panel"], fg=C["text"],
            font=self._f_mono, bd=0, relief="flat",
            state="disabled", wrap="word",
            highlightthickness=0,
        )
        self._log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Configure log tags for colouring
        self._log_text.tag_configure("low",      foreground=C["text_dim"])
        self._log_text.tag_configure("medium",   foreground=C["yellow"])
        self._log_text.tag_configure("high",     foreground=C["orange"])
        self._log_text.tag_configure("critical", foreground=C["red"])
        self._log_text.tag_configure("time",     foreground=C["accent"])

        # ── Show placeholder on canvas ─────────────────────────────────────────
        self._draw_placeholder()

    # ─── Placeholder ──────────────────────────────────────────────────────────

    def _draw_placeholder(self):
        """Draw a 'waiting for camera' image on the canvas before start."""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (22, 27, 34)  # C["panel"] in BGR

        # Gradient-ish tinted centre
        for i in range(480):
            alpha = abs(i - 240) / 240
            placeholder[i, :] = (
                int(22 + 10 * (1 - alpha)),
                int(27 + 15 * (1 - alpha)),
                int(34 + 25 * (1 - alpha)),
            )

        # Text
        cv2.putText(placeholder, "ExamGuard", (175, 210),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (88, 166, 255), 2)
        cv2.putText(placeholder, "Press  Start Session  to begin",
                    (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (139, 148, 158), 1)
        cv2.putText(placeholder, "Motion Detection Proctoring System",
                    (90, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (63, 185, 80), 1)

        self._update_canvas(placeholder)

    # ─── Camera / Processing Thread ───────────────────────────────────────────

    def _processing_loop(self):
        """Runs in a daemon thread. Captures → analyses → queues results."""
        reporter_opened = False
        try:
            self._reporter.open_session()
            reporter_opened = True
        except Exception as e:
            print(f"[Reporter] Could not open session: {e}")

        # Cached face analysis — updated every 3rd frame to avoid blocking
        _cached_face = FaceAnalysis()

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            frame = self._camera.read()
            if frame is None:
                time.sleep(0.02)
                continue

            # Ensure consistent resolution regardless of camera native res
            if frame.shape[0] != settings.FRAME_HEIGHT or frame.shape[1] != settings.FRAME_WIDTH:
                frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))

            self._frame_counter += 1

            # ── Analysis ──────────────────────────────────────────────────────
            # MOG2 motion runs every frame (fast, C++ only)
            regions, fg_mask, motion_level = self._motion.process(frame)

            # Haar face detection runs every 3rd frame (heavier)
            if self._frame_counter % 3 == 0:
                _cached_face = self._face.process(frame)
            face_analysis = _cached_face

            new_events = self._alert.evaluate(
                face_analysis, motion_level, regions, frame
            )

            # ── Handle new alerts ──────────────────────────────────────────────
            for event in new_events:
                screenshot_path = ""
                if event.frame_snapshot is not None:
                    screenshot_path = self._reporter.save_screenshot(
                        event.frame_snapshot, event
                    )
                if reporter_opened:
                    self._reporter.log_event(event, screenshot_path)

            # ── Draw HUD on frame ──────────────────────────────────────────────
            hud_frame = self._draw_hud(
                frame.copy(), face_analysis, regions, fg_mask, motion_level
            )

            # ── Push to UI queue (drop old frame if full) ──────────────────────
            result = {
                "frame":        hud_frame,
                "face_analysis":face_analysis,
                "motion_level": motion_level,
                "new_events":   new_events,
                "risk":         self._alert.get_risk_level(),
                "score":        self._alert.score,
                "all_events":   self._alert.recent_events,
            }
            try:
                self._frame_queue.put_nowait(result)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(result)
                except queue.Empty:
                    pass

        if reporter_opened:
            self._reporter.close_session(
                final_score=self._alert.score,
                duration_secs=time.time() - self._start_time,
            )

    # ─── HUD Drawing ──────────────────────────────────────────────────────────

    def _draw_hud(self, frame, face_analysis, regions, fg_mask, motion_level):
        """Overlay detection results onto the BGR frame."""
        h, w = frame.shape[:2]

        # Zone dividers (subtle)
        zone_y1 = int(h * settings.ZONE_HEAD_BOTTOM)
        zone_y2 = int(h * settings.ZONE_BODY_BOTTOM)
        cv2.line(frame, (0, zone_y1), (w, zone_y1), (48, 54, 61), 1)
        cv2.line(frame, (0, zone_y2), (w, zone_y2), (48, 54, 61), 1)

        # Motion contours
        for region in regions:
            x, y, rw, rh = region.bbox
            color = {
                "large":  (248, 81,  73),   # red
                "medium": (230, 126, 34),    # orange
                "small":  (88,  166, 255),   # blue
            }.get(region.magnitude, (88, 166, 255))
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 2)
            cv2.putText(frame, f"{region.zone}/{region.magnitude}",
                        (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Face boxes
        if face_analysis.faces:
            for i, face in enumerate(face_analysis.faces):
                fx, fy, fw, fh = face.bbox
                is_primary = (i == 0)
                color = (63, 185, 80) if (is_primary and face.eye_count >= 1) \
                         else (248, 81, 73)
                thickness = 2 if is_primary else 1
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh),
                              color, thickness)
                label = f"Face {i+1} | {face.gaze} | eyes:{face.eye_count}"
                cv2.putText(frame, label,
                            (fx, fy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1)

        # Top-left status badge
        risk = self._alert.get_risk_level()
        badge_color = {
            "LOW":      (63,  185, 80),
            "MEDIUM":   (210, 153, 34),
            "HIGH":     (230, 126, 34),
            "CRITICAL": (248, 81,  73),
        }.get(risk.label, (88, 166, 255))

        cv2.rectangle(frame, (8, 8), (220, 38), (22, 27, 34), -1)
        cv2.rectangle(frame, (8, 8), (220, 38), badge_color, 1)
        cv2.putText(frame, f"SCORE:{self._alert.score}  RISK:{risk.label}",
                    (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 1)

        # Motion level badge (top right)
        ml_color = {
            "none":   (63,  185, 80),
            "low":    (63,  185, 80),
            "medium": (210, 153, 34),
            "high":   (248, 81,  73),
        }.get(motion_level, (88, 166, 255))
        cv2.rectangle(frame, (w - 175, 8), (w - 8, 38), (22, 27, 34), -1)
        cv2.rectangle(frame, (w - 175, 8), (w - 8, 38), ml_color, 1)
        cv2.putText(frame, f"MOTION: {motion_level.upper()}",
                    (w - 168, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ml_color, 1)

        return frame

    # ─── Tkinter UI refresh (runs on main thread via after()) ────────────────

    def _poll_queue(self):
        """Called by Tkinter every ~33ms to consume processed frames."""
        if self._destroyed:
            return

        processed = 0
        while not self._frame_queue.empty() and processed < 2:
            try:
                result = self._frame_queue.get_nowait()
                self._apply_result(result)
                processed += 1
            except queue.Empty:
                break

        if self._running and not self._destroyed:
            self._root.after(33, self._poll_queue)

        # Update timer
        if self._running and not self._paused and not self._destroyed:
            elapsed = time.time() - self._start_time
            self._timer_var.set(str(datetime.timedelta(seconds=int(elapsed))))

    def _apply_result(self, result: dict):
        """Update all UI widgets from the latest processed frame result."""
        # Camera canvas
        self._update_canvas(result["frame"])

        # Score + risk
        risk  = result["risk"]
        score = result["score"]
        self._score_var.set(str(score))
        self._risk_var.set(f"{risk.emoji} {risk.label}")
        color = RISK_COLOURS.get(risk.label, C["text"])
        self._score_lbl.configure(fg=color)
        self._risk_lbl.configure(fg=color)

        # Detection indicators
        fa = result["face_analysis"]
        ml = result["motion_level"]

        face_text  = f"{fa.face_count} detected"
        face_color = C["green"] if fa.face_count == 1 else (
            C["red"] if fa.face_count == 0 else C["orange"]
        )
        self._ind_face_var.set(face_text)
        self._ind_face_lbl.configure(fg=face_color)

        eyes_text  = "Visible ✓" if fa.eyes_visible else "Not visible ✗"
        eyes_color = C["green"] if fa.eyes_visible else C["red"]
        self._ind_eyes_var.set(eyes_text)
        self._ind_eyes_lbl.configure(fg=eyes_color)

        gaze_text  = fa.gaze.capitalize()
        gaze_color = C["green"] if fa.gaze == "centre" else (
            C["red"] if fa.gaze == "away" else C["yellow"]
        )
        self._ind_gaze_var.set(gaze_text)
        self._ind_gaze_lbl.configure(fg=gaze_color)

        motion_color = {
            "none":   C["green"],
            "low":    C["green"],
            "medium": C["yellow"],
            "high":   C["red"],
        }.get(ml, C["text"])
        self._ind_motion_var.set(ml.capitalize())
        self._ind_motion_lbl.configure(fg=motion_color)

        # Session stats
        all_events = result["all_events"]
        self._total_alerts_var.set(str(len(all_events)))
        high_count = sum(1 for e in all_events if e.severity == "high")
        self._high_alerts_var.set(str(high_count))

        # New alert log entries
        for event in result["new_events"]:
            self._append_log_entry(event)

    def _update_canvas(self, frame: np.ndarray):
        """Convert BGR frame → Tkinter PhotoImage and put on canvas."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self._canvas.create_image(0, 0, anchor="nw", image=photo)
        self._photo_ref = photo   # prevent GC

    def _append_log_entry(self, event: AlertEvent):
        """Add one coloured line to the alert log Text widget."""
        ts  = datetime.datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        sev = event.severity

        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"[{ts}] ", ("time",))
        self._log_text.insert("end", f"{event.label}\n", (sev,))
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    # ─── Blinking REC indicator ───────────────────────────────────────────────

    def _blink_rec(self):
        if not self._running:
            return
        self._rec_blink = not self._rec_blink
        if self._rec_blink:
            self._rec_var.set("⏺ REC")
            self._rec_lbl.configure(fg=C["rec_red"])
        else:
            self._rec_var.set("   REC")
            self._rec_lbl.configure(fg=C["text_dim"])
        self._root.after(800, self._blink_rec)

    # ─── Button handlers ──────────────────────────────────────────────────────

    def _on_start(self):
        if self._running:
            return
        ok = self._camera.start()
        if not ok:
            self._append_raw_log("❌ Could not open webcam. Check connection.", "critical")
            return

        self._running     = True
        self._start_time  = time.time()
        self._paused      = False
        self._start_btn.configure(state="disabled")
        self._pause_btn.configure(state="normal")

        # Reset background model for fresh start
        self._motion.reset()

        # Start processing thread
        t = threading.Thread(target=self._processing_loop,
                             daemon=True, name="ProcessingThread")
        t.start()

        # Start Tkinter polling + blinking
        self._root.after(33, self._poll_queue)
        self._blink_rec()

    def _on_pause(self):
        if not self._running:
            return
        self._paused = not self._paused
        if self._paused:
            self._pause_btn.configure(text="▶  Resume")
            self._rec_var.set("⏸ PAUSED")
            self._rec_lbl.configure(fg=C["yellow"])
        else:
            self._pause_btn.configure(text="⏸  Pause")
            self._blink_rec()

    def _on_reset_score(self):
        self._alert.reset_score()
        self._score_var.set("0")
        self._risk_var.set("🟢 LOW")
        self._score_lbl.configure(fg=C["green"])
        self._risk_lbl.configure(fg=C["green"])
        self._append_raw_log("⟳ Score manually reset.", "low")

    def _on_save_report(self):
        if not self._running:
            self._append_raw_log("⚠ Start a session first.", "medium")
            return
        path = self._reporter.close_session(
            final_score=self._alert.score,
            duration_secs=time.time() - self._start_time,
        )
        self._append_raw_log(f"💾 Report saved → {path}", "low")
        # Re-open so logging continues
        try:
            self._reporter.open_session()
        except Exception:
            pass

    def _on_quit(self):
        if self._destroyed:
            return
        self._destroyed = True
        self._running = False
        self._camera.stop()
        try:
            self._reporter.close_session(
                final_score=self._alert.score,
                duration_secs=time.time() - self._start_time,
            )
        except Exception:
            pass
        try:
            self._root.destroy()
        except Exception:
            pass

    def _append_raw_log(self, text: str, tag: str = "low"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"[{ts}] ", ("time",))
        self._log_text.insert("end", f"{text}\n", (tag,))
        self._log_text.see("end")
        self._log_text.configure(state="disabled")
