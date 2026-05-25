"""
report_generator.py — Session CSV logging and end-of-exam report.

Responsibilities:
  • Write one CSV row per AlertEvent to logs/session_<timestamp>.csv
  • Save flagged-frame screenshots to logs/screenshots/
  • Generate a plain-text summary report at session end
"""

from __future__ import annotations
import os
import csv
import time
import datetime
import cv2
from typing import List
from config import settings
from core.alert_engine import AlertEvent


class ReportGenerator:
    """
    Handles all I/O for a single proctoring session.

    Call flow:
        rg = ReportGenerator(student_name="Alice")
        rg.open_session()
        rg.log_event(event)           # called by dashboard
        rg.save_screenshot(frame, event)
        rg.close_session(score=42)    # writes summary
    """

    def __init__(self, student_name: str = "Unknown"):
        self._student_name   = student_name
        self._session_start  = time.time()
        self._session_id     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_dir        = settings.LOG_DIR
        self._screenshot_dir = os.path.join(
            self._log_dir, settings.SCREENSHOTS_SUBDIR, self._session_id
        )
        self._csv_path   = os.path.join(
            self._log_dir, f"session_{self._session_id}.csv"
        )
        self._csv_file   = None
        self._csv_writer = None
        self._event_count = 0

    # ─── Session lifecycle ────────────────────────────────────────────────────

    def open_session(self):
        """Create directories and open the CSV file for writing."""
        os.makedirs(self._log_dir, exist_ok=True)
        if settings.SAVE_SCREENSHOTS:
            os.makedirs(self._screenshot_dir, exist_ok=True)

        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "datetime", "violation_key", "label",
            "score_delta", "severity", "screenshot_path"
        ])
        self._csv_file.flush()

    def log_event(self, event: AlertEvent, screenshot_path: str = "") -> None:
        """Append one alert event row to the CSV."""
        if self._csv_writer is None:
            return
        dt = datetime.datetime.fromtimestamp(event.timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self._csv_writer.writerow([
            f"{event.timestamp:.3f}",
            dt,
            event.violation_key,
            event.label,
            event.score_delta,
            event.severity,
            screenshot_path,
        ])
        self._csv_file.flush()
        self._event_count += 1

    def save_screenshot(self, frame, event: AlertEvent) -> str:
        """
        Save the flagged frame as a JPEG.
        Returns the saved file path, or "" if saving is disabled.
        """
        if not settings.SAVE_SCREENSHOTS or frame is None:
            return ""

        fname = (
            f"{event.violation_key}_"
            f"{datetime.datetime.fromtimestamp(event.timestamp).strftime('%H%M%S_%f')}"
            ".jpg"
        )
        path = os.path.join(self._screenshot_dir, fname)
        cv2.imwrite(path, frame)
        return path

    def close_session(self, final_score: int, duration_secs: float) -> str:
        """
        Flush the CSV, write a plain-text summary, and return the summary path.
        """
        if self._csv_file:
            self._csv_file.close()

        summary_path = os.path.join(
            self._log_dir, f"summary_{self._session_id}.txt"
        )
        duration_str = str(datetime.timedelta(seconds=int(duration_secs)))

        risk_label = "LOW"
        for lo, hi, label, *_ in settings.RISK_LEVELS:
            if lo <= final_score <= hi:
                risk_label = label

        lines = [
            "=" * 60,
            "  EXAM PROCTORING SESSION REPORT",
            "=" * 60,
            f"  Student      : {self._student_name}",
            f"  Session ID   : {self._session_id}",
            f"  Duration     : {duration_str}",
            f"  Total alerts : {self._event_count}",
            f"  Final score  : {final_score}",
            f"  Risk level   : {risk_label}",
            "-" * 60,
            f"  CSV log      : {self._csv_path}",
            f"  Screenshots  : {self._screenshot_dir}",
            "=" * 60,
        ]

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return summary_path

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def csv_path(self) -> str:
        return self._csv_path

    @property
    def screenshot_dir(self) -> str:
        return self._screenshot_dir

    @property
    def session_id(self) -> str:
        return self._session_id
