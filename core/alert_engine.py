"""
alert_engine.py — Violation scoring, cooldowns, and risk level tracking.

The engine receives detection results each frame and:
  1. Checks each violation condition
  2. Applies cooldown so identical alerts don't spam
  3. Adds / decays the running risk score
  4. Emits AlertEvent objects for the UI and logger
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from config import settings
from core.motion_detector import MotionRegion   # type: ignore
from core.face_detector import FaceAnalysis     # type: ignore


@dataclass
class AlertEvent:
    """One fired violation alert."""
    timestamp: float       # time.time()
    violation_key: str
    label: str
    score_delta: int
    severity: str          # "low" | "medium" | "high"
    frame_snapshot: Optional[object] = None   # BGR ndarray if screenshot saving on


@dataclass
class RiskLevel:
    label: str    # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    color: str    # hex
    emoji: str


class AlertEngine:
    """
    Stateful alert engine.  Call `evaluate()` once per processed frame.
    """

    def __init__(self):
        self.score: int = 0
        self._cooldowns: Dict[str, float] = {}   # violation_key → next_allowed_time
        self._last_decay: float = time.time()
        self._events: List[AlertEvent] = []       # all-time log
        self._recent_events: List[AlertEvent] = [] # last N for the HUD

    # ─── Public ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        face_analysis,         # FaceAnalysis from face_detector
        motion_level: str,     # "none"|"low"|"medium"|"high"
        motion_regions,        # List[MotionRegion]
        frame=None,            # BGR ndarray (for screenshot)
    ) -> List[AlertEvent]:
        """
        Evaluate one frame's detections.
        Returns a (possibly empty) list of newly fired AlertEvents.
        """
        self._apply_decay()
        new_events: List[AlertEvent] = []

        # ── Face-based violations ─────────────────────────────────────────────
        if face_analysis.face_count == 0:
            ev = self._try_fire("no_face", frame)
            if ev:
                new_events.append(ev)

        if face_analysis.face_count > 1:
            ev = self._try_fire("multiple_faces", frame)
            if ev:
                new_events.append(ev)

        if face_analysis.status == "away":
            ev = self._try_fire("looking_away", frame)
            if ev:
                new_events.append(ev)

        # ── Motion-based violations ───────────────────────────────────────────
        if motion_level == "high":
            ev = self._try_fire("excess_motion", frame)
            if ev:
                new_events.append(ev)
        elif motion_level == "medium":
            # Check if motion is in body zone (likely hands reaching for something)
            body_motion = any(r.zone == "body" for r in motion_regions)
            if body_motion:
                ev = self._try_fire("large_motion", frame)
                if ev:
                    new_events.append(ev)

        # ── Seat empty ────────────────────────────────────────────────────────
        seat_motion = any(r.zone == "seat" and r.magnitude == "large"
                          for r in motion_regions)
        if face_analysis.face_count == 0 and seat_motion:
            ev = self._try_fire("seat_empty", frame)
            if ev:
                new_events.append(ev)

        return new_events

    def reset_score(self):
        """Called by the user pressing the Reset button."""
        self.score = 0
        self._cooldowns.clear()

    def get_risk_level(self) -> RiskLevel:
        for lo, hi, label, color, emoji in settings.RISK_LEVELS:
            if lo <= self.score <= hi:
                return RiskLevel(label=label, color=color, emoji=emoji)
        return RiskLevel("CRITICAL", "#e74c3c", "🔴")

    @property
    def recent_events(self) -> List[AlertEvent]:
        """Last 50 events for display."""
        return self._events[-50:]

    @property
    def all_events(self) -> List[AlertEvent]:
        return list(self._events)

    # ─── Internal ────────────────────────────────────────────────────────────

    def _try_fire(self, key: str, frame=None) -> Optional[AlertEvent]:
        """Fire an alert if the cooldown has expired."""
        now = time.time()
        if now < self._cooldowns.get(key, 0):
            return None   # still cooling down

        score_delta, cooldown, label, severity = settings.VIOLATIONS[key]
        self.score = min(self.score + score_delta, 999)
        self._cooldowns[key] = now + cooldown

        # Optionally attach screenshot
        snapshot = None
        if (settings.SAVE_SCREENSHOTS
                and frame is not None
                and self.score >= settings.SCREENSHOT_MIN_SCORE):
            snapshot = frame.copy()

        event = AlertEvent(
            timestamp=now,
            violation_key=key,
            label=label,
            score_delta=score_delta,
            severity=severity,
            frame_snapshot=snapshot,
        )
        self._events.append(event)
        return event

    def _apply_decay(self):
        """Gradually reduce score during clean periods."""
        now = time.time()
        if now - self._last_decay >= settings.SCORE_DECAY_INTERVAL:
            self.score = max(0, self.score - settings.SCORE_DECAY_AMOUNT)
            self._last_decay = now
