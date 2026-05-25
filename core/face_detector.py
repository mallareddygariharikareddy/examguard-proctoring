"""
face_detector.py — Haar Cascade face + eye detection.

Returns a FaceAnalysis object per frame describing:
  • Number of faces detected
  • Whether both eyes are visible on the primary face
  • Bounding boxes for HUD overlay
  • Gaze direction estimate (left / centre / right / away)
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from config import settings


@dataclass
class FaceResult:
    """Detected face metadata."""
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in frame coords
    eye_count: int                     # 0, 1, or 2
    gaze: str                          # "centre"|"left"|"right"|"away"
    area: int


@dataclass
class FaceAnalysis:
    """Aggregated result for a single frame."""
    faces: List[FaceResult] = field(default_factory=list)
    face_count: int   = 0
    primary: Optional[FaceResult] = None   # largest face
    eyes_visible: bool  = False
    gaze: str           = "unknown"        # from primary face
    status: str         = "ok"            # "ok"|"no_face"|"multiple"|"away"


class FaceDetector:
    """
    Wraps OpenCV's Haar Cascade detectors for faces and eyes.
    Falls back gracefully if the XML files are missing.
    """

    def __init__(self):
        # OpenCV ships these; cv2.data.haarcascades gives the install path
        base = cv2.data.haarcascades
        self._face_cascade = cv2.CascadeClassifier(
            f"{base}haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            f"{base}haarcascade_eye.xml"
        )
        self._no_face_streak = 0   # consecutive frames with no face

    # ─── Public ──────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> FaceAnalysis:
        """Analyse one BGR frame and return a FaceAnalysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improve contrast in dim lighting

        raw_faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=settings.FACE_SCALE_FACTOR,
            minNeighbors=settings.FACE_MIN_NEIGHBORS,
            minSize=settings.FACE_MIN_SIZE,
        )

        analysis = FaceAnalysis()

        if len(raw_faces) == 0:
            self._no_face_streak += 1
            analysis.face_count = 0
            analysis.status = "no_face"
            return analysis

        self._no_face_streak = 0

        faces: List[FaceResult] = []
        for (x, y, w, h) in raw_faces:
            # Only search for eyes inside the face ROI
            roi_gray = gray[y: y + h, x: x + w]
            eyes = self._eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=settings.EYE_SCALE_FACTOR,
                minNeighbors=settings.EYE_MIN_NEIGHBORS,
                minSize=settings.EYE_MIN_SIZE,
            )
            eye_count = min(len(eyes), 2)
            gaze = self._estimate_gaze(x, w, frame.shape[1], eye_count)
            faces.append(FaceResult(
                bbox=(x, y, w, h),
                eye_count=eye_count,
                gaze=gaze,
                area=w * h,
            ))

        # Sort by area descending — primary = largest face
        faces.sort(key=lambda f: f.area, reverse=True)
        primary = faces[0]

        analysis.faces       = faces
        analysis.face_count  = len(faces)
        analysis.primary     = primary
        analysis.eyes_visible = primary.eye_count >= 1
        analysis.gaze        = primary.gaze

        if len(faces) > 1:
            analysis.status = "multiple"
        elif primary.eye_count == 0:
            analysis.status = "away"
        else:
            analysis.status = "ok"

        return analysis

    @property
    def no_face_streak(self) -> int:
        return self._no_face_streak

    # ─── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_gaze(face_x: int, face_w: int,
                       frame_w: int, eye_count: int) -> str:
        """Rough gaze direction from face centre position."""
        if eye_count == 0:
            return "away"

        face_cx = face_x + face_w // 2
        rel = face_cx / frame_w   # 0 = far left, 1 = far right

        if rel < 0.30:
            return "right"   # mirrored — face centre is frame-left → looking right
        if rel > 0.70:
            return "left"
        return "centre"
