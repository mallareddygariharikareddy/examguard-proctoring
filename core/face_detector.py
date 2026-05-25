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
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from config import settings

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision import face_detector as mp_face_detector
except ImportError:
    mp = None
    base_options = None
    mp_face_detector = None


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
        self._face_alt_cascade = cv2.CascadeClassifier(
            f"{base}haarcascade_frontalface_alt2.xml"
        )
        self._profile_cascade = cv2.CascadeClassifier(
            f"{base}haarcascade_profileface.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            f"{base}haarcascade_eye.xml"
        )
        self._mp_face_detection = None
        if (
            settings.FACE_DETECTOR_BACKEND == "mediapipe"
            and mp is not None
            and base_options is not None
            and mp_face_detector is not None
        ):
            model_path = Path(settings.MEDIAPIPE_FACE_MODEL_PATH)
            if not model_path.is_absolute():
                model_path = Path(__file__).resolve().parents[1] / model_path
            if model_path.exists():
                options = mp_face_detector.FaceDetectorOptions(
                    base_options=base_options.BaseOptions(
                        model_asset_path=str(model_path)
                    ),
                    min_detection_confidence=settings.MEDIAPIPE_FACE_MIN_CONFIDENCE,
                )
                self._mp_face_detection = (
                    mp_face_detector.FaceDetector.create_from_options(options)
                )
        self._no_face_streak = 0   # consecutive frames with no face

    # ─── Public ──────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> FaceAnalysis:
        """Analyse one BGR frame and return a FaceAnalysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improve contrast in dim lighting

        raw_faces = self._detect_faces(frame, gray)
        raw_faces = self._filter_faces(raw_faces)

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

    def _detect_faces(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe first, with Haar as a fallback."""
        if self._mp_face_detection is not None:
            faces = self._detect_faces_mediapipe(frame)
            if faces:
                return faces
        return self._detect_faces_haar(gray)

    def _detect_faces_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces with MediaPipe's fast webcam face detector."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._mp_face_detection.detect(image)
        if not results.detections:
            return []

        frame_h, frame_w = frame.shape[:2]
        detections: List[Tuple[int, int, int, int]] = []
        for detection in results.detections:
            bbox = detection.bounding_box
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)

            x = max(0, x)
            y = max(0, y)
            w = min(frame_w - x, w)
            h = min(frame_h - y, h)
            if w > 0 and h > 0:
                detections.append((x, y, w, h))

        return detections

    def _detect_faces_haar(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect frontal and side-profile faces with OpenCV Haar cascades."""
        detections: List[Tuple[int, int, int, int]] = []
        frame_w = gray.shape[1]

        for cascade in (self._face_cascade, self._face_alt_cascade):
            if cascade.empty():
                continue
            detections.extend(
                (int(x), int(y), int(w), int(h))
                for (x, y, w, h) in cascade.detectMultiScale(
                    gray,
                    scaleFactor=settings.FACE_SCALE_FACTOR,
                    minNeighbors=settings.FACE_MIN_NEIGHBORS,
                    minSize=settings.FACE_MIN_SIZE,
                )
            )

        if not self._profile_cascade.empty():
            detections.extend(
                (int(x), int(y), int(w), int(h))
                for (x, y, w, h) in self._profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=settings.FACE_SCALE_FACTOR,
                    minNeighbors=settings.FACE_PROFILE_MIN_NEIGHBORS,
                    minSize=settings.FACE_MIN_SIZE,
                )
            )

            flipped = cv2.flip(gray, 1)
            for (x, y, w, h) in self._profile_cascade.detectMultiScale(
                flipped,
                scaleFactor=settings.FACE_SCALE_FACTOR,
                minNeighbors=settings.FACE_PROFILE_MIN_NEIGHBORS,
                minSize=settings.FACE_MIN_SIZE,
            ):
                detections.append((int(frame_w - x - w), int(y), int(w), int(h)))

        return detections

    @staticmethod
    def _filter_faces(raw_faces) -> List[Tuple[int, int, int, int]]:
        """
        Remove duplicate and low-confidence face boxes from Haar output.

        Haar cascades can return several overlapping boxes for one real face,
        and occasionally small false positives from posters, shadows, or room
        patterns. Keep the largest box from each overlap group and ignore tiny
        secondary boxes that are unlikely to be another person near the camera.
        """
        if len(raw_faces) == 0:
            return []

        boxes = [
            (int(x), int(y), int(w), int(h))
            for (x, y, w, h) in raw_faces
        ]
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

        kept: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            if all(
                FaceDetector._iou(box, existing) < settings.FACE_NMS_IOU_THRESHOLD
                for existing in kept
            ):
                kept.append(box)

        if not kept:
            return []

        primary_area = kept[0][2] * kept[0][3]
        min_secondary_area = primary_area * settings.FACE_SECONDARY_MIN_AREA_RATIO
        return [
            box for i, box in enumerate(kept)
            if i == 0 or (box[2] * box[3]) >= min_secondary_area
        ]

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        a_area = aw * ah
        b_area = bw * bh
        return inter_area / float(a_area + b_area - inter_area)

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
