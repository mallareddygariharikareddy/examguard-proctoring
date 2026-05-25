"""
motion_detector.py — MOG2 background subtraction + contour analysis.

For every frame it returns:
  • A list of MotionRegion named-tuples describing each detected motion blob
  • The cleaned foreground mask (for HUD overlay)
  • An overall motion level string: "none" | "low" | "medium" | "high"
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from config import settings


@dataclass
class MotionRegion:
    """One detected motion blob."""
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    area: int
    zone: str                          # "head" | "body" | "seat" | "unknown"
    magnitude: str                     # "small" | "medium" | "large"
    cx: int                            # centroid x
    cy: int                            # centroid y


class MotionDetector:
    """
    Wraps OpenCV's MOG2 background subtractor with morphological cleanup
    and zone-aware contour classification.
    """

    def __init__(self, frame_height: int = settings.FRAME_HEIGHT):
        self._frame_height = frame_height
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=settings.MOG2_HISTORY,
            varThreshold=settings.MOG2_VAR_THRESHOLD,
            detectShadows=settings.MOG2_DETECT_SHADOWS,
        )
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, settings.MORPH_KERNEL_SIZE
        )

    # ─── Public ──────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> Tuple[List[MotionRegion], np.ndarray, str]:
        """
        Analyse one BGR frame.

        Returns:
            regions      – list of MotionRegion objects
            clean_mask   – binary uint8 mask (for overlay)
            motion_level – "none" | "low" | "medium" | "high"
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, settings.MOTION_BLUR_KERNEL, 0)

        # Apply MOG2 — returns uint8 mask (255 = foreground, 127 = shadow)
        fg_mask = self._mog2.apply(blurred)

        # Remove shadows (set them to background)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological clean-up: erode noise, then dilate to merge blobs
        fg_mask = cv2.erode(fg_mask,  self._morph_kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self._morph_kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: List[MotionRegion] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < settings.MOTION_MIN_AREA_SMALL:
                continue  # pure noise

            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            zone      = self._classify_zone(cy, frame.shape[0])
            magnitude = self._classify_magnitude(area)

            regions.append(MotionRegion(
                bbox=(x, y, w, h),
                area=int(area),
                zone=zone,
                magnitude=magnitude,
                cx=cx,
                cy=cy,
            ))

        motion_level = self._overall_level(regions, frame.shape[0] * frame.shape[1])
        return regions, fg_mask, motion_level

    def reset(self):
        """Re-initialise the background model (e.g. at exam start)."""
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=settings.MOG2_HISTORY,
            varThreshold=settings.MOG2_VAR_THRESHOLD,
            detectShadows=settings.MOG2_DETECT_SHADOWS,
        )

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _classify_zone(self, cy: int, frame_h: int) -> str:
        rel = cy / frame_h
        if settings.ZONE_HEAD_TOP <= rel < settings.ZONE_HEAD_BOTTOM:
            return "head"
        if settings.ZONE_BODY_TOP <= rel < settings.ZONE_BODY_BOTTOM:
            return "body"
        if settings.ZONE_SEAT_TOP <= rel <= settings.ZONE_SEAT_BOTTOM:
            return "seat"
        return "unknown"

    def _classify_magnitude(self, area: int) -> str:
        if area >= settings.MOTION_MIN_AREA_LARGE:
            return "large"
        if area >= settings.MOTION_MIN_AREA_MEDIUM:
            return "medium"
        return "small"

    @staticmethod
    def _overall_level(regions: List[MotionRegion], frame_area: int) -> str:
        if not regions:
            return "none"
        total_area = sum(r.area for r in regions)
        fraction   = total_area / frame_area
        if fraction > 0.22 or any(r.magnitude == "large" for r in regions):
            return "high"
        if fraction > 0.08 or any(r.magnitude == "medium" for r in regions):
            return "medium"
        return "low"
