"""
EdgeSight – Image / Video Preprocessor
======================================
Applies enhancement pipeline before detection.

Two modes:
  • enhance_only()  – CLAHE + denoise only, NO resize.
                      Use this before passing frames to YOLO so that
                      coordinates returned by the model are in the
                      original frame's pixel space (correct bounding boxes).
  • process_frame() – full pipeline including resize (kept for data-prep use).

All transforms operate on BGR numpy arrays (OpenCV convention).
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class Preprocessor:
    """
    Lightweight preprocessing pipeline optimised for edge deployment.

    Parameters
    ----------
    target_size   : (width, height) to resize every frame.
    apply_clahe   : contrast-limited adaptive histogram equalisation.
    clahe_clip    : CLAHE clip limit (higher → more contrast, more noise).
    clahe_tile    : CLAHE tile grid size.
    apply_denoise : Non-local means denoising (slower – disable on edge).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        apply_clahe: bool = True,
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
        apply_denoise: bool = False,
    ) -> None:
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.apply_denoise = apply_denoise

        if apply_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=clahe_clip, tileGridSize=clahe_tile
            )

    def enhance_only(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply quality enhancements WITHOUT resizing.

        Use this before passing frames to YOLO for detection so that YOLO
        handles its own internal letterbox resize and returns bounding-box
        coordinates in the original frame's pixel space.
        """
        if self.apply_denoise:
            frame = self._denoise(frame)
        if self.apply_clahe:
            frame = self._apply_clahe(frame)
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Full pipeline: resize → denoise → CLAHE.  Use for data preparation."""
        frame = self._resize(frame)
        if self.apply_denoise:
            frame = self._denoise(frame)
        if self.apply_clahe:
            frame = self._apply_clahe(frame)
        return frame

    def process_image_path(self, path: str | Path) -> Optional[np.ndarray]:
        """Load an image from disk and preprocess it."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return self.process_frame(img)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        tw, th = self.target_size
        if (w, h) == (tw, th):
            return frame
        interp = cv2.INTER_AREA if (w > tw or h > th) else cv2.INTER_LINEAR
        return cv2.resize(frame, (tw, th), interpolation=interp)

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @classmethod
    def from_config(cls, cfg: dict) -> "Preprocessor":
        """Construct from a parsed YAML config dict."""
        pp = cfg.get("preprocessing", {})
        size = tuple(pp.get("target_size", [640, 640]))
        return cls(
            target_size=size,
            apply_clahe=pp.get("clahe", True),
            clahe_clip=pp.get("clahe_clip_limit", 2.0),
            clahe_tile=tuple(pp.get("clahe_tile_grid", [8, 8])),
            apply_denoise=pp.get("denoise", False),
        )
