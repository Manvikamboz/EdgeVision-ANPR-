"""
EdgeSight – Shared Utilities
==============================
File I/O, logging, drawing helpers, and YOLO label conversion.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def get_logger(name: str = "EdgeSight", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p


def list_images(directory: str | Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    d = Path(directory)
    return sorted(p for ext in exts for p in d.rglob(f"*{ext}"))


def timestamped_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext.lstrip('.')}"


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info(f"Saved JSON → {path}")


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_image(path: str | Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        logger.warning(f"Could not load image: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    cv2.imwrite(str(path), img)


def get_video_info(source: str | int) -> Dict:
    cap = cv2.VideoCapture(source)
    info = {
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def create_video_writer(
    output_path: str | Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir(Path(output_path).parent)
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,80), 2, cv2.LINE_AA)
    return frame


# ── YOLO label helpers ────────────────────────────────────────────────────────
def yolo_bbox_to_pixel(cx, cy, bw, bh, img_w, img_h) -> List[int]:
    x1 = int((cx - bw/2) * img_w);  y1 = int((cy - bh/2) * img_h)
    x2 = int((cx + bw/2) * img_w);  y2 = int((cy + bh/2) * img_h)
    return [x1, y1, x2, y2]


def pixel_to_yolo_bbox(x1, y1, x2, y2, img_w, img_h) -> List[float]:
    return [round(((x1+x2)/2)/img_w, 6), round(((y1+y2)/2)/img_h, 6),
            round((x2-x1)/img_w, 6),    round((y2-y1)/img_h, 6)]


def parse_yolo_label(label_path: str | Path) -> List[dict]:
    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            annotations.append({"class_id": cls_id, "cx": cx, "cy": cy, "bw": bw, "bh": bh})
    return annotations
