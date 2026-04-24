"""
EdgeSight – Vehicle Detector
=============================
Wraps YOLOv8n (Ultralytics/PyTorch) and returns structured Detection objects.

Supports:
    • Single image
    • Batch of images
    • Video file  (frame-by-frame generator)
    • Live webcam / RTSP stream
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Generator, List, Optional

import cv2
import numpy as np
import yaml

from src.preprocessor import Preprocessor


# ─── Data models ─────────────────────────────────────────────────────────────
@dataclass
class Detection:
    """Single vehicle detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]          # [x1, y1, x2, y2]  pixel coords
    bbox_norm: List[float]   # [cx, cy, w, h]     normalised 0-1
    frame_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (f"Detection(class={self.class_name!r}, "
                f"conf={self.confidence:.2f}, bbox=[{x1},{y1},{x2},{y2}])")


@dataclass
class FrameResult:
    """Aggregated result for one frame / image."""
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_shape: tuple = ()

    @property
    def count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "vehicle_count": self.count,
            "detections": [d.to_dict() for d in self.detections],
        }


# ─── Detector ────────────────────────────────────────────────────────────────
class VehicleDetector:
    """
    Lightweight YOLOv8n vehicle detector for edge deployment.
    Uses PyTorch as backend via the Ultralytics library.

    Target classes (COCO): car(2), motorcycle(3), bus(5), truck(7)
    """

    DEFAULT_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        confidence: float = 0.45,
        iou_threshold: float = 0.45,
        target_classes: Optional[dict] = None,
        img_size: int = 640,
        device: str = "cpu",
        preprocessor: Optional[Preprocessor] = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = weights_path
        self.conf = confidence
        self.iou = iou_threshold
        self.img_size = img_size
        self.device = device
        self.target_classes = target_classes or self.DEFAULT_VEHICLE_CLASSES
        self.preprocessor = preprocessor or Preprocessor()

        print(f"[EdgeSight] Loading YOLOv8n: {weights_path}  device={device}")
        self.model = YOLO(weights_path)
        self.model.to(device)
        print("[EdgeSight] Model ready ✓")

    @classmethod
    def from_config(cls, config_path: str | Path) -> "VehicleDetector":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        m = cfg["model"]
        weights = m.get("custom_weights") or m.get("weights", "yolov8n.pt")
        class_names: dict = {int(k): v for k, v in cfg.get("vehicle_class_names", {}).items()}
        return cls(
            weights_path=weights,
            confidence=m.get("confidence_threshold", 0.45),
            iou_threshold=m.get("iou_threshold", 0.45),
            target_classes=class_names or None,
            img_size=m.get("img_size", 640),
            device=m.get("device", "cpu"),
            preprocessor=Preprocessor.from_config(cfg),
        )

    # ── Core inference ────────────────────────────────────────────────────────
    def detect_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        """
        Run inference on a single BGR numpy frame.

        Enhancement (CLAHE / denoise) is applied at the original resolution.
        YOLO handles its own internal letterbox resize via imgsz, so the
        returned bounding-box coordinates are in the original frame's pixel
        space — draw_detections() can draw them directly without any scaling.
        """
        h, w = frame.shape[:2]

        # Enhance quality (CLAHE, denoise) — do NOT resize here
        enhanced = self.preprocessor.enhance_only(frame.copy())

        t0 = time.perf_counter()
        results = self.model.predict(
            source=enhanced,          # original resolution, enhanced
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,      # YOLO letterboxes internally → correct coords
            classes=list(self.target_classes.keys()),
            verbose=False,
        )
        inf_ms = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.target_classes:
                    continue
                # Coordinates are now in original frame pixel space ✓
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Clamp to frame boundaries
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=self.target_classes[cls_id],
                    confidence=round(float(box.conf[0]), 4),
                    bbox=[x1, y1, x2, y2],
                    bbox_norm=[round(cx,4), round(cy,4), round(bw,4), round(bh,4)],
                    frame_id=frame_id,
                ))

        return FrameResult(
            frame_id=frame_id,
            detections=detections,
            inference_time_ms=inf_ms,
            image_shape=(h, w),
        )

    def detect_image(self, image_path: str | Path) -> FrameResult:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load: {image_path}")
        return self.detect_frame(img, frame_id=0)

    def detect_video(
        self, source: str | int, skip_frames: int = 0
    ) -> Generator[FrameResult, None, None]:
        """Yield FrameResult for every selected frame from a video / stream."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % (skip_frames + 1) == 0:
                    yield self.detect_frame(frame, frame_id=frame_id)
                frame_id += 1
        finally:
            cap.release()

    # ── Visualisation ─────────────────────────────────────────────────────────
    def draw_detections(
        self, frame: np.ndarray, result: FrameResult,
        thickness: int = 2, font_scale: float = 0.6,
    ) -> np.ndarray:
        """Return annotated copy of the frame."""
        palette = {2: (0,200,255), 3: (0,255,120), 5: (255,80,80), 7: (200,0,255)}
        vis = frame.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = palette.get(det.class_id, (200,200,200))
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, thickness)
            label = f"{det.class_name} {det.confidence:.0%}"
            (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(vis, (x1,y1-th-6), (x1+tw+4,y1), color, -1)
            cv2.putText(vis, label, (x1+2,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(vis,
            f"Vehicles: {result.count}  |  {result.inference_time_ms:.1f} ms",
            (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return vis
