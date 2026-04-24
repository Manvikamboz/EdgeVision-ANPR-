"""
EdgeSight – Performance Metrics
================================
Compute mAP@50, inference time, FPS, and model size for Task 1 evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """Axis-aligned IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0]);  ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]);  yb = min(box_a[3], box_b[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_average_precision(
    detections: List[Tuple[float, int]], n_gt: int
) -> float:
    """VOC 11-point interpolated Average Precision."""
    if n_gt == 0 or not detections:
        return 0.0
    detections.sort(key=lambda x: x[0], reverse=True)
    tp = np.array([d[1] for d in detections], dtype=float)
    fp = 1 - tp
    tp_cum = np.cumsum(tp);  fp_cum = np.cumsum(fp)
    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
    ap = sum(max(precisions[recalls >= t]) if len(precisions[recalls >= t]) else 0
             for t in np.linspace(0, 1, 11)) / 11
    return float(ap)


class MetricsEngine:
    """Evaluate a VehicleDetector on a labelled test set."""

    def __init__(self, detector, iou_threshold: float = 0.5) -> None:
        self.detector = detector
        self.iou_threshold = iou_threshold
        self._report: Dict = {}

    def compute_map(
        self,
        image_paths: List[str],
        ground_truths: List[List[dict]],
    ) -> Dict[str, float]:
        from collections import defaultdict
        class_detections: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        class_gt_counts: Dict[int, int] = defaultdict(int)

        for img_path, gts in zip(image_paths, ground_truths):
            result = self.detector.detect_image(img_path)
            matched_gt = set()
            for gt in gts:
                class_gt_counts[gt["class_id"]] += 1
            for det in result.detections:
                best_iou, best_idx = 0.0, -1
                for j, gt in enumerate(gts):
                    if gt["class_id"] != det.class_id or j in matched_gt:
                        continue
                    iou = compute_iou(det.bbox, gt["bbox"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, j
                tp = 1 if best_iou >= self.iou_threshold and best_idx >= 0 else 0
                if tp:
                    matched_gt.add(best_idx)
                class_detections[det.class_id].append((det.confidence, tp))

        ap_per_class: Dict[str, float] = {}
        for cls_id, dets in class_detections.items():
            name = self.detector.target_classes.get(cls_id, str(cls_id))
            ap_per_class[name] = compute_average_precision(dets, class_gt_counts[cls_id])

        map50 = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
        return {**ap_per_class, "mAP@50": round(map50, 4)}

    def benchmark_speed(self, image_path: str, iterations: int = 100) -> Dict[str, float]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.detector.detect_frame(img)
            times.append((time.perf_counter() - t0) * 1000)
        times_arr = np.array(times)
        return {
            "iterations": iterations,
            "mean_ms":  round(float(times_arr.mean()), 2),
            "std_ms":   round(float(times_arr.std()),  2),
            "min_ms":   round(float(times_arr.min()),  2),
            "max_ms":   round(float(times_arr.max()),  2),
            "fps_mean": round(1000 / float(times_arr.mean()), 1),
        }

    def model_size(self) -> Dict[str, float]:
        path = Path(self.detector.weights_path)
        if not path.exists():
            return {"size_mb": -1.0, "note": "weights not found on disk"}
        return {"weights_path": str(path), "size_mb": round(path.stat().st_size / (1024**2), 2)}

    def full_report(
        self,
        image_paths: List[str],
        ground_truths: Optional[List[List[dict]]] = None,
        benchmark_image: Optional[str] = None,
        iterations: int = 100,
    ) -> Dict:
        report: Dict = {}
        bm_img = benchmark_image or (image_paths[0] if image_paths else None)
        if bm_img:
            report["speed"] = self.benchmark_speed(bm_img, iterations)
        report["model"] = self.model_size()
        if ground_truths:
            report["detection"] = self.compute_map(image_paths, ground_truths)
        self._report = report
        return report

    def save_report(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._report, f, indent=2)
        print(f"[Metrics] Report saved → {path}")

    def print_report(self) -> None:
        print("\n" + "═"*50)
        print("  EdgeSight – Performance Report (Task 1)")
        print("═"*50)
        for section, data in self._report.items():
            print(f"\n▸ {section.upper()}")
            for k, v in data.items():
                print(f"    {k:<20} {v}")
        print("═"*50 + "\n")
