"""
EdgeSight – Main CLI Entry Point
==================================
Usage
-----
# Single image
python main.py --source data/raw/sample.jpg

# Image folder
python main.py --source data/raw/images/ --save

# Video file
python main.py --source data/raw/traffic.mp4 --save --json

# Webcam
python main.py --source 0 --live

# Custom weights + metrics benchmark
python main.py --source data/raw/sample.jpg --weights models/best_vehicle.pt --metrics
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from src.detector import VehicleDetector
from src.metrics import MetricsEngine
from src.utils import (
    ensure_dir, get_logger, get_video_info,
    create_video_writer, list_images, save_json,
)

logger     = get_logger()
CONFIG     = "configs/vehicle_detection.yaml"
OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EdgeSight – Vehicle Detection (Task 1)")
    p.add_argument("--source",   required=True, help="Image/video path, folder, or webcam index.")
    p.add_argument("--weights",  default=None,  help="Custom .pt weights path.")
    p.add_argument("--conf",     type=float, default=None, help="Confidence threshold override.")
    p.add_argument("--device",   default=None,  help="'0' for GPU or 'cpu'.")
    p.add_argument("--config",   default=CONFIG, help=f"YAML config (default: {CONFIG}).")
    p.add_argument("--save",     action="store_true", help="Save annotated output.")
    p.add_argument("--live",     action="store_true", help="Show live window.")
    p.add_argument("--skip",     type=int, default=0,  help="Skip N frames between detections.")
    p.add_argument("--metrics",  action="store_true", help="Run speed benchmark.")
    p.add_argument("--json",     action="store_true", help="Save detections as JSON.")
    return p.parse_args()


def run_image(detector: VehicleDetector, args: argparse.Namespace) -> None:
    src = Path(args.source)
    paths = list_images(src) if src.is_dir() else [src]
    logger.info(f"Processing {len(paths)} image(s)…")
    ensure_dir(OUTPUT_DIR / "images")
    all_results = []

    for img_path in paths:
        result  = detector.detect_image(img_path)
        img     = cv2.imread(str(img_path))
        annotated = detector.draw_detections(img, result)
        logger.info(f"{img_path.name} → {result.count} vehicle(s) [{result.inference_time_ms:.1f} ms]")

        if args.save:
            cv2.imwrite(str(OUTPUT_DIR / "images" / img_path.name), annotated)
        if args.live:
            cv2.imshow("EdgeSight", annotated)
            cv2.waitKey(0)
        all_results.append(result.to_dict())

    if args.json:
        save_json(all_results, OUTPUT_DIR / "detections.json")
    cv2.destroyAllWindows()


def run_video(detector: VehicleDetector, args: argparse.Namespace) -> None:
    source = int(args.source) if str(args.source).isdigit() else args.source
    info   = get_video_info(source)
    fps, w, h = info["fps"] or 30, info["width"], info["height"]

    writer = None
    if args.save:
        ensure_dir(OUTPUT_DIR / "videos")
        name = "webcam_out.mp4" if isinstance(source, int) else Path(str(source)).stem + "_out.mp4"
        writer = create_video_writer(OUTPUT_DIR / "videos" / name, fps, w, h)

    frame_results = []
    prev = time.time()
    for result in detector.detect_video(source, skip_frames=args.skip):
        frame_results.append(result.to_dict())
        cur_fps = 1 / (time.time() - prev + 1e-9)
        prev = time.time()
        logger.info(f"Frame {result.frame_id} → {result.count} vehicle(s) "
                    f"[{result.inference_time_ms:.1f} ms | {cur_fps:.1f} FPS]")

    if writer:
        writer.release()
    if args.json:
        save_json(frame_results, OUTPUT_DIR / "detections_video.json")


def main() -> None:
    args     = parse_args()
    detector = VehicleDetector.from_config(args.config)
    if args.weights:
        from ultralytics import YOLO
        detector.model = YOLO(args.weights)
    if args.conf is not None:
        detector.conf = args.conf
    if args.device is not None:
        detector.device = args.device

    if args.metrics:
        engine = MetricsEngine(detector)
        report = engine.full_report([args.source])
        engine.save_report(OUTPUT_DIR / "metrics.json")
        engine.print_report()

    src = args.source
    is_video = str(src).isdigit() or str(src).endswith((".mp4",".avi",".mov",".mkv",".webm"))
    run_video(detector, args) if is_video else run_image(detector, args)


if __name__ == "__main__":
    main()
