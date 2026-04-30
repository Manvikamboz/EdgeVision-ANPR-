import cv2
import easyocr
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PlateDetector:
    def __init__(
        self, 
        weights_path: str = "models/yolo11n.pt", 
        confidence: float = 0.35,
        img_size: int = 320,
        device: str = "cpu"
    ):
        """
        Initializes the YOLO-based License Plate Detector and OCR Engine.
        """
        from ultralytics import YOLO
        logger.info(f"[EdgeSight] Initializing PlateDetector (YOLO + EasyOCR) using {weights_path}...")
        
        self.model = YOLO(weights_path)
        self.model.to(device)
        self.conf = confidence
        self.img_size = img_size
        
        # Initialize EasyOCR Reader for English
        self.reader = easyocr.Reader(['en'], gpu=(device != "cpu"))
        logger.info("[EdgeSight] PlateDetector Ready ✓")

    def detect_and_read(self, frame, vehicle_bbox=None):
        """
        Detects a license plate. If vehicle_bbox is provided, it searches within that box.
        Otherwise, it searches the entire frame (Fallback Mode).
        """
        if vehicle_bbox is None:
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
        else:
            x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        
        # Crop the search area from the frame
        vehicle_crop = frame[y1:y2, x1:x2]
        
        if vehicle_crop.size == 0:
            return None, "", 0.0

        # Run YOLO inference on the vehicle crop to find the plate
        # NOTE: If using a custom model, the plate class will likely be ID 0.
        # If using a pre-trained COCO model as placeholder, we might filter by a broad set of small objects.
        results = self.model.predict(
            source=vehicle_crop,
            conf=self.conf,
            imgsz=self.img_size,
            verbose=False
        )
        
        if not results or len(results[0].boxes) == 0:
            return None, "", 0.0
            
        # Get the highest confidence detection
        best_box = results[0].boxes[0]
        px1, py1, px2, py2 = best_box.xyxy[0].tolist()
        
        # Plate bounding box relative to the original frame
        plate_bbox = [
            int(x1 + px1),
            int(y1 + py1),
            int(x1 + px2),
            int(y1 + py2)
        ]
        
        # Crop just the plate for OCR
        plate_crop = vehicle_crop[int(py1):int(py2), int(px1):int(px2)]
        
        if plate_crop.size == 0:
            return plate_bbox, "", 0.0
            
        # Enhance plate crop for better OCR accuracy
        plate_crop = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
        
        # Read text using EasyOCR
        ocr_results = self.reader.readtext(plate_gray)
        
        if not ocr_results:
            return plate_bbox, "", 0.0
            
        # Get the highest confidence text
        best_ocr = max(ocr_results, key=lambda x: x[2])
        text = ''.join(e for e in best_ocr[1].upper() if e.isalnum())
        conf = best_ocr[2]
        
        if len(text) < 3: 
            return plate_bbox, "", 0.0
            
        return plate_bbox, text, float(conf)
