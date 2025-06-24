# yolo_detector.py
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

class YOLODetector:
    def __init__(self, model_path="yolo11n.pt"):
        """Initialize YOLO model."""
        self.model = YOLO(model_path)

    def detect_objects(self, image):
        """Detect objects in the provided image and return results."""
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            # Convert RGB to BGR for OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            # Perform detection
            results = self.model(img_array)
            # Extract detected objects and confidence scores
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    label = result.names[class_id]
                    detections.append({"label": label, "confidence": confidence})
            return detections
        except Exception as e:
            return {"error": f"Object detection failed: {str(e)}"}
