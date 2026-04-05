from ultralytics import YOLO
import cv2
import numpy as np

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load pretrained YOLOv8 model
        self.model = YOLO(model_path)
        self.model.to("cpu")

    def detect(self, image: np.ndarray):
        """
        image: RGB np.ndarray
        returns: list of detections [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model(image, device="cpu")[0]  # take first result

        detections = []

        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confs[i]
            cls = int(classes[i])

            detections.append([x1, y1, x2, y2, conf, cls])

        return detections
