from ultralytics import YOLO
import cv2
import numpy as np
import time

# import onnxruntime as ort
# print("ONNX Runtime version:", ort.__version__)

class YOLOv8Detector:
    def __init__(self, node, model_path="yolov8n.onnx"):
        # Load pretrained YOLOv8 model
        # self.model = YOLO(model_path)
        self.model = YOLO(model_path, task="detect")    # ONNX model (TODO: still torch model, low fps)
        # self.model.to("cpu")                          # Not needed for ONNX
        print(type(self.model.model))                   # ONNX / TORCH
        self.node = node
        self.prev_time = time.time()

    def detect(self, image: np.ndarray):
        """
        image: RGB np.ndarray
        returns: list of detections [x1, y1, x2, y2, confidence, class_id]
        """
        start = time.time()
        # results = self.model(image, device="cpu")[0]  # take first result
        # results = self.model(image, device="cpu", conf=0.4)[0]
        results = self.model.predict(image, device="cpu", conf=0.4)[0]

        end = time.time()

        # inference time
        inference_time = end - start

        # FPS
        now = time.time()
        fps = 1.0 / (now - self.prev_time)
        self.prev_time = now
        print(f"[YOLO] Inference: {inference_time*1000:.1f} ms | FPS: {fps:.2f}")
        self.node.get_logger().info(
            f"[YOLO] Inference: {inference_time*1000:.1f} ms | FPS: {fps:.2f}",
            throttle_duration_sec=1.0
            )

        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        # visualize
        annotated = results.plot()
        cv2.putText(annotated,
                    f"[YOLO] Inference: {inference_time*1000:.1f} ms | FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
        )
        cv2.imshow(f"YOLO", annotated)
        cv2.waitKey(1)

        # debug class names
        # for cls in results.boxes.cls:
        #     print(results.names[int(cls)])

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confs[i]
            cls = int(classes[i])

            class_name = results.names[cls]
            # if class_name not in ["chair", "couch", "cup", "backpack"]:
            #     continue

            print(f"[DETECTED] {class_name} | conf={conf:.2f}")
            self.node.get_logger().info(f"{class_name} detected ({conf:.2f})")
            detections.append([x1, y1, x2, y2, conf, cls])

        return detections

