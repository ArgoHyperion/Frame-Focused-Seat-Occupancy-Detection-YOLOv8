import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, model):
        self.model = YOLO(model)

    # Detect function
    def detectObjects(self,frame, confThreshold=0.4, nmsThreshold=0.3):
        # Ensure the input frame is in RGB format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 inference
        results = self.model.predict(source=frame, conf=confThreshold, iou=nmsThreshold, verbose=False)

        # Prepare the DataFrame
        df = pd.DataFrame(columns=["ClassIds", "Confidences", "TLpoint", "BRpoint"])

        if results and len(results) > 0:
            detections = results[0].boxes.data.cpu().numpy()  # Convert to numpy array
            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                row = {
                    "ClassIds": int(class_id),  # YOLO class IDs start at 0
                    "Confidences": float(confidence),
                    "TLpoint": [int(x1), int(y1)], # Top Left point
                    "BRpoint": [int(x2), int(y2)], # Bottom Right point
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        return df