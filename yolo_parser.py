import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Load class name mapping from cards.yaml
names = yaml.safe_load(Path("cards.yaml").read_text())["names"]

# Load YOLOv8 model once
model = YOLO("weights/best.pt")


def detect_cards(frame, conf_threshold=0.01):
    result = model(frame, conf=conf_threshold, device="mps", verbose=False)[0]

    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()

    parsed = []
    for cls_id, box, conf in zip(cls_ids, boxes, confs):
        x1, y1, x2, y2 = box
        parsed.append({
            "class_id": int(cls_id),
            "name": names[cls_id],
            "confidence": float(conf),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]  # (x, y, w, h)
        })

    return parsed
