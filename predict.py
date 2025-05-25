from ultralytics import YOLO
import yaml
from pathlib import Path

# this model was trained as ~/D/Poker-Parser►yolo detect train model=yolov8n.pt data=cards.yaml imgsz=416 batch=4 epochs=50 workers=2 device=mps                                                                                                               (base) 1309.958s (main|💩?↑5) 12:36

model = YOLO("/opt/homebrew/runs/detect/train2/weights/best.pt")
result = model("out.jpg", conf=0.01, device="mps")[0]

# 2) get class‑IDs, confidences, and boxes
cls_ids = result.boxes.cls.cpu().numpy().astype(int)        # e.g. [12  7 30]
confs   = result.boxes.conf.cpu().numpy()                   # confidence scores
boxes   = result.boxes.xyxy.cpu().numpy()                   # x1 y1 x2 y2

# 3) load the names list from your cards.yaml
names = yaml.safe_load(Path("cards.yaml").read_text())["names"]  # {0:'2C', …, 51:'KC'}

# 4) pretty‑print
print(f"{'id':>3} {'card':<3} {'conf':>6}   (x1, y1) – (x2, y2)")
print("-"*50)
for i, cid in enumerate(cls_ids):
    label = names[cid]
    c = confs[i]
    x1, y1, x2, y2 = boxes[i]
    print(f"{cid:>3} {label:<3} {c:6.2f}   ({x1:4.0f},{y1:4.0f}) – ({x2:4.0f},{y2:4.0f})")