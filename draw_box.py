import cv2, yaml
from ultralytics import YOLO
from pathlib import Path

# 1) load model and run inference
names = yaml.safe_load(Path("cards.yaml").read_text())["names"]
model = YOLO("/opt/homebrew/runs/detect/train2/weights/best.pt")
name = "roi_sample"
res   = model(name + ".jpg", conf=0.4, device="mps")[0]

# 2) get boxes and sort by x‑coordinate (left → right)
boxes = res.boxes.xyxy.cpu().numpy()
cls   = res.boxes.cls.cpu().numpy().astype(int)
conf  = res.boxes.conf.cpu().numpy()

order = boxes[:, 0].argsort()
boxes, cls, conf = boxes[order], cls[order], conf[order]

# 3) draw with dynamic y‑offset to avoid overlaps
img = cv2.imread(name + ".jpg")
used_y = {}                      # track last y position per x-range

for (x1, y1, x2, y2), cid, c in zip(boxes, cls, conf):
    label = f"{names[cid]} {c:.2f}"
    
    # Compute baseline y that avoids overlap around x1..x2
    x_key = int((x1 + x2) / 10)   # coarse bucket by X
    y_text = int(y1) - 5          # start just above box
    if x_key in used_y and abs(used_y[x_key] - y_text) < 15:
        y_text = used_y[x_key] - 15   # push upward if collision
    used_y[x_key] = y_text
    
    # Draw box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    # Draw background for text
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (int(x1), y_text - th - 4), (int(x1) + tw, y_text), (0,255,0), -1)
    # Put text
    cv2.putText(img, label, (int(x1), y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

cv2.imwrite(name + "_annotated.jpg", img)
print("Saved -> " + name + "_annotated.jpg")
