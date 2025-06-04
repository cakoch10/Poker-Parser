import cv2
import json

IMG = "4boxes.jpg"            # your saved frame
img = cv2.imread(IMG)
rois = cv2.selectROIs("Select each numeric field (ESC when done)", img, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Nice printout
print("ROIs = [")
for (x, y, w, h) in rois:
    print(f"    ({x}, {y}, {w}, {h}),")
print("]")
