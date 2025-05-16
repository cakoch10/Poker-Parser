import cv2

name = "out.jpg"
image = cv2.imread(name)
image_copy = image.copy()

# List to store multiple ROIs
rois = []

print("Draw multiple ROIs. Press ENTER after each selection. Press ESC when done.")
while True:
    roi = cv2.selectROI("Select ROI", image_copy, fromCenter=False, showCrosshair=True)
    if roi == (0, 0, 0, 0):  # ESC or empty selection ends the loop
        break
    rois.append(roi)
    x, y, w, h = roi
    # Optionally draw rectangles on the copy to visualize selections
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.destroyAllWindows()

# Print and optionally save each ROI
for i, (x, y, w, h) in enumerate(rois):
    print(f"ROI {i+1}: x={x}, y={y}, width={w}, height={h}")
    roi_crop = image[y:y+h, x:x+w]
    cv2.imwrite(f"roi_crop_{i+1}.jpg", roi_crop)
