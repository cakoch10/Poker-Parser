import cv2
import os
import numpy as np
from collections import defaultdict
import shutil

# --- CONFIGURATION ---
VIDEO_PATH = "out_8C.mp4"
FRAME_INTERVAL = 30  # Extract 1 frame every 30 frames
ROIS = [
    (1504, 871, 75, 75),  # ROI 1
    (1425, 870, 75, 75),  # ROI 2
    (1587, 872, 75, 75),  # ROI 3
    (1666, 869, 75, 75),  # ROI 4
    (1746, 869, 75, 75),  # ROI 5
    # (190, 710, 66, 66),   # ROI 6
    (100, 710, 66, 66),   # ROI 6
    # (260, 710, 66, 66),   # ROI 7
    (170, 710, 66, 66),   # ROI 7
    # (190, 855, 66, 66),   # ROI 8
    (100, 855, 66, 66),   # ROI 8
    # (260, 855, 66, 66),   # ROI 9
    (170, 855, 66, 66),   # ROI 9
]
OUTPUT_DIR = "extracted_cards"
SIMILARITY_THRESHOLD = 0.95  # 1.0 = perfect match

STANDARD_SIZE = (70, 70)  # all images get rescaled to this size
WHITE_THRESHOLD = 230  # pixel value threshold to consider a pixel as white-ish
WHITE_PERCENT_REQUIRED = 0.30  # 30% of pixels must be white
BRIGHTNESS_THRESHOLD = 40  # filter out very dark frames based on mean brightness

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- UTILITY FUNCTIONS ---
def is_similar(img1, img2, threshold):
    if img1.shape != img2.shape:
        return False
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    max_diff = np.max(diff)
    similarity = 1 - (np.sum(diff) / (img1.size * 255))
    return similarity >= threshold

def has_sufficient_white_pixels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_pixels = np.sum(gray > WHITE_THRESHOLD)
    total_pixels = gray.size
    return (white_pixels / total_pixels) >= WHITE_PERCENT_REQUIRED

def is_not_too_dark(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > BRIGHTNESS_THRESHOLD


# --- MAIN ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

# Store known card images and assign labels
card_buckets = defaultdict(list)
card_paths = defaultdict(list)
card_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if current_frame % FRAME_INTERVAL == 0:
        for idx, (x, y, w, h) in enumerate(ROIS):
            # pad = 15
            # x0 = max(0, x - pad)
            # y0 = max(0, y - pad)
            # x1 = min(frame.shape[1], x + w + pad)
            # y1 = min(frame.shape[0], y + h + pad)
            # roi_crop = frame[y0:y1, x0:x1]

            roi_crop = frame[y:y+h, x:x+w]

            # roi_crop = cv2.resize(roi_crop, STANDARD_SIZE)
            if not has_sufficient_white_pixels(roi_crop):
                continue
            matched = False

            for label, samples in card_buckets.items():
                if any(is_similar(roi_crop, s, SIMILARITY_THRESHOLD) for s in samples):
                    samples.append(roi_crop)
                    matched = True
                    out_path = os.path.join(OUTPUT_DIR, f"card_{label}", f"{current_frame}_roi{idx+1}.jpg")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    cv2.imwrite(out_path, roi_crop)
                    break

            if not matched:
                label = f"{card_counter:02d}"
                card_buckets[label].append(roi_crop)
                out_path = os.path.join(OUTPUT_DIR, f"card_{label}", f"{current_frame}_roi{idx+1}.jpg")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, roi_crop)
                card_counter += 1
            
            card_paths[label].append(out_path)

    current_frame += 1

cap.release()


# Remove buckets with only one image (unmatched singletons)
for label, paths in card_paths.items():
    if len(paths) <= 3:
        for path in paths:
            os.remove(path)
        dir_path = os.path.dirname(paths[0])
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)

remaining_dirs = sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
for new_index, old_name in enumerate(remaining_dirs):
    old_path = os.path.join(OUTPUT_DIR, old_name)
    new_path = os.path.join(OUTPUT_DIR, f"card_{new_index:02d}")
    if old_path != new_path:
        os.rename(old_path, new_path)


print("Done extracting and grouping cards.")