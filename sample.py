import cv2
import os
import random
import subprocess

# === Config ===
VIDEO_PATH = "downloads/day2.mkv"
OUTPUT_DIR = "sampled_frames"
NUM_FRAMES = 100
OCR_SCRIPT = "ocr3.py"

# === Create output directory if it doesn't exist ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load video ===
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === Sample frame indices ===
sample_indices = sorted(random.sample(range(total_frames), NUM_FRAMES))

# === Process each sampled frame ===
for idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {idx}")
        continue

    frame_filename = os.path.join(OUTPUT_DIR, f"frame_{idx:05}.jpg")
    cv2.imwrite(frame_filename, frame)

    # Call your OCR script
    print(f"Running OCR on frame {idx}")
    subprocess.run(["python3", OCR_SCRIPT, frame_filename])

cap.release()
