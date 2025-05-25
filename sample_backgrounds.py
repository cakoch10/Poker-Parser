#!/usr/bin/env python3
"""
sample_backgrounds.py

Usage:
    python sample_backgrounds.py video.mp4 --count 200

This will grab 200 random frames from video.mp4 and drop them into
synthetic_cards/backgrounds/ as bg_0001.jpg, bg_0002.jpg, ...
"""

import cv2
import os
import argparse
import random
from pathlib import Path
from tqdm import tqdm

# -------------------- argument parsing --------------------
parser = argparse.ArgumentParser()
parser.add_argument("video", help="Path to input video (.mp4, .mkv, ...)")
parser.add_argument("--count", type=int, default=200,
                    help="Number of random frames to extract")
parser.add_argument("--outdir", default="synthetic_cards/backgrounds",
                    help="Output directory for background jpgs")
args = parser.parse_args()

video_path = Path(args.video)
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# -------------------- probe video --------------------
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open {video_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.count > frame_count:
    raise ValueError("Requested more frames than exist in the video.")

# Random frame indices to sample
rand_indices = sorted(random.sample(range(frame_count), args.count))

print(f"Sampling {args.count} of {frame_count} frames "
      f"from {video_path.name} …")

# -------------------- extraction loop --------------------
next_idx = rand_indices.pop(0)
saved = 0
pbar = tqdm(total=args.count, unit="frame")

for idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    if idx == next_idx:
        out_name = outdir / f"bg_{saved:05d}.jpg"
        cv2.imwrite(str(out_name), frame)
        saved += 1
        pbar.update(1)
        if rand_indices:
            next_idx = rand_indices.pop(0)
        else:
            break  # done

cap.release()
pbar.close()
print(f"Saved {saved} background frames to {outdir}")
