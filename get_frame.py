#!/usr/bin/env python3
"""
Usage:
    python extract_frame.py pokerstars_video.mp4 8:54

This will extract the closest frame at 8 minutes 54 seconds from the video
and save it as 'frame_8_54.jpg'.
"""

import cv2
import sys
import os

def timestamp_to_msec(timestamp_str):
    """Convert a timestamp string like '8:54' or '01:12:30' to milliseconds."""
    parts = [int(p) for p in timestamp_str.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return (minutes * 60 + seconds) * 1000
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return (hours * 3600 + minutes * 60 + seconds) * 1000
    else:
        raise ValueError("Invalid timestamp format. Use MM:SS or HH:MM:SS")

def extract_frame_at_timestamp(video_path, timestamp_str):
    msec = timestamp_to_msec(timestamp_str)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video file: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, msec)
    ret, frame = cap.read()

    if ret:
        # Sanitize timestamp for filename
        ts_clean = timestamp_str.replace(":", "_")
        basename = os.path.splitext(os.path.basename(video_path))[0]
        out_filename = f"{basename}_frame_{ts_clean}.jpg"
        cv2.imwrite(out_filename, frame)
        print(f"✅ Frame saved to: {out_filename}")
    else:
        print("❌ Could not read frame at specified timestamp.")

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_frame.py <video_path> <timestamp>")
        sys.exit(1)

    video_file = sys.argv[1]
    timestamp = sys.argv[2]

    extract_frame_at_timestamp(video_file, timestamp)
