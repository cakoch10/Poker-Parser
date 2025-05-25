#!/usr/bin/env python
"""
generation.py

Creates synthetic frames by pasting random card combos onto background
images according to a JSON template. Saves YOLO labels automatically.
"""

import json, os, random, itertools
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import yaml

# -------- CONFIG --------
DATA_ROOT = Path("synthetic_cards")
OUT_ROOT  = Path("cards_dataset")
TEMPLATE  = json.load(open(DATA_ROOT/"templates"/"pokerstars.json"))

N_IMAGES  = 15000           # total synthetic frames to make
MAX_PLAYERS = 3            # how many player slots to fill each frame
ROT_JITTER = 2             # ± degrees rotation jitter
SCALE_JITTER = 0.05        # ± relative scale
# ------------------------

BG_DIR   = DATA_ROOT / "backgrounds"
CARD_DIR = DATA_ROOT / "card_crops"   # contains 52 subdirs (AS, 2S, …)

out_imgs   = OUT_ROOT / "images" / "train"
out_labels = OUT_ROOT / "labels" / "train"
out_imgs.mkdir(parents=True, exist_ok=True)
out_labels.mkdir(parents=True, exist_ok=True)

# ➊ Build a dict: class_name -> list_of_files  and a numeric class map
card_dirs = sorted([d for d in CARD_DIR.iterdir() if d.is_dir()])
card_images = {d.name: sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")) for d in card_dirs}
class_map = {class_name: idx for idx, class_name in enumerate(card_images)}

slots = TEMPLATE["slots"]
slot_names = list(slots.keys())

# ➋ Utility to sample a random (class_name, path)
def sample_random_card():
    class_name = random.choice(list(card_images))
    img_path   = random.choice(card_images[class_name])
    class_id   = class_map[class_name]
    return class_name, class_id, img_path

def random_transform(card, slot_box):
    """Apply tiny rotation/scale jitter so training is not too deterministic."""
    cx, cy, w, h = slot_box
    angle = random.uniform(-ROT_JITTER, ROT_JITTER)
    scale = 1.0 + random.uniform(-SCALE_JITTER, SCALE_JITTER)
    new_w, new_h = int(w * scale), int(h * scale)
    card = card.resize((new_w, new_h), Image.BICUBIC).rotate(angle, expand=True, resample=Image.BICUBIC)
    return card

def paste_card(bg, card_img, slot_box):
    cx, cy, w, h = slot_box
    card_w, card_h = card_img.size
    # Center card inside the slot
    x = int(cx + w/2 - card_w/2)
    y = int(cy + h/2 - card_h/2)
    bg.paste(card_img, (x, y), card_img.convert("RGBA"))
    # Return actual bbox of pasted card for YOLO
    return (x, y, x+card_w, y+card_h)

def bbox_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    cx = (xmin + xmax) / 2 / img_w
    cy = (ymin + ymax) / 2 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return cx, cy, bw, bh

bg_files = list(BG_DIR.glob("*.jpg*"))
print(f"{len(bg_files)} backgrounds, {len(card_images)} card crops.")

for i in tqdm(range(N_IMAGES), desc="Generating"):
    bg_path = random.choice(bg_files)
    bg = Image.open(bg_path).convert("RGB").resize(TEMPLATE["image_size"], Image.BICUBIC)

    chosen_slots = random.sample(slot_names, k=MAX_PLAYERS*2)  # 2 cards per player
    random.shuffle(chosen_slots)  # avoid same ordering every frame

    labels = []

    for slot_name in chosen_slots:
        class_name, class_id, card_path = sample_random_card()
        card_img = Image.open(card_path).convert("RGBA")

        slot_box  = slots[slot_name]   # [x, y, w, h]
        slot_box  = tuple(slot_box)

        card_img  = random_transform(card_img, slot_box)
        xmin, ymin, xmax, ymax = paste_card(bg, card_img, slot_box)
        yolo_box = bbox_yolo(xmin, ymin, xmax, ymax, *TEMPLATE["image_size"])

        labels.append(f"{class_id} {' '.join(f'{v:.6f}' for v in yolo_box)}")

    # --- save outputs ---
    img_name   = f"synth_{i:05d}.jpg"
    label_name = f"synth_{i:05d}.txt"
    bg.save(out_imgs / img_name, quality=95)
    (out_labels / label_name).write_text("\n".join(labels))

print("Synthetic generation complete.")
