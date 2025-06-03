#!/usr/bin/env python3
"""
ocr3.py – Extract name, position, stack, and action from PokerStars HUD tiles.

Usage:
    python ocr3.py frame.jpg
    # or integrate process_frame(frame) in your video loop
"""

from __future__ import annotations
import cv2, pytesseract, numpy as np, re, sys, pprint
from pathlib import Path
import os

import easyocr
reader = easyocr.Reader(['en'], gpu=True)  # or gpu=True if using CUDA/MPS

# supress annoying pin memory warning
import warnings
warnings.filterwarnings("ignore", module="torch.utils.data.dataloader")



# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

DEBUG_BIN_DIR = "debug_bin"
os.makedirs(DEBUG_BIN_DIR, exist_ok=True)


# Bounding boxes
TILES = {
    "P4": (92, 416, 324, 139),
    "P3": (94, 561, 321, 140),
    "P2": (95, 704, 320, 141),
    "P1": (96, 849, 320, 141)
    # "ONE": (96, 525, 317, 29)
}     

# Sub‑ROI offsets INSIDE **one tile** – fractions of tile (w,h)
# OFFSETS = {
#     'name': (0.015, 0.496, 0.787, 0.252),
#     'pos': (0.809, 0.504, 0.182, 0.252),
#     'stack': (0.691, 0.77, 0.306, 0.237),
#     'action': (0.015, 0.77, 0.667, 0.216)
# }

OFFSETS = {
    'name': (0.015, 0.496, 0.787, 0.26),
    'pos': (0.809, 0.49, 0.182, 0.252),
    'stack': (0.691, 0.77, 0.306, 0.237),
    'action': (0.015, 0.77, 0.667, 0.216)
}

# Threshold (fraction of white pixels) to decide "has text"
CONTENT_THRESH = {
    "name":   0.01,
    "pos":    0.01,
    "stack":  0.01,
    "action": 0.01,
}

POT_ROI    = (1488, 944, 338, 43)


# Path to Tesseract binary (Homebrew default). Comment out if auto‑detected.
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# ------------------------------------------------------------------
# IMAGE PRE‑PROCESSING
# ------------------------------------------------------------------
def binarize(crop: np.ndarray) -> np.ndarray:
    """Adaptive threshold + auto‑invert so glyphs are white on black."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7
    )
    if th.mean() < 127:          # background ended up black → invert
        th = 255 - th
    return cv2.dilate(th, np.ones((2, 2), np.uint8), iterations=1)

def has_content(bin_img: np.ndarray, kind: str) -> bool:
    return (bin_img > 0).mean() > CONTENT_THRESH[kind]

def preprocess_crop(crop):
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    return crop

# ------------------------------------------------------------------
# OCR HELPERS
# ------------------------------------------------------------------
WHITELIST = {
    # note the outer double‑quotes and the escaped single‑quote \'
    "name":   "--psm 8 -c tessedit_char_whitelist=\"ABCDEFGHIJKLMNOPQRSTUVWXYZ-'\"",
    "pos":    "--psm 6 -c tessedit_char_whitelist=UTGJHCO+DSB210O",
    "stack":  "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789KMkm.,",
    "action": "--psm 8 -c tessedit_char_whitelist=BETCHECKRAISEALLIN0123456789, ",
    "pot":  "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789KMkm.,"
}

# def ocr_field(bin_img: np.ndarray, kind: str, index=None) -> str:
#     upscale = cv2.resize(bin_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#     txt = pytesseract.image_to_string(upscale, config=WHITELIST[kind]).strip()
#     # txt = pytesseract.image_to_string(bin_img, config=WHITELIST[kind]).strip()
#     if index is not None:
#         filename = f"{kind}_{index}.png"
#         cv2.imwrite(os.path.join(DEBUG_BIN_DIR, filename), bin_img)

#     return txt.upper()

def ocr_field(image, kind, index=None):
    """
    OCR wrapper for EasyOCR.
    `image` should be a color or grayscale crop (not binarized).
    """
    results = None
    if kind == "pos":
        results = reader.readtext(image, detail=0, allowlist="UTGJHC+DSB21")
        # results = reader.readtext(image, detail=0, allowlist="D")
    else:
        results = reader.readtext(image, detail=0)
    txt = results[0] if results else None
    print(results)
    return txt.upper().strip() if txt else None


def _num_to_int(token: str) -> int:
    token = token.replace(",", "")
    if token.endswith("K"):
        return int(float(token[:-1]) * 1_000)
    if token.endswith("M"):
        return int(float(token[:-1]) * 1_000_000)
    return int(token)

def clean_stack(txt: str | None) -> int | None:
    if not txt:
        return None
    m = re.match(r"(\d+(?:\.\d+)?)([KM]?)", txt.replace(",", ""))
    return _num_to_int("".join(m.groups())) if m else None

def parse_action(txt: str | None) -> dict | None:
    if not txt:
        return None
    txt = txt.replace(",", "").strip()

    if txt.startswith("CHECK"):
        return {"type": "CHECK", "amount": 0}

    if m := re.match(r"BET\s+(\d+K?)", txt):
        return {"type": "BET", "amount": _num_to_int(m[1])}

    if m := re.match(r"RAISE(?:\s+TO)?\s+(\d+K?)", txt):
        return {"type": "RAISE", "amount": _num_to_int(m[1])}

    if m := re.match(r"ALL\s*IN(?:\s+FOR)?\s+(\d+K?)", txt):
        return {"type": "ALLIN", "amount": _num_to_int(m[1])}

    if m := re.match(r"CALL\s+(\d+K?)", txt):
        return {"type": "CALL", "amount": _num_to_int(m[1])}

    return None

# ------------------------------------------------------------------
# ROI CALCULATION
# ------------------------------------------------------------------
def sub_rois(tile_bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    x, y, w, h = tile_bbox
    rois = {}
    for k, (xf, yf, wf, hf) in OFFSETS.items():
        rx, ry = int(x + xf * w), int(y + yf * h)
        rw, rh = int(wf * w), int(hf * h)
        rois[k] = (rx, ry, rw, rh)
    return rois

# ------------------------------------------------------------------
# MAIN EXTRACTOR FOR ONE FRAME
# ------------------------------------------------------------------
def process_frame(frame: np.ndarray) -> dict[str, dict]:
    """
    Returns:
        { 'SB': {'name':'ZHENG','pos':'SB','stack':72000,'action': {...}}, ... }
    """
    players = {}
    for seat, tile in TILES.items():
        rois = sub_rois(tile)
        info = {}
        for kind, roi in rois.items():
            x,y,w,h = roi
            crop = frame[y:y+h, x:x+w]
            crop = preprocess_crop(crop)
            # bin_img = binarize(crop)
            # if not has_content(bin_img, kind):
            #     print("FAIL")
            #     continue

            raw = ocr_field(crop, kind, index=x)
            print(raw)
            if kind == "stack":
                info[kind] = clean_stack(raw)
            elif kind == "action":
                info[kind] = parse_action(raw)
            else:
                info[kind] = raw

        if info:                                # skip empty tiles
            players[seat] = info
        print(info)
    # Extract and OCR the pot box
    players["pot"] = None
    x, y, w, h = POT_ROI
    pot_crop = frame[y:y+h, x:x+w]
    pot_crop = preprocess_crop(pot_crop)
    # bin_pot = binarize(pot_crop)  # same binarization as other fields
    # if not has_content(bin_pot, "stack"):
    #     return players
    pot_value = ocr_field(pot_crop, "pot")
    players["pot"] = clean_stack(pot_value)
    return players

# ------------------------------------------------------------------
# TEST / CLI
# ------------------------------------------------------------------


def debug_draw(frame: np.ndarray, results: dict[str, dict]) -> np.ndarray:
    dbg = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for seat, tile in TILES.items():
        x,y,w,h = tile

        rois = sub_rois(tile)
        for kind, (x2, y2, w2, h2) in rois.items():
            color = {
                "name": (0, 255, 0),
                "pos": (255, 255, 0),
                "stack": (0, 255, 255),
                "pot": (0, 255, 255),
                "action": (255, 0, 255),
            }[kind]
            cv2.rectangle(dbg, (x2, y2), (x2 + w2, y2 + h2), color, 1)
        cv2.rectangle(dbg, (x,y), (x+w,y+h), (0,255,0), 1)
        if seat in results:
            t = results[seat]
            label = f"{t.get('name','')} {t.get('pos','')} {t.get('stack','')}"
            cv2.putText(dbg, label, (x+5, y+15), font, 0.5, (0,255,0), 1)
            if act := t.get("action"):
                cv2.putText(dbg, f"{act['type']} {act['amount']}",
                            (x+5, y+h-8), font, 0.5, (0,255,0), 1)
    # draw box around pot
    color = (0, 255, 255)
    x,y,w,h = POT_ROI
    cv2.rectangle(dbg, (x,y), (x+w,y+h), color, 1)
    return dbg

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_parser.py frame.jpg")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    img = cv2.imread(str(img_path))
    assert img is not None, f"Could not read {img_path}"

    res = process_frame(img)
    pprint.pp(res)

    dbg = debug_draw(img, res)
    out = img_path.with_name(img_path.stem + "_annot.jpg")
    cv2.imwrite(str(out), dbg)
    print("Saved debug overlay:", out)
