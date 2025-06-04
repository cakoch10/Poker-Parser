# calc_offsets.py
from pprint import pprint

def calc_offsets(tile, subs, labels=None, round_to=3):
    """
    tile : (x, y, w, h)      -- outer HUD tile in frame coords
    subs : [(x, y, w, h), …] -- sub‑ROIs in frame coords
    labels : ["name","pos",…] -- same length as subs (optional)
    """
    tx, ty, tw, th = tile
    out = {}
    for idx, (sx, sy, sw, sh) in enumerate(subs):
        xf = round((sx - tx) / tw, round_to)
        yf = round((sy - ty) / th, round_to)
        wf = round(sw / tw, round_to)
        hf = round(sh / th, round_to)
        key = labels[idx] if labels else f"roi_{idx}"
        out[key] = (xf, yf, wf, hf)
    return out


if __name__ == "__main__":
    # ----- example values -----
    tile_box = (94, 561, 321, 140)

    sub_boxes = [
        (97, 485, 255, 35),   # name
        (354, 486, 59, 35),   # position
        (345, 668, 70, 31),   # stack
        (97, 523, 216, 30)   # action
    ]
    labels = ["name", "pos", "stack", "action"]
    # --------------------------

    offsets = calc_offsets(tile_box, sub_boxes, labels)
    print("OFFSETS =")
    pprint(offsets, sort_dicts=False)
