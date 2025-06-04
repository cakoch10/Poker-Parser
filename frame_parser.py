import json
import cv2
from typing import Dict
from yolo_parser import detect_cards 
from ocr3 import process_frame 


def assign_cards_to_sources(cards, player_boxes, frame_shape):
    h, w = frame_shape[:2]
    community_x_range = (0.65, 1.0)  # rightmost third of the screen
    community_y_range = (0.75, 1.0)  # bottom quarter of the screen

    player_hole_cards = {pid: [] for pid in player_boxes}
    community = []

    for card in cards:
        x, y, cw, ch = card["bbox"]
        cx, cy = (x + cw / 2) / w, (y + ch / 2) / h

        assigned = False
        for pid in player_boxes:
            (px, py, pw, ph) = player_boxes[pid]["bbox"]
        # for pid, (px, py, pw, ph) in player_boxes.items():
            if px / w <= cx <= (px + pw) / w and py / h <= cy <= (py + ph) / h:
                player_hole_cards[pid].append(card)
                assigned = True
                break

        if not assigned and community_x_range[0] <= cx <= community_x_range[1] and community_y_range[0] <= cy <= community_y_range[1]:
            community.append(card)

    community.sort(key=lambda c: c["bbox"][0])
    return player_hole_cards, community


def parse_frame(frame) -> Dict:
    state = {
        "players": {},
        "cards": [],
        "pot": 0
    }

    # === Step 1: Run OCR pipeline ===
    frame_data = process_frame(frame)  # returns a dict with HUD box data
    # player_boxes = {}
    if "pot" in frame_data:
        state["pot"] = frame_data["pot"]
        del frame_data["pot"]
    state["players"] = frame_data
    

    # for pid, hud in ocr_data.get("players", {}).items():
    #     state["players"][pid] = {
    #         "name": hud.get("name"),
    #         "stack": hud.get("stack"),
    #         "pos": hud.get("pos"),
    #         "action": hud.get("action")
    #     }
    #     player_boxes[pid] = hud.get("bbox")  # bounding box of player HUD tile


    # === Step 2: Run YOLO card detection ===
    parsed_cards = detect_cards(frame)  # list of {class_id, bbox}
    # parsed_cards = []
    # for card in raw_cards:
    #     class_id = card["class_id"]
    #     if class_id in CARD_LABELS:
    #         card_info = CARD_LABELS[class_id].copy()
    #         card_info["bbox"] = card["bbox"]
    #         parsed_cards.append(card_info)

    # === Step 3: Assign cards to players or board ===
    player_cards, board_cards = assign_cards_to_sources(parsed_cards, state["players"], frame.shape)

    for pid, cards in player_cards.items():
        state["players"][pid]["cards"] = cards

    state["cards"] = board_cards
    return state
