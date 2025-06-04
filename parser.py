import cv2
import json
from typing import List, Dict, Optional
from frame_parser import parse_frame


def is_new_hand(prev: Optional[Dict], curr: Dict) -> bool:
    if prev is None:
        return True

    pot_reset = prev.get("pot", 0) > 0 and curr.get("pot", 0) == 0
    board_reset = len(prev.get("cards", [])) > 0 and len(curr.get("cards", [])) == 0

    stack_changes = any(
        prev["players"].get(pid, {}).get("stack") != curr["players"].get(pid, {}).get("stack")
        for pid in curr["players"]
    )

    actions_cleared = all(
        not curr["players"].get(pid, {}).get("action") for pid in curr["players"]
    )

    if pot_reset and board_reset:
        return True
    if pot_reset and stack_changes and actions_cleared:
        return True

    return False


def should_emit_new_state(curr: Dict, prev: Optional[Dict]) -> bool:
    if prev is None:
        return True

    if curr.get("pot") != prev.get("pot"):
        return True
    if curr.get("cards") != prev.get("cards"):
        return True

    for pid in curr["players"]:
        if pid not in prev["players"]:
            return True
        if curr["players"][pid].get("action") != prev["players"][pid].get("action"):
            return True

    return False


def normalize_action(action_str: str) -> str:
    action_str = action_str.upper().strip()
    if action_str.startswith("RAISE"):
        amt = action_str.replace("RAISE", "").strip()
        return f"raises {amt}" if amt else "raises"
    elif action_str.startswith("CALL"):
        amt = action_str.replace("CALL", "").strip()
        return f"calls {amt}" if amt else "calls"
    elif action_str.startswith("BET"):
        amt = action_str.replace("BET", "").strip()
        return f"bets {amt}" if amt else "bets"
    elif "CHECK" in action_str:
        return "checks"
    elif "FOLD" in action_str:
        return "folds"
    return action_str.lower()


def hand_to_phh(hand_frames: List[Dict]) -> str:
    players = {}  # pid -> {name, stack, position}
    actions = []  # [(street, name, action_str)]
    board = []
    seen_actions = set()

    for state in hand_frames:
        for pid, info in state["players"].items():
            players.setdefault(pid, {
                "name": info.get("name", f"Player{pid}"),
                "stack": info.get("stack", 0),
                "position": info.get("pos", "")
            })

        for pid, info in state["players"].items():
            action = info.get("action")
            key = (pid, action)
            if action and key not in seen_actions:
                actions.append(("preflop", players[pid]["name"], normalize_action(action)))
                seen_actions.add(key)

        cards = state.get("cards", [])
        if len(cards) > len(board):
            board = cards

    lines = []
    lines.append("Table: 1")
    for p in players.values():
        lines.append(f"Seat: {p['position']} {p['name']} ({p['stack']})")

    sb = next((p for p in players.values() if p['position'] == "SB"), None)
    bb = next((p for p in players.values() if p['position'] == "BB"), None)
    if sb and bb:
        lines.append(f"Blinds: {sb['name']} posts small blind, {bb['name']} posts big blind")

    for street, name, act in actions:
        lines.append(f"{name}: {act}")

    if board:
        board_str = " ".join([f"{c['rank']}{c['suit'][0].upper()}" for c in board])
        lines.append(f"Board: {board_str}")

    lines.append("Summary:")
    return "\n".join(lines)


def stream_parse_video(video_path: str, output_path: str, frame_skip: int = 10):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    last_state = None
    current_hand = []
    all_hands = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        state = parse_frame(frame)
        if state["players"]:
            print(json.dumps(state, indent=2))
        state["frame"] = frame_idx

        if is_new_hand(last_state, state):
            if current_hand:
                all_hands.append(current_hand)
            current_hand = []

        if should_emit_new_state(state, last_state):
            current_hand.append(state)

        last_state = state

    if current_hand:
        all_hands.append(current_hand)

    cap.release()

    with open(output_path, "w") as f:
        for hand in all_hands:
            f.write(hand_to_phh(hand))
            f.write("\n\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to poker broadcast video")
    parser.add_argument("output", help="Path to save parsed hand history .phh file")
    parser.add_argument("--frame-skip", type=int, default=10, help="Number of frames to skip")
    args = parser.parse_args()

    stream_parse_video(args.video, args.output, frame_skip=args.frame_skip)
