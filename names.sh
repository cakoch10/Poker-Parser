python - <<'PY'
import pathlib, textwrap, json
CARD_DIR = pathlib.Path("synthetic_cards/card_crops")  # adjust if needed
names = sorted([d.name for d in CARD_DIR.iterdir() if d.is_dir()])
print("names:")
for idx, n in enumerate(names):
    print(f"  {idx}: {n}")
PY

