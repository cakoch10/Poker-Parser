
import random, shutil, glob, os, pathlib

ROOT = pathlib.Path("cards_dataset")
IMG_SRC = ROOT / "images" / "train"
LBL_SRC = ROOT / "labels" / "train"

TRAIN_IMG = ROOT / "images" / "train"; TRAIN_LBL = ROOT / "labels" / "train"
VAL_IMG   = ROOT / "images" / "val";   VAL_LBL   = ROOT / "labels" / "val"
for p in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]: p.mkdir(parents=True, exist_ok=True)

images = glob.glob(str(IMG_SRC/"*.jpg"))
random.seed(42)
random.shuffle(images)

val_ratio = 0.2
split_idx = int(len(images) * (1 - val_ratio))
train_imgs, val_imgs = images[:split_idx], images[split_idx:]

def move_pair(img_list, dst_img_dir, dst_lbl_dir):
    for img_path in img_list:
        img_path = pathlib.Path(img_path)
        lbl_path = LBL_SRC / (img_path.stem + ".txt")
        shutil.move(img_path, dst_img_dir / img_path.name)
        shutil.move(lbl_path, dst_lbl_dir / lbl_path.name)

move_pair(train_imgs, TRAIN_IMG, TRAIN_LBL)
move_pair(val_imgs,   VAL_IMG,   VAL_LBL)
print(f"Done. Train: {len(train_imgs)}, Val: {len(val_imgs)}")
