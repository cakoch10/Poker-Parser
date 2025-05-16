import os
import shutil

OUTPUT_DIR = "extracted_cards_keep"

for folder in os.listdir(OUTPUT_DIR):
    folder_path = os.path.join(OUTPUT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    if len(jpg_files) <= 3:
        print(f"Removing folder with {len(jpg_files)} image(s): {folder_path}")
        shutil.rmtree(folder_path, ignore_errors=True)
    if len(jpg_files) >= 10:
        for file_to_delete in jpg_files[10:]:
            os.remove(os.path.join(folder_path, file_to_delete))
        print(f"Trimmed {len(jpg_files) - 10} image(s) from: {folder_path}")


print("Cleanup complete.")
