import os
import shutil

from config import OUTPUT_DIR


def organize_file(path, label, face_id=None):
    if label == "person" and face_id is not None:
        label = f"person_{face_id}"

    dst_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(dst_dir, exist_ok=True)

    filename = os.path.basename(path)
    dst_path = os.path.join(dst_dir, filename)

    if not os.path.exists(dst_path):
        shutil.copy2(path, dst_path)