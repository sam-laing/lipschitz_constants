import os
from typing import Dict
import numpy as np
from datasets import load_dataset
from PIL import Image
import io

DATA_DIR = "/fast/slaing/data/vision/cifar10_5class_5k"
REPO_ID = "r-three/cifar10-5class-5k"


def item_to_array(item) -> np.ndarray:
    img = item.get("img") if isinstance(item, dict) else item
    if img is None:
        raise RuntimeError("No image found in item")
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    # img might be bytes or a dict with bytes
    if isinstance(img, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(bytes(img))).convert("RGB"))
    if isinstance(img, dict):
        # common HF packaging: {"bytes": b"..."} or {"array": ...}
        for k in ("bytes", "buffer", "data", "array"):
            if k in img and img[k] is not None:
                return item_to_array(img[k])
    # fallback: try to convert via numpy
    return np.asarray(img)


def save_split(split_name: str, out_dir: str, max_items: int = None):
    os.makedirs(out_dir, exist_ok=True)
    imgs = []
    labels = []
    ds = load_dataset(REPO_ID, split=split_name, streaming=True)
    for i, item in enumerate(ds):
        if max_items and i >= max_items:
            break
        imgs.append(item_to_array(item))
        lbl = item.get("label") if isinstance(item, dict) else None
        labels.append(int(lbl) if lbl is not None else -1)
    imgs = np.stack(imgs, axis=0)
    labels = np.array(labels, dtype=np.int64)
    npz_path = os.path.join(out_dir, "data.npz")
    np.savez_compressed(npz_path, images=imgs, labels=labels)
    print(f"Saved {imgs.shape[0]} examples to {npz_path}")


if __name__ == "__main__":
    # rebuild both splits; adjust max_items if you want a subset
    save_split("train", os.path.join(DATA_DIR, "train"))
    save_split("test", os.path.join(DATA_DIR, "test"))