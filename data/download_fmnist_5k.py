import os
import numpy as np
import torchvision

SELECTED_CLASSES = list(range(5))
SAMPLES_PER_CLASS_TRAIN = 1000
SAMPLES_PER_CLASS_TEST = 1000

DATA_DIR = os.path.join(os.path.dirname(__file__), "fmnist_5k")


def save_split(dataset, out_dir: str, samples_per_class: int):
    os.makedirs(out_dir, exist_ok=True)

    class_indices = {c: [] for c in SELECTED_CLASSES}
    for idx in range(len(dataset)):
        label = int(dataset.targets[idx])
        if label in class_indices:
            class_indices[label].append(idx)

    rng = np.random.default_rng(0)
    selected = []
    for c in SELECTED_CLASSES:
        idxs = np.array(class_indices[c])
        n = min(samples_per_class, len(idxs))
        chosen = rng.choice(idxs, size=n, replace=False)
        selected.extend(chosen.tolist())

    imgs, labels = [], []
    for idx in selected:
        img, label = dataset[idx]
        imgs.append(np.array(img))  # PIL -> (28, 28) uint8
        labels.append(label)

    imgs = np.stack(imgs, axis=0)          # (N, 28, 28) uint8
    labels = np.array(labels, dtype=np.int64)
    npz_path = os.path.join(out_dir, "data.npz")
    np.savez_compressed(npz_path, images=imgs, labels=labels)
    print(f"Saved {imgs.shape[0]} examples to {npz_path}")


if __name__ == "__main__":
    raw_dir = os.path.join(DATA_DIR, "raw")
    train_ds = torchvision.datasets.FashionMNIST(root=raw_dir, train=True, download=True)
    test_ds = torchvision.datasets.FashionMNIST(root=raw_dir, train=False, download=True)
    save_split(train_ds, os.path.join(DATA_DIR, "train"), SAMPLES_PER_CLASS_TRAIN)
    save_split(test_ds, os.path.join(DATA_DIR, "test"), SAMPLES_PER_CLASS_TEST)
