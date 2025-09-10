# prepare_dataset.py
import argparse
import os
import pickle
import random
from pathlib import Path

from PIL import Image
from torchvision import datasets

def build_dataset(dataset, source_split, out_dir, num_samples, seed=0, ext="PNG"):
    out_dir = Path(out_dir)
    val_dir = out_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load raw dataset from torchvision
    if dataset.lower() == "mnist":
        ds = datasets.MNIST(root=str(out_dir / "_raw"), train=(source_split=="train"),
                            download=True)
        prefix = "MNIST_val"
        # MNIST images are 'L' (grayscale) PIL Images already
    elif dataset.lower() == "cifar10":
        ds = datasets.CIFAR10(root=str(out_dir / "_raw"), train=(source_split=="train"),
                              download=True)
        prefix = "CIFAR10_val"
        # CIFAR10 images are RGB PIL Images
    else:
        raise ValueError("dataset must be one of: mnist, cifar10")

    # 2) Subsample deterministically
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_samples]

    data_pair = []  # entries like: ["", "FILENAME.PNG", label]

    # 3) Save images and build list
    for i, idx in enumerate(indices, start=1):
        img, label = ds[idx]  # PIL Image, int label
        # Ensure PIL.Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        fname = f"{prefix}_{i:08d}.{ext.lower()}"
        img_path = val_dir / fname

        # Keep original mode; your loader will .convert('RGB') anyway
        # so MNIST 'L' -> RGB later is fine.
        img.save(img_path)

        data_pair.append(["", fname, int(label)])

    # 4) Write datapair.dat (root is ignored by your loader, but we keep it sane)
    record = {"root": "val", "data_pair": data_pair}
    with open(out_dir / "datapair.dat", "wb") as fp:
        pickle.dump(record, fp)

    print(f"Done. Wrote {len(data_pair)} images to {val_dir}")
    print(f"datapair.dat at: {out_dir / 'datapair.dat'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"])
    ap.add_argument("--out_dir", type=str, required=True, help="e.g., data/mnist")
    ap.add_argument("--num-samples", type=int, default=5000,
                    help="how many images to dump into val/")
    ap.add_argument("--source-split", type=str, default="test", choices=["train", "test"],
                    help="which split to pull samples from before writing to val/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ext", type=str, default="PNG", choices=["PNG", "JPEG"])
    args = ap.parse_args()

    build_dataset(args.dataset, args.source_split, args.out_dir,
                  args.num_samples, seed=args.seed, ext=args.ext)
