"""Download (if needed) and standardize datasets for FALCON Tier 1+.

Usage:
    python scripts/prepare_data.py --datasets cifar10,mnist
    python scripts/prepare_data.py --datasets all

Resolves the raw-data root via falcon.data_paths.data_root():
- machine with existing torchvision data: set FALCON_DATA_ROOT to that root
  (nothing is re-downloaded);
- fresh machine (co-author): defaults to ./data and downloads automatically.

Each dataset is exported once to a standardized numpy pickle at
<root>/processed/<name>.pkl with keys x_train, y_train, x_test, y_test
(uint8 images, int64 labels). The FL pipeline consumes only these pickles,
so torch/torchvision are needed for this script only.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from falcon.data_paths import data_root, processed_path  # noqa: E402

DATASETS = ("mnist", "fmnist", "cifar10", "cifar100", "svhn")


def _load(name: str, root: Path):
    import torchvision.datasets as tvd

    if name == "mnist":
        tr, te = tvd.MNIST(root, train=True, download=True), tvd.MNIST(root, train=False, download=True)
        xtr, xte = tr.data.numpy(), te.data.numpy()
        ytr, yte = np.asarray(tr.targets), np.asarray(te.targets)
    elif name == "fmnist":
        tr, te = tvd.FashionMNIST(root, train=True, download=True), tvd.FashionMNIST(root, train=False, download=True)
        xtr, xte = tr.data.numpy(), te.data.numpy()
        ytr, yte = np.asarray(tr.targets), np.asarray(te.targets)
    elif name == "cifar10":
        tr, te = tvd.CIFAR10(root, train=True, download=True), tvd.CIFAR10(root, train=False, download=True)
        xtr, xte, ytr, yte = tr.data, te.data, np.asarray(tr.targets), np.asarray(te.targets)
    elif name == "cifar100":
        tr, te = tvd.CIFAR100(root, train=True, download=True), tvd.CIFAR100(root, train=False, download=True)
        xtr, xte, ytr, yte = tr.data, te.data, np.asarray(tr.targets), np.asarray(te.targets)
    elif name == "svhn":
        # existing machines keep .mat files under <root>/svhn (torchvision convention varies)
        svhn_root = root / "svhn" if (root / "svhn").exists() else root
        tr = tvd.SVHN(svhn_root, split="train", download=True)
        te = tvd.SVHN(svhn_root, split="test", download=True)
        xtr = np.transpose(tr.data, (0, 2, 3, 1))  # NCHW -> NHWC like CIFAR
        xte = np.transpose(te.data, (0, 2, 3, 1))
        ytr, yte = np.asarray(tr.labels), np.asarray(te.labels)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return (
        np.ascontiguousarray(xtr, dtype=np.uint8),
        np.asarray(ytr, dtype=np.int64),
        np.ascontiguousarray(xte, dtype=np.uint8),
        np.asarray(yte, dtype=np.int64),
    )


def prepare(name: str, force: bool = False) -> dict:
    out = processed_path(name)
    if out.exists() and not force:
        with out.open("rb") as f:
            d = pickle.load(f)
        return {"dataset": name, "status": "cached", "train": len(d["y_train"]), "test": len(d["y_test"])}
    xtr, ytr, xte, yte = _load(name, data_root())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(
            {"x_train": xtr, "y_train": ytr, "x_test": xte, "y_test": yte},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )
    return {"dataset": name, "status": "prepared", "train": len(ytr), "test": len(yte),
            "shape": list(xtr.shape[1:]), "classes": int(np.unique(ytr).size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all", help=f"comma list of {DATASETS} or 'all'")
    ap.add_argument("--force", action="store_true", help="re-export even if pickle exists")
    args = ap.parse_args()
    names = DATASETS if args.datasets == "all" else tuple(s.strip() for s in args.datasets.split(","))
    results = []
    for n in names:
        print(f"[prepare_data] {n} ...", flush=True)
        results.append(prepare(n, force=args.force))
        print(f"  -> {results[-1]}", flush=True)
    manifest = data_root() / "processed" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[prepare_data] manifest: {manifest}")


if __name__ == "__main__":
    main()
