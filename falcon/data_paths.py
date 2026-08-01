"""Dataset location resolution.

Machine-specific data lives OUTSIDE the repo. Resolution order:

1. ``FALCON_DATA_ROOT`` environment variable (e.g. ``D:\\pythondata\\torch data``);
2. ``./data`` relative to the current working directory (auto-created; datasets
   are downloaded there by ``scripts/prepare_data.py``).

Both layouts are standard torchvision roots, so the same code runs on any
machine without editing configs.
"""
import os
from pathlib import Path


def data_root() -> Path:
    env = os.environ.get("FALCON_DATA_ROOT")
    root = Path(env) if env else Path("data")
    root.mkdir(parents=True, exist_ok=True)
    return root


def processed_path(dataset: str) -> Path:
    """Location of the standardized numpy pickle produced by prepare_data.py."""
    return data_root() / "processed" / f"{dataset}.pkl"
