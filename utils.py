from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Set, Union

import inspect
import json
import random
import numpy as np
import torch


def extract_dataset_labels(root_dataset: Union[str, Path]) -> Tuple[
    int, Dict[int, int], Dict[int, str], Dict[str, int], Path, Path, Path, Path
]:
    """
    Load COCO-style categories and build contiguous id mappings.

    Returns:
        num_labels, old2new, id2label, label2id,
        train_ann, val_ann, train_ann_path, val_ann_path
    """
    seed = 42
    root = Path(root_dataset).expanduser().resolve()
    ann_dir = root / "annotations"

    train_ann = find_ann(ann_dir, "train")
    val_ann = find_ann(ann_dir, "val")
    train_ann_path, val_ann_path = train_ann, val_ann  # kept for backward-compat

    # Reproducibility (safe on CPU-only as well)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cats = json.loads(train_ann.read_text())["categories"]
    old2new: Dict[int, int] = {int(c["id"]): i for i, c in enumerate(cats)}
    id2label: Dict[int, str] = {i: c["name"] for i, c in enumerate(cats)}
    label2id: Dict[str, int] = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    return (
        num_labels,
        old2new,
        id2label,
        label2id,
        train_ann,
        val_ann,
        train_ann_path,
        val_ann_path,
    )


def sig_params(obj: Any) -> Set[str]:
    """Return parameter names of a callable's signature (empty set if unavailable)."""
    try:
        return set(inspect.signature(obj).parameters.keys())
    except (ValueError, TypeError):
        return set()


def find_ann(ann_dir: Union[str, Path], split: str) -> Path:
    """
    Find an annotation file for a given split.
    Priority:
      1) {ann_dir}/{split}.json
      2) instances_*{split}*.json
      3) *{split}*.json
    """
    ann_dir = Path(ann_dir)
    cand = ann_dir / f"{split}.json"
    if cand.exists():
        return cand

    # Broader patterns
    for p in list(ann_dir.glob(f"instances_*{split}*.json")) + list(ann_dir.glob(f"*{split}*.json")):
        if p.exists():
            return p

    raise FileNotFoundError(f"Could not find annotations for split='{split}' in {ann_dir}")
