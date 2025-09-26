from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Union

import json
from PIL import Image
from torch.utils.data import Dataset


class CocoDS(Dataset):
    """
    Minimal COCO-style dataset:
      - Expects {root}/images/{split}/file_name or {root}/images/file_name.
      - Remaps original category ids via old2new (contiguous ids).
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        ann_path: Union[str, Path],
        old2new: Dict[int, int],
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.ann_path = Path(ann_path)
        self.old2new = {int(k): int(v) for k, v in old2new.items()}

        js = json.loads(self.ann_path.read_text())
        self.images: List[Dict[str, Any]] = js.get("images", [])
        self.ann_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for a in js.get("annotations", []):
            cid = int(a["category_id"])
            img_id = int(a["image_id"])
            bbox = a["bbox"]
            self.ann_by_img[img_id].append(
                {
                    "bbox": bbox,
                    "category_id": self.old2new.get(cid, cid),
                    "iscrowd": a.get("iscrowd", 0),
                    "area": a.get("area", bbox[2] * bbox[3]),
                }
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        info = self.images[i]
        file_name = info["file_name"]

        p1 = self.root / "images" / self.split / file_name
        p2 = self.root / "images" / file_name
        img_path = p1 if p1.exists() else p2

        img = Image.open(img_path).convert("RGB")
        img_id = int(info["id"])

        return {
            "image": img,
            "image_id": img_id,
            "width": int(info.get("width", img.width)),
            "height": int(info.get("height", img.height)),
            "annotations": self.ann_by_img.get(img_id, []),
        }
