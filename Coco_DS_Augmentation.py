from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Union, Optional

import albumentations as A
import numpy as np
from PIL import Image

from Coco_DS import CocoDS


class CocoDSAug(CocoDS):
    """
    COCO-style dataset with on-the-fly Albumentations.
    Keeps CocoDS output schema and ensures keys required by RTDetrImageProcessor
    (area, iscrowd) are present.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        ann_path: Union[str, Path],
        old2new: Dict[int, int],
        augmentations: List[Any],
        min_visibility: float = 0.2,
    ) -> None:
        super().__init__(root, split, ann_path, old2new)
        self.tf = A.Compose(
            augmentations,
            bbox_params=A.BboxParams(
                format="coco",            # [x, y, w, h] in pixels
                label_fields=["category_ids"],
                min_visibility=min_visibility,
            ),
        )

    def __getitem__(self, i: int) -> Dict[str, Any]:
        info = self.images[i]

        # Resolve path like CocoDS
        p1 = self.root / "images" / self.split / info["file_name"]
        p2 = self.root / "images" / info["file_name"]
        img_path = p1 if p1.exists() else p2

        img = Image.open(img_path).convert("RGB")
        anns = self.ann_by_img.get(int(info["id"]), [])

        # Extract bboxes/labels (already contiguous via CocoDS)
        bboxes = [a["bbox"] for a in anns]
        labels = [a["category_id"] for a in anns]

        if bboxes:
            out = self.tf(image=np.asarray(img), bboxes=bboxes, category_ids=labels)
            aug_img = out["image"]
            aug_bboxes = out["bboxes"]
            aug_labels = out["category_ids"]

            # Rebuild annotations ensuring area/iscrowd and filter degenerate boxes
            anns_aug: List[Dict[str, Any]] = []
            for b, c in zip(aug_bboxes, aug_labels):
                x, y, w, h = map(float, b)
                if w > 0.0 and h > 0.0:
                    anns_aug.append(
                        {
                            "bbox": [x, y, w, h],
                            "category_id": int(c),
                            "area": float(w * h),
                            "iscrowd": 0,
                        }
                    )

            if anns_aug:
                img = Image.fromarray(aug_img)
                anns = anns_aug
            else:
                # If all boxes were dropped by augmentation, fallback to original (valid only)
                anns = [
                    {
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "category_id": int(c),
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                    for (x, y, w, h), c in zip(bboxes, labels)
                    if float(w) > 0.0 and float(h) > 0.0
                ]
        else:
            # No annotations: return as-is (RTDetrImageProcessor supports empty list)
            anns = []

        return {
            "image": img,
            "image_id": int(info["id"]),
            "width": int(info.get("width", img.width)),
            "height": int(info.get("height", img.height)),
            "annotations": anns,
        }
