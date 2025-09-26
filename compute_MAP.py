# compute_MAP.py
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class ComputeCOCOEval:
    """
    Official COCO bbox metric.

    Flow:
      - Receives (logits, pred_boxes) from HF Trainer (already preprocessed).
      - Uses processor.post_process_object_detection to map to xyxy at native size.
      - Converts to COCO results (xywh) and remaps 'category_id' with new2old.
      - Runs pycocotools COCOeval and returns key metrics.
    """

    def __init__(
        self,
        processor,
        ann_file: str,
        val_images: List[Dict[str, Any]],  # COCO-style dicts: 'id', 'height', 'width', ...
        new2old: Optional[Dict[int, int]] = None,
        score_thr: float = 0.001,
    ) -> None:
        self.p = processor
        self.ann_file = ann_file
        self.val_images = val_images
        self.thr = float(score_thr)
        self.new2old = {int(k): int(v) for k, v in (new2old or {}).items()}

    @staticmethod
    def _sanitize_coco_gt(coco_gt: COCO) -> None:
        """Ensure required keys exist in the COCO GT dataset."""
        ds = coco_gt.dataset
        if "info" not in ds:
            ds["info"] = {"description": "auto-added by ComputeCOCOEval"}
        if "licenses" not in ds:
            ds["licenses"] = []

    def __call__(self, eval_pred):
        (logits, pred_boxes), _ = eval_pred.predictions, eval_pred.label_ids
        results: List[Dict[str, Any]] = []

        B = logits.shape[0]
        for i in range(B):
            H = int(self.val_images[i]["height"])
            W = int(self.val_images[i]["width"])
            img_id = int(self.val_images[i]["id"])

            li = torch.as_tensor(logits[i], dtype=torch.float32)      # (Q, C+1)
            bi = torch.as_tensor(pred_boxes[i], dtype=torch.float32)  # (Q, 4) cxcywh in [0,1]

            det = self.p.post_process_object_detection(
                SimpleNamespace(logits=li.unsqueeze(0), pred_boxes=bi.unsqueeze(0)),
                target_sizes=torch.tensor([(H, W)], dtype=torch.long),
                threshold=self.thr,
            )[0]  # dict: 'boxes'(xyxy), 'scores', 'labels' (new contiguous ids)

            # xyxy -> xywh
            bxyxy = det["boxes"].cpu()
            xywh = bxyxy.clone()
            xywh[:, 2:] = bxyxy[:, 2:] - bxyxy[:, :2]

            scores = det["scores"].cpu().tolist()
            labels_new = det["labels"].cpu().tolist()

            for bb, sc, cid_new in zip(xywh.tolist(), scores, labels_new):
                cid_old = self.new2old.get(int(cid_new), int(cid_new))
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": cid_old,
                        "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                        "score": float(sc),
                    }
                )

        # COCOeval
        coco_gt = COCO(self.ann_file)
        self._sanitize_coco_gt(coco_gt)
        try:
            coco_dt = coco_gt.loadRes(results if results else [])
        except KeyError:
            self._sanitize_coco_gt(coco_gt)
            coco_dt = coco_gt.loadRes(results if results else [])

        E = COCOeval(coco_gt, coco_dt, iouType="bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()

        # Provide both "coco/*" and "eval_coco/*" aliases.
        out = {
            "coco/AP": float(E.stats[0]),
            "coco/AP50": float(E.stats[1]),
            "coco/AP75": float(E.stats[2]),
            "coco/AR100": float(E.stats[8]),
        }
        out.update(
            {
                "eval_coco/AP": out["coco/AP"],
                "eval_coco/AP50": out["coco/AP50"],
                "eval_coco/AP75": out["coco/AP75"],
                "eval_coco/AR100": out["coco/AR100"],
            }
        )
        return out
