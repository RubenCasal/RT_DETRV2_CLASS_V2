from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from transformers import TrainerCallback
from PIL import Image, ImageDraw, ImageFont
import torch
import random


class SaveValImgsInCkpt(TrainerCallback):
    """
    On each checkpoint, dumps K validation images with:
      - Ground truth (green):  val{j}_label.jpg
      - Predictions (red):     val{j}_pred.jpg
    Output dir: {output_dir}/checkpoint-{global_step}/val_vis/

    Optionally logs images to Weights & Biases.
    """

    def __init__(
        self,
        processor,
        val_ds,
        id2label: Dict[int, str],
        k: int = 4,
        thr: float = 0.01,
        infer_size: Optional[Tuple[int, int]] = None,  # (H, W). If None, use native size
        log_to_wandb: bool = False,
        wandb_prefix: str = "val_vis",
        seed: int = 123,
    ) -> None:
        self.p = processor
        self.ds = val_ds
        self.id2label = {int(k_): v for k_, v in id2label.items()}
        self.k = int(k)
        self.thr = float(thr)
        self.infer_size = infer_size
        self.log_to_wandb = log_to_wandb
        self.wandb_prefix = wandb_prefix.rstrip("/")
        self.rng = random.Random(seed)

        # Font fallback
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        except Exception:
            self.font = ImageFont.load_default()

    # ---------- drawing helpers ----------

    def _draw_gt(self, img: Image.Image, anns: List[Dict[str, Any]]) -> Image.Image:
        im = img.copy().convert("RGB")
        draw = ImageDraw.Draw(im)
        for a in anns or []:
            x, y, w, h = a.get("bbox", (0, 0, 0, 0))
            cls = int(a.get("category_id", -1))
            name = self.id2label.get(cls, str(cls))
            x1, y1, x2, y2 = x, y, x + w, y + h
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            draw.text((x1, max(0, y1 - 12)), name, fill=(0, 255, 0), font=self.font)
        return im

    def _draw_pred(self, img: Image.Image, det: Dict[str, Any]) -> Image.Image:
        im = img.copy().convert("RGB")
        draw = ImageDraw.Draw(im)
        boxes = det.get("boxes", [])
        scores = det.get("scores", [])
        labels = det.get("labels", [])
        for b, sc, lb in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in b]
            name = self.id2label.get(int(lb), str(int(lb)))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1, max(0, y1 - 12)), f"{name}:{float(sc):.2f}", fill=(255, 0, 0), font=self.font)
        return im

    # ---------- HF callback ----------

    def on_save(self, args, state, control, **kwargs):
        out_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "val_vis"
        out_dir.mkdir(parents=True, exist_ok=True)

        model = kwargs["model"].eval()
        device = next(model.parameters()).device

        # Sample K validation indices (without replacement)
        num_items = len(self.ds)
        if num_items == 0:
            return control
        k = min(self.k, num_items)
        idxs = self.rng.sample(range(num_items), k=k)

        # Lazy import W&B
        wb = None
        if self.log_to_wandb:
            try:
                import wandb as _wandb
                wb = _wandb
            except Exception:
                wb = None  # no logging if W&B is missing

        with torch.no_grad():
            images_to_log = []
            for j, idx in enumerate(idxs, 1):
                sample = self.ds[idx]
                img: Image.Image = sample["image"].convert("RGB")
                H, W = img.height, img.width

                # --- GT ---
                im_gt = self._draw_gt(img, sample.get("annotations", []))
                gt_path = out_dir / f"val{j}_label.jpg"
                im_gt.save(gt_path, quality=95)

                # --- Pred ---
                proc_kwargs = {"images": img, "return_tensors": "pt", "do_resize": False}
                if self.infer_size is not None:
                    ih, iw = int(self.infer_size[0]), int(self.infer_size[1])
                    proc_kwargs.update({"do_resize": True, "size": {"height": ih, "width": iw}})

                inputs = self.p(**proc_kwargs).to(device)
                outputs = model(**inputs)
                det = self.p.post_process_object_detection(
                    outputs,
                    target_sizes=torch.tensor([(H, W)], device=device),
                    threshold=self.thr,
                )[0]

                im_pd = self._draw_pred(img, det)
                pd_path = out_dir / f"val{j}_pred.jpg"
                im_pd.save(pd_path, quality=95)

                if wb is not None:
                    images_to_log.append(wb.Image(im_gt, caption=f"GT idx={idx}"))
                    images_to_log.append(wb.Image(im_pd, caption=f"Pred idx={idx} thr={self.thr}"))

            if wb is not None and images_to_log:
                wb.log({self.wandb_prefix: images_to_log, "global_step": state.global_step})

        return control
