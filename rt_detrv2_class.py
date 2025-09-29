# ==============================
# rt_detrv2_class.py
# ==============================

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any, Optional
import re
import math
import numpy as np
from PIL import Image
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    RTDetrV2Config, RTDetrV2ForObjectDetection,
    AutoImageProcessor, TrainingArguments, Trainer, TrainerCallback
)
from Coco_DS import CocoDS
from Coco_DS_Augmentation import CocoDSAug
from utils import sig_params


class SaveProcessorCallback(TrainerCallback):
    """Save the processor at each checkpoint and at the end."""
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        try:
            self.processor.save_pretrained(ckpt_dir)
            print(f"[Processor] saved at {ckpt_dir}")
        except Exception as e:
            print(f"[Processor] save error at checkpoint: {e}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        out_dir = Path(args.output_dir)
        try:
            self.processor.save_pretrained(out_dir)
            print(f"[Processor] saved at {out_dir} (end of training)")
        except Exception as e:
            print(f"[Processor] save error at end: {e}")
        return control


class _ParamGroupTrainer(Trainer):
    """
    Two param groups: backbone vs head (head with no weight decay).
    Hybrid LR schedule: cosine (backbone) + constant with warmup (head).
    """
    DEFAULT_HEAD_PATTERNS = [
        r"\bclass_embed\b", r"\bbox_embed\b",
        r"\bcls_head\b", r"\breg_head\b",
        r"\bpred[^.]*cls\b", r"\bpred[^.]*bbox\b",
        r"\bclassification_head\b", r"\bbox_head\b",
        r"\bhead\.", r"\bheads?\.", r"\bdecoder\..*(cls|bbox)"
    ]

    def _is_head_param(self, name: str) -> bool:
        return any(re.search(pat, name) for pat in self.DEFAULT_HEAD_PATTERNS)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        wd_backbone = getattr(self.args, "weight_decay", 0.005)
        wd_head = 0.0
        backbone_lr = getattr(self.args, "backbone_lr", 5e-5)
        head_lr = getattr(self.args, "head_lr", 1e-3)

        no_decay_keys = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        backbone_decay, backbone_nodecay, head_decay, head_nodecay = [], [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_nodecay = any(nd in name for nd in no_decay_keys)
            is_head = self._is_head_param(name)
            if is_head:
                (head_nodecay if is_nodecay else head_decay).append(param)
            else:
                (backbone_nodecay if is_nodecay else backbone_decay).append(param)

        param_groups = []
        if backbone_decay:
            param_groups.append({"params": backbone_decay, "lr": backbone_lr, "weight_decay": wd_backbone})
        if backbone_nodecay:
            param_groups.append({"params": backbone_nodecay, "lr": backbone_lr, "weight_decay": 0.0})
        if head_decay:
            param_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": wd_head})
        if head_nodecay:
            param_groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0})

        self.optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[Optimizer] = None):
        if optimizer is not None:
            self.optimizer = optimizer
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        if self.optimizer is None:
            self.create_optimizer()

        head_lr = getattr(self.args, "head_lr", 1e-3)
        warmup_steps = int(getattr(self.args, "warmup_ratio", 0.1) * num_training_steps)

        def cosine_decay(step, total):
            progress = step / max(1, total)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        lambdas = []
        for g in self.optimizer.param_groups:
            is_head = abs(g["lr"] - head_lr) < 1e-12
            if is_head:
                def fn(step, _=None):
                    if step < warmup_steps:
                        return float(step) / max(1, warmup_steps)
                    return 1.0
            else:
                def fn(step, _=None):
                    if step < warmup_steps:
                        return float(step) / max(1, warmup_steps)
                    return cosine_decay(step - warmup_steps, num_training_steps - warmup_steps)
            lambdas.append(fn)

        self.lr_scheduler = LambdaLR(self.optimizer, lambdas)
        return self.lr_scheduler


class RtDetvr2Trainer:
    """Wrapper to build datasets, processor, model, and a custom Trainer."""

    def __init__(
        self,
        num_labels: int,
        old2new: Dict[int, int],
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        train_ann: Dict[str, Any],
        val_ann: Dict[str, Any],
        model_config: Optional[RTDetrV2Config],
        training_config: Dict[str, Any],
        augmentations: Any,
        root: str | Path,
        image_size: Tuple[int, int] = (256, 256),
        from_checkpoint: Optional[str] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.10,
        # non-standard config hypers (optional)
        eos_coef: Optional[float] = None,
        matcher_class_cost: Optional[float] = None,
        matcher_bbox_cost: Optional[float] = None,
        class_loss_coef: Optional[float] = None,
    ):
        self.root = root
        self.num_labels = num_labels
        self.old2new = old2new
        self.label2id = label2id
        self.id2label = id2label
        self.train_ann = train_ann
        self.val_ann = val_ann
        self.cfg = model_config
        self.augmentations = augmentations
        self._use_focal = use_focal_loss
        self._focal_gamma = focal_gamma
        self._focal_alpha = focal_alpha
        self._label_smoothing = label_smoothing

        # expose optional config hypers
        self._eos_coef = eos_coef
        self._matcher_class_cost = matcher_class_cost
        self._matcher_bbox_cost = matcher_bbox_cost
        self._class_loss_coef = class_loss_coef

        # Processor
        Ht, Wt = image_size
        ckpt_for_processor = from_checkpoint or "PekingU/rtdetr_v2_r50vd"
        self.processor = AutoImageProcessor.from_pretrained(
            ckpt_for_processor,
            do_resize=True,
            size={"max_height": int(Ht), "max_width": int(Wt)},  
            do_pad=True,
            pad_size={"height": int(Ht), "width": int(Wt)},      
        )

        # Model
        if from_checkpoint is not None:
            self.model = RTDetrV2ForObjectDetection.from_pretrained(
                from_checkpoint,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            assert self.cfg is not None, "model_config must be provided if from_checkpoint is None"
            self.model = RTDetrV2ForObjectDetection(self.cfg)

        if hasattr(self.model, "config"):
            self.model.config.return_dict = True

        # Targeted config adjustments (only if provided)
        cfg = self.model.config

        def set_if(attr_names: List[str], value: float | bool | str):
            for k in attr_names:
                if hasattr(cfg, k):
                    try:
                        setattr(cfg, k, value)
                    except Exception:
                        pass

        if self._eos_coef is not None:
            set_if(["eos_coefficient", "eos_coef", "no_object_weight", "background_weight"], float(self._eos_coef))

        if self._matcher_class_cost is not None:
            set_if(["matcher_class_cost", "set_cost_class", "matcher_cost_class", "class_cost"], float(self._matcher_class_cost))

        if self._matcher_bbox_cost is not None:
            set_if(["matcher_bbox_cost", "set_cost_bbox", "matcher_cost_bbox", "bbox_cost"], float(self._matcher_bbox_cost))

        if self._class_loss_coef is not None:
            set_if(["class_loss_coef", "loss_cls_weight", "weight_loss_cls", "cls_loss_weight"], float(self._class_loss_coef))

        # Focal loss and label smoothing via config (if supported)
        if self._use_focal:
            set_if(["use_focal_loss", "focal_loss", "cls_use_focal"], True)
            set_if(["focal_gamma", "gamma", "cls_gamma"], float(self._focal_gamma))
            set_if(["focal_alpha", "alpha", "cls_alpha"], float(self._focal_alpha))
            set_if(["classification_loss_type", "class_loss_type", "loss_cls_type"], "focal")

        if self._label_smoothing and self._label_smoothing > 0:
            set_if(["label_smoothing", "label_smoothing_factor", "cls_label_smoothing"], float(self._label_smoothing))

        # Datasets
        self.train_ds = CocoDSAug(self.root, "train", self.train_ann, self.old2new, augmentations=self.augmentations)
        self.val_ds = CocoDS(self.root, "val", self.val_ann, self.old2new)

        # Collate
        def _collate(batch: List[Dict[str, Any]]):
            images = [b["image"] for b in batch]
            anns = [{"image_id": b["image_id"], "annotations": b["annotations"]} for b in batch]
            bf = self.processor(images=images, annotations=anns, return_tensors="pt")
            out = {"pixel_values": bf["pixel_values"], "labels": bf["labels"]}
            if "pixel_mask" in bf:
                out["pixel_mask"] = bf["pixel_mask"]
            return out

        self.collate_fn = _collate

        # TrainingArguments
        ta_params = sig_params(TrainingArguments.__init__)
        ta_kwargs = dict(training_config)

        # Extract custom LRs
        self._backbone_lr = ta_kwargs.pop("backbone_lr", 5e-5)
        self._head_lr = ta_kwargs.pop("head_lr", 1e-3)

        # Ensure step strategies exist when supported by current HF version
        if "evaluation_strategy" in ta_params and "evaluation_strategy" not in ta_kwargs:
            ta_kwargs["evaluation_strategy"] = "steps"
        elif "eval_strategy" in ta_params and "eval_strategy" not in ta_kwargs:
            ta_kwargs["eval_strategy"] = "steps"
        if "save_strategy" in ta_params and "save_strategy" not in ta_kwargs:
            ta_kwargs["save_strategy"] = "steps"
        if "label_names" in ta_params:
            ta_kwargs["label_names"] = ["labels"]

        # Label smoothing is handled via model.config
        ta_kwargs["label_smoothing_factor"] = 0.0

        self.args = TrainingArguments(**ta_kwargs)
        setattr(self.args, "backbone_lr", self._backbone_lr)
        setattr(self.args, "head_lr", self._head_lr)
        setattr(self.args, "greater_is_better", True)

        # Make best-model selection explicit
        mf = getattr(self.args, "metric_for_best_model", None)
        if mf in (None, "map", "eval_map"):
            setattr(self.args, "metric_for_best_model", "eval_coco/AP")
            if not getattr(self.args, "load_best_model_at_end", False):
                setattr(self.args, "load_best_model_at_end", True)

        # Trainer
        tr_params = sig_params(Trainer.__init__)
        tr_kwargs = dict(
            model=self.model,
            args=self.args,
            data_collator=self.collate_fn,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
        )
        if "preprocess_logits_for_metrics" in tr_params:
            tr_kwargs["preprocess_logits_for_metrics"] = self.preprocess_logits_for_metrics
        if "compute_metrics" in tr_params:
            tr_kwargs["compute_metrics"] = self._noop_metrics

        self.trainer = _ParamGroupTrainer(**tr_kwargs)
        self.trainer.add_callback(SaveProcessorCallback(self.processor))

    # ---- metrics plumbing ----
    def _extract_logits_boxes(self, obj: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        def to_t(x):
            return torch.from_numpy(x) if isinstance(x, np.ndarray) else x

        if hasattr(obj, "logits") and hasattr(obj, "pred_boxes"):
            return obj.logits.detach().cpu(), obj.pred_boxes.detach().cpu()

        if isinstance(obj, dict):
            if "logits" in obj and "pred_boxes" in obj:
                return to_t(obj["logits"]).detach().cpu(), to_t(obj["pred_boxes"]).detach().cpu()
            for v in obj.values():
                try:
                    return self._extract_logits_boxes(v)
                except Exception:
                    pass

        if isinstance(obj, (list, tuple)):
            arrays: List[torch.Tensor] = []
            for x in obj:
                try:
                    return self._extract_logits_boxes(x)
                except Exception:
                    pass
                x = to_t(x)
                if torch.is_tensor(x) and x.ndim >= 3:
                    arrays.append(x)

            boxes_cands = [a for a in arrays if a.size(-1) == 4]
            if boxes_cands:
                boxes = max(boxes_cands, key=lambda a: (a.shape[1], a.shape[-1]))
                B_, Q_ = boxes.shape[0], boxes.shape[1]
                logits_cands = [a for a in arrays if a.shape[:2] == (B_, Q_) and a.size(-1) != 4]
                logits_pref = [a for a in logits_cands if a.size(-1) in (self.num_labels, self.num_labels + 1)]
                logits = (logits_pref[0] if logits_pref else (logits_cands[0] if logits_cands else None))
                if logits is not None:
                    return logits.detach().cpu(), boxes.detach().cpu()

        raise RuntimeError("Could not extract (logits, pred_boxes) from model output.")

    def preprocess_logits_for_metrics(self, outputs, labels):
        logits, boxes = self._extract_logits_boxes(outputs)
        return (logits, boxes)

    @staticmethod
    def _noop_metrics(eval_pred) -> Dict[str, float]:
        return {}

    @staticmethod
    def _monitor_confidence(eval_pred) -> Dict[str, float]:
        import torch as _torch, numpy as _np
        logits, _ = eval_pred.predictions
        if isinstance(logits, _np.ndarray):
            logits = _torch.from_numpy(logits)
        probs = _torch.softmax(logits, dim=-1)
        if probs.size(-1) > 1:
            probs = probs[..., :-1]
        max_p = probs.max(dim=-1).values
        return {
            "eval_confidence/mean_max": float(max_p.mean().item()),
            "eval_confidence/p75_max": float(_torch.quantile(max_p.flatten(), 0.75).item()),
        }

    def attach_map_evaluator(self, compute_metrics_fn: Callable):
        """Combine COCO mAP with confidence diagnostics."""
        def _combined(eval_pred):
            d1 = compute_metrics_fn(eval_pred)
            d2 = self._monitor_confidence(eval_pred)
            return {**d1, **d2}
        self.trainer.compute_metrics = _combined

    # ---- public API ----
    def train(self):
        return self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()

    @torch.no_grad()
    def predict_image(self, image_path: str | Path, threshold: float = 0.5, device: str = "cpu"):
        self.model.to(device).eval()
        img = Image.open(image_path).convert("RGB")
        H, W = img.height, img.width
        inputs = self.processor(images=img, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        det = self.processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(H, W)], device=device), threshold=threshold
        )[0]
        det["labels"] = [self.id2label[int(i)] for i in det["labels"]]
        return det

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(path)
        try:
            self.processor.save_pretrained(path)
            print(f"[Processor] saved at {path} (save())")
        except Exception as e:
            print(f"[Processor] save error in save(): {e}")
