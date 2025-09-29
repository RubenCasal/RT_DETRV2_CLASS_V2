from pathlib import Path
from math import ceil
import os
import json

from transformers.integrations import WandbCallback
from rt_detrv2_class import RtDetvr2Trainer
from compute_MAP import ComputeCOCOEval
from save_imgs_val import SaveValImgsInCkpt
import wandb
from utils import extract_dataset_labels
import albumentations as A

USE_WANDB = True
WANDB_PROJECT = "Pt-Fue"
WANDB_RUN_NAME = "rtdetrv2-coco-pretrained2"
CKPT = "PekingU/rtdetr_v2_r50vd"

if USE_WANDB:
    
    WANDB_API_KEY = "78d292744788af62441ee14891bd488ef500e3b6"
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

# Dataset / output
ROOT = "/home/rcasal/Desktop/projects/PtFue/detvr2_training/dataset_prueba2"
OUTPUT_DIR = "rtdetrv2_pretrained_out"

# Training hyperparameters
TRAIN_BATCH = 2
VAL_BATCH = 2
GRAD_ACCUM = 2
EPOCHS = 60
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.005
WARMUP_RATIO = 0.1
USE_FP16 = False
USE_BF16 = True
IMAGE_SIZE = (255, 255)
EVAL_RATE_EPOCHS = 0.5

# Loss settings set via model.config inside the trainer
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 1.0
FOCAL_ALPHA = 0.25
LABEL_SMOOTHING_IN_MODEL = 0.02

# --- Non-standard config hypers (applied to model.config) ---
EOS_COEF = 1e-5             # lower "no-object/background" weight to avoid over-suppressing logits
MATCHER_CLASS_COST = 2.0    # Hungarian matcher: class term weight
MATCHER_BBOX_COST = 4.0     # Hungarian matcher: bbox term weight
CLASS_LOSS_COEF = 2.0       # classification loss weight in the total loss

# Labels / annotations
num_labels, old2new, id2label, label2id, train_ann, val_ann, train_ann_path, val_ann_path = extract_dataset_labels(ROOT)

with open(Path(train_ann_path), "r") as f:
    num_train_samples = len(json.load(f)["images"])

# Steps and evaluation/save frequency
world_size = int(os.environ.get("WORLD_SIZE", "1"))
effective_bs = TRAIN_BATCH * GRAD_ACCUM * world_size
steps_per_epoch = max(1, ceil(num_train_samples / max(1, effective_bs)))
eval_save_steps = max(1, int(steps_per_epoch * EVAL_RATE_EPOCHS))

# TrainingArguments (dict) — scheduler is overridden by custom trainer
training_arguments = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=VAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=WARMUP_RATIO,
    fp16=USE_FP16,
    bf16=USE_BF16,
    remove_unused_columns=False,
    report_to=("wandb" if USE_WANDB and os.environ.get("WANDB_API_KEY") else "none"),
    eval_steps=eval_save_steps,
    save_steps=eval_save_steps,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    label_smoothing_factor=0.0,  # label smoothing is set via model.config in trainer
    seed=42,
    data_seed=42,
    max_grad_norm=1.0,
    # Differential LRs
    backbone_lr=1e-6,
    head_lr=1e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_coco/AP",
    greater_is_better=True,
)

# Albumentations pipeline
AUGMENTATIONS = [
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.3),
    A.Affine(
        scale=(0.95, 1.05),
        translate_percent=(0.0, 0.03),
        rotate=(-7, 7),
        shear=(-3, 3),
        p=0.5,
    ),
]

# Trainer (COCO-pretrained checkpoint)
trainer = RtDetvr2Trainer(
    root=ROOT,
    num_labels=num_labels,
    old2new=old2new,
    label2id=label2id,
    id2label=id2label,
    train_ann=train_ann,
    val_ann=val_ann,
    model_config=None,
    training_config=training_arguments,
    augmentations=AUGMENTATIONS,
    image_size=IMAGE_SIZE,
    from_checkpoint=CKPT,
    use_focal_loss=USE_FOCAL_LOSS,
    focal_gamma=FOCAL_GAMMA,
    focal_alpha=FOCAL_ALPHA,
    label_smoothing=LABEL_SMOOTHING_IN_MODEL,
    # non-standard config hypers
    eos_coef=EOS_COEF,
    matcher_class_cost=MATCHER_CLASS_COST,
    matcher_bbox_cost=MATCHER_BBOX_COST,
    class_loss_coef=CLASS_LOSS_COEF,
)

if USE_WANDB:
    trainer.trainer.add_callback(WandbCallback)

# COCO evaluator + confidence metrics
new2old = {v: k for k, v in trainer.old2new.items()}
coco_eval = ComputeCOCOEval(
    processor=trainer.processor,
    ann_file=str(val_ann_path),
    val_images=trainer.val_ds.images,
    new2old=new2old,
    score_thr=0.01,
)
trainer.attach_map_evaluator(coco_eval)

# Save sample validation images per checkpoint
trainer.trainer.add_callback(
    SaveValImgsInCkpt(
        trainer.processor,
        trainer.val_ds,
        trainer.id2label,
        infer_size=IMAGE_SIZE,
        k=8,
        thr=0.01,
        log_to_wandb=USE_WANDB,
    )
)

trainer.train()
print(trainer.evaluate())
