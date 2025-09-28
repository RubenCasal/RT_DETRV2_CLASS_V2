
# ⚙️ Model Configuration — RT-DETRv2

This section documents the configuration arguments used in the **`train_model_finetuned.py`** script for fine-tuning **RT-DETRv2**.

---

##  Training Parameters

| Parameter               | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| **TRAIN_BATCH**         | Training batch size per device.                                  |
| **VAL_BATCH**           | Validation batch size per device.                                |
| **GRAD_ACCUM**          | Gradient accumulation steps. Allows larger effective batch size. |
| **EPOCHS**              | Total number of training epochs.                                 |
| **LEARNING_RATE**       | Base learning rate for optimizer.                                |
| **WEIGHT_DECAY**        | Weight decay (L2 regularization).                                |
| **WARMUP_RATIO**        | Proportion of training steps used for LR warm-up.                |
| **USE_FP16 / USE_BF16** | Mixed precision training flags.                                  |
| **IMAGE_SIZE**          | Target image size (height, width).                               |
| **EVAL_RATE_EPOCHS**    | Frequency of evaluation within an epoch.                         |

---

## Loss & Regularization

| Parameter                    | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| **USE_FOCAL_LOSS**           | Enables focal loss instead of standard cross-entropy.          |
| **FOCAL_GAMMA**              | Gamma value for focal loss (focus on hard examples).           |
| **FOCAL_ALPHA**              | Alpha balancing factor for focal loss.                         |
| **LABEL_SMOOTHING_IN_MODEL** | Label smoothing factor applied inside the model configuration. |

---

## Non-standard Model Config

| Parameter              | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| **EOS_COEF**           | Weight for "no-object" class to prevent over-suppression of logits. |
| **MATCHER_CLASS_COST** | Weight of classification term in Hungarian matcher.                 |
| **MATCHER_BBOX_COST**  | Weight of bounding box term in Hungarian matcher.                   |
| **CLASS_LOSS_COEF**    | Relative weight of classification loss in total loss.               |

---



