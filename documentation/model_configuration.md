
# ⚙️ Model Configuration — RT-DETRv2

This section documents the configuration arguments used in the **`train_model_finetuned.py`** script for fine-tuning **RT-DETRv2**.

---


| Parameter                       | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| **num_labels**                  | Number of classes the model needs to predict.                               |
| **id2label / label2id**         | Ensure predictions are correctly translated into human-readable labels.      |
| **backbone**                    | Defines the base network used as feature extractor (e.g., `microsoft/resnet-50`). |
| **use_pretrained_backbone**     | Controls whether to start from pretrained weights (e.g., ImageNet) or train from scratch. |
| **freeze_backbone_batch_norms** | Option to freeze BatchNorm layers to stabilize training.                     |
| **backbone_kwargs (out_indices)** | Selects which backbone layers are used to generate feature maps (e.g., stages 1, 2, and 3 of ResNet). |
| **encoder_in_channels**         | Number of channels the encoder receives from the backbone.                   |
| **feat_strides**                | Defines the scales at which objects are analyzed (e.g., `[8, 16, 32]`).      |
| **decoder_method**              | Strategy used by the decoder to process queries (e.g., `"discrete"`).        |

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



