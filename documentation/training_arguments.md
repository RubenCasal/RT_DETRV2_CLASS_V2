
# 🎛️ Training Arguments — RT-DETRv2

This section defines the **training setup** for RT-DETRv2. These parameters directly control **performance, stability, and convergence speed**.

---
| Parameter                       | Description                                                                               |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| **per_device_train_batch_size** | Number of images processed per GPU during training (mini-batch size).                     |
| **per_device_eval_batch_size**  | Batch size used during validation.                                                        |
| **gradient_accumulation_steps** | Accumulates gradients over multiple steps to simulate a larger batch size.                |
| **num_train_epochs**            | Total number of passes over the dataset.                                                  |
| **learning_rate**               | Initial learning rate for the optimizer. Controls how fast the model updates weights.     |
| **weight_decay**                | L2 regularization to reduce overfitting by penalizing large weights.                      |
| **lr_scheduler_type**           | Strategy for adapting the learning rate during training (e.g., `"constant_with_warmup"`). |
| **warmup_ratio**                | Percentage of training steps used for LR warm-up to stabilize early training.             |
| **max_grad_norm**               | Clips gradient norm to prevent exploding gradients.                                       |



