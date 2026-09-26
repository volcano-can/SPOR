"""Ablation C: matched 6k fine-tuning with ODFA only.

QOBR and ROPC are disabled.  All initialization, data, optimization, and
freeze/unfreeze settings inherit from the full SPOR configuration, making the
comparison against ablation B and full SPOR isolate ODFA's contribution.
"""

from .odise_label_coco_50e_spor import dataloader, lr_multiplier, model, optimizer, train


# Keep ODFA exactly as configured in full SPOR.
model.sem_seg_head.enable_ordinal_feature_aggregation = True
model.sem_seg_head.use_ordinal_feature_aggregation_inference = True

# Disable the remaining two contributions.
predictor = model.sem_seg_head.transformer_predictor
predictor.use_qobr = False
model.criterion.enable_ropc = False

train.max_iter = 6_000
# Retain all six periodic checkpoints: 1k through 6k.
train.checkpointer.period = 1_000
train.checkpointer.max_to_keep = 6
train.eval_period = 6_000
train.log_period = 20
train.ddp.find_unused_parameters = True
train.cfg_name = "odise_label_coco_50e_spor_ablation_c"

# Run COCO evaluation only at completion; cross-dataset transfer is reserved
# for the final full model comparison.
dataloader.pop("extra_task", None)
