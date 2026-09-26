"""Evaluation-only configuration for the completed component-free B baseline."""

from .odise_label_coco_50e import dataloader, lr_multiplier, model, optimizer, train


# These are the original ODISE defaults, matching B after all three SPOR
# components were disabled. `dataloader.extra_task` remains intact so COCO,
# ADE20K, Pascal Context, and Pascal VOC evaluation all run.
train.cfg_name = "odise_label_coco_50e_ablation_b_eval"
