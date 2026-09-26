"""Iteration-zero COCO-only identity evaluation for SPOR."""

from .odise_label_coco_50e_spor import dataloader, lr_multiplier, model, optimizer, train


# Identity validation only needs the three COCO metrics (AP, mIoU, PQ).
dataloader.pop("extra_task", None)
train.cfg_name = "odise_label_coco_50e_spor_identity"
