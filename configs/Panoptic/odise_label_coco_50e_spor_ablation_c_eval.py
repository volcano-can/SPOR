"""Evaluation configuration for the ODFA-only (C) ablation.

It intentionally retains the full cross-dataset evaluation suite from SPOR.
"""

from .odise_label_coco_50e_spor import dataloader, lr_multiplier, model, optimizer, train


# Match the architecture that produced C's checkpoints: ODFA on, QOBR and
# ROPC off.  Keep ``extra_task`` so eval-only runs all configured datasets.
model.sem_seg_head.enable_ordinal_feature_aggregation = True
model.sem_seg_head.use_ordinal_feature_aggregation_inference = True
model.sem_seg_head.transformer_predictor.use_qobr = False
model.criterion.enable_ropc = False

train.cfg_name = "odise_label_coco_50e_spor_ablation_c_eval"
