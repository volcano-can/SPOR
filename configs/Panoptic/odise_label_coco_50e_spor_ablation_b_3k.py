"""Ablation B at 3k iterations, matched to C's 3k checkpoint.

All SPOR modules are removed but the data pipeline, official initialization,
optimizer, freeze schedule, batch size, and random seed are inherited from
the SPOR setting.  The retained extra tasks provide the complete transfer
evaluation suite at the final iteration.
"""

from .odise_label_coco_50e_spor import dataloader, lr_multiplier, model, optimizer, train


model.sem_seg_head.enable_ordinal_feature_aggregation = False
model.sem_seg_head.use_ordinal_feature_aggregation_inference = False
model.sem_seg_head.ordinal_feature_aggregator = None
model.sem_seg_head.transformer_predictor.use_qobr = False
model.criterion.enable_ropc = False

train.max_iter = 3_000
train.checkpointer.period = 1_000
train.checkpointer.max_to_keep = 3
train.eval_period = 3_000
train.log_period = 20
train.ddp.find_unused_parameters = True
train.cfg_name = "odise_label_coco_50e_spor_ablation_b_3k"
