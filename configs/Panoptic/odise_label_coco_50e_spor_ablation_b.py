"""Ablation B: matched 6k ODISE fine-tuning without SPOR modules.

This control uses the exact SPOR data pipeline, initialization, batch size,
optimizer schedule, frozen backbone, decoder unfreeze point, and seed.  It
only removes ODFA, QOBR, and ROPC, so any difference against full SPOR cannot
be attributed to the extra 6k optimization budget.
"""

from .odise_label_coco_50e_spor import dataloader, lr_multiplier, model, optimizer, train


# Remove each proposed component while retaining the depth-aware mapper used
# by the full configuration as a data-pipeline control.
model.sem_seg_head.enable_ordinal_feature_aggregation = False
model.sem_seg_head.use_ordinal_feature_aggregation_inference = False
model.sem_seg_head.ordinal_feature_aggregator = None

predictor = model.sem_seg_head.transformer_predictor
predictor.use_qobr = False

model.criterion.enable_ropc = False

# Match the full 6k schedule, but evaluate COCO only at completion.  The
# cross-dataset suite is deliberately excluded here to reserve time for the
# remaining cumulative ablations.
train.max_iter = 6_000
train.checkpointer.period = 6_000
train.checkpointer.max_to_keep = 1
train.eval_period = 6_000
train.log_period = 20
train.ddp.find_unused_parameters = True
train.cfg_name = "odise_label_coco_50e_spor_ablation_b"

dataloader.pop("extra_task", None)
