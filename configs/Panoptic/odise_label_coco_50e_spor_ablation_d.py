"""Ablation D: matched 6k fine-tuning with ODFA and QOBR, without ROPC."""

from .odise_label_coco_50e_spor import dataloader, lr_multiplier, model, optimizer, train


# ODFA and QOBR are inherited enabled from the full SPOR configuration.
# Remove only ROPC for the cumulative ablation.
model.criterion.enable_ropc = False

train.max_iter = 6_000
train.checkpointer.period = 1_000
train.checkpointer.max_to_keep = 6
train.eval_period = 6_000
train.log_period = 20
train.ddp.find_unused_parameters = True
train.cfg_name = "odise_label_coco_50e_spor_ablation_d"

# Unlike B/C, retain the full configuration's extra_task suite. Those tasks
# already have final_iter_only=True, so all datasets are evaluated only at 6k.
