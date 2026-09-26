"""Conservative SPOR continuation from the completed 6k checkpoint.

This is a new 6k optimization phase (conceptual iterations 6k--12k).  It
keeps the successful frozen-backbone setup and lowers both trainable learning
rates instead of re-running the original warm-up stages.
"""

from detectron2.config import LazyCall as L

from odise.engine.spor import get_spor_optimizer_params

from .odise_label_coco_50e_spor import dataloader, lr_multiplier, model, optimizer, train


# The starting checkpoint has already completed the ROPC and QOBR ramps.
# Keep both objectives fully active throughout this continuation phase.
model.criterion.ropc_weight = 0.10
model.criterion.ropc_warmup_iters = 1
model.criterion.ropc_full_weight_iters = 1

# `get_spor_optimizer_params` omits every `backbone.*` parameter when no
# backbone adapter LR is provided, including feature_projections.
optimizer.params = L(get_spor_optimizer_params)(
    method_lr=2e-5,
    decoder_lr=2e-6,
    weight_decay_norm=0.0,
    weight_decay_bias=0.0,
)
optimizer.lr = 2e-5

train.max_iter = 6_000
train.grad_clip = 0.01
train.checkpointer.period = 1_000
train.checkpointer.max_to_keep = 10
train.eval_period = 1_000
train.log_period = 20
train.ddp.find_unused_parameters = True
train.cfg_name = "odise_label_coco_50e_spor_continue_6k"
train.spor = dict(
    enabled=True,
    method_lr=2e-5,
    decoder_lr=2e-6,
    lr_warmup_iters=100,
    decoder_start_iter=0,
    qobr_warmup_iters=1,
    qobr_activation_floor=1.0,
)

# Intermediate validations stay COCO-only; the final 12k-equivalent model
# receives the complete cross-dataset evaluation suite.
for _task in dataloader.extra_task.values():
    _task.final_iter_only = True
