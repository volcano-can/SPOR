"""One-shot SPOR continuation tuned to finish within 24 hours.

Starts from the completed 6k SPOR checkpoint. The diffusion feature extractor
stays frozen; only its lightweight feature projections are unfrozen.
"""

from detectron2.config import LazyCall as L

from odise.engine.spor import get_spor_optimizer_params

from .odise_label_coco_50e_spor import dataloader, lr_multiplier, model, optimizer, train


# Keep the complete ROPC objective: it is a core contribution and the 6k run
# ultimately achieved its best result at the original target weight.
model.criterion.ropc_weight = 0.10
model.criterion.ropc_warmup_iters = 1
model.criterion.ropc_full_weight_iters = 1

optimizer.params = L(get_spor_optimizer_params)(
    method_lr=1e-4,
    decoder_lr=3e-5,
    backbone_adapter_lr=5e-6,
    backbone_adapter_prefixes=("backbone.feature_projections.",),
    weight_decay_norm=0.0,
    weight_decay_bias=0.0,
)
optimizer.lr = 1e-4

train.max_iter = 3_000
train.grad_clip = 0.10
train.checkpointer.period = 500
train.checkpointer.max_to_keep = 10
train.eval_period = 1_000
train.log_period = 20
train.ddp.find_unused_parameters = True
train.cfg_name = "odise_label_coco_50e_spor_boost_3k"
train.spor = dict(
    enabled=True,
    method_lr=1e-4,
    decoder_lr=3e-5,
    backbone_adapter_lr=5e-6,
    lr_warmup_iters=100,
    decoder_start_iter=0,
    qobr_warmup_iters=1,
    qobr_activation_floor=1.0,
    decay_start_iter=2_000,
    final_lr_scale=0.10,
)

# Keep intermediate validation COCO-only. The final iteration evaluates the
# complete ODISE transfer suite.
for _task in dataloader.extra_task.values():
    _task.final_iter_only = True
