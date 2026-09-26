"""SPOR: ODFA + QOBR routing + ROPC last-three on full COCO."""

from detectron2.config import LazyCall as L

from odise.engine.spor import get_spor_optimizer_params
from odise.modeling.geometry.ordinal_feature_aggregation import (
    OrdinalDepthFeatureAggregator,
)

from .odise_label_coco_50e import dataloader, lr_multiplier, model, optimizer, train


# ODFA: zero-initialized bounded residuals preserve the official checkpoint at
# iteration zero and keep open-vocabulary semantic pooling on the RGB feature.
model.sem_seg_head.ordinal_feature_aggregator = L(OrdinalDepthFeatureAggregator)(
    channels=256,
    num_feature_levels=3,
    hidden_channels=64,
    depth_temperature=0.08,
    num_depth_layers=3,
    max_residual=0.25,
)
model.sem_seg_head.enable_ordinal_feature_aggregation = True
model.sem_seg_head.use_ordinal_feature_aggregation_inference = True

# QOBR routing-only. Its activation buffer starts at zero for exact iteration-0
# identity and is ramped to one by SPORStageHook during the first 500 steps.
predictor = model.sem_seg_head.transformer_predictor
# The base LazyConfig uses a relative interpolation here. The optimizer path
# instantiates the model independently, so make the COCO label cardinality
# explicit rather than relying on that transient parent scope.
predictor.num_classes = 133
predictor.class_embed.num_classes = 133
predictor.in_channels = 256
predictor.num_queries = 100
predictor.post_mask_embed.hidden_dim = 256
predictor.post_mask_embed.mask_dim = 256
predictor.post_mask_embed.projection_dim = 256
# `instantiate_odise` later moves the criterion/category subconfigs while
# resolving the model. Keep their values self-contained for the SPOR config.
criterion = model.criterion
criterion.num_layers = 9
criterion.num_classes = 133
criterion.matcher.cost_class = 2.0
criterion.matcher.cost_mask = 5.0
criterion.matcher.cost_dice = 5.0
criterion.matcher.num_points = 12544
model.category_head.projection_dim = 256
predictor.use_qobr = True
predictor.qobr_start_layer = 1
predictor.qobr_num_bins = 16
predictor.qobr_max_strength = 0.75
predictor.qobr_strength_init = 0.20
predictor.qobr_warmup_iters = 0
predictor.qobr_interval_mass = 0.70
predictor.qobr_max_interval_width = 0.35
predictor.qobr_routing_eta = 0.50
predictor.qobr_continuous_bias_scale = 0.0
predictor.qobr_activation_init = 0.0

# ROPC last-three with the requested two-stage weight ramp:
# 0 -> 50% by iter 500, then 50% -> 100% by iter 2000.
criterion.enable_ropc = True
criterion.ropc_weight = 0.10
criterion.ropc_warmup_iters = 500
criterion.ropc_full_weight_iters = 2000
criterion.ropc_max_pairs = 4096
criterion.ropc_temperature = 1.0
criterion.ropc_reliability_sigma = 0.10
criterion.ropc_positive_tau = 0.10
criterion.ropc_negative_tau = 0.10
criterion.ropc_offsets = ((1, 0), (0, 1), (2, 0), (0, 2), (4, 0), (0, 4))
criterion.ropc_aux_weights = (0.25, 0.5)

# Parameter roles are explicit: backbone is omitted permanently, ODFA uses
# 1e-4, and decoder/head groups stay at LR=0 until iteration 500 then use 1e-5.
optimizer.params = L(get_spor_optimizer_params)(
    method_lr=1e-4,
    decoder_lr=1e-5,
    weight_decay_norm=0.0,
    weight_decay_bias=0.0,
)
optimizer.lr = 1e-4
optimizer.weight_decay = 0.05

train.max_iter = 6_000
train.checkpointer.period = 500
train.checkpointer.max_to_keep = 20
train.eval_period = 1_000
train.log_period = 20
train.ddp.find_unused_parameters = True
train.cfg_name = "odise_label_coco_50e_spor"
train.spor = dict(
    enabled=True,
    method_lr=1e-4,
    decoder_lr=1e-5,
    lr_warmup_iters=100,
    decoder_start_iter=500,
    qobr_warmup_iters=500,
)

# Intermediate evaluations are COCO-only. The cross-dataset suite still runs
# at the final 6k evaluation.
for _task in dataloader.extra_task.values():
    _task.final_iter_only = True
