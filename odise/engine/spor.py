"""Training schedule utilities for ODFA + QOBR + ROPC (SPOR)."""

import logging
import math

from detectron2.engine import HookBase
from detectron2.solver.build import get_default_optimizer_params


_LOGGER = logging.getLogger(__name__)


def get_spor_optimizer_params(
    model,
    method_lr=1e-4,
    decoder_lr=1e-5,
    backbone_adapter_lr=None,
    backbone_adapter_prefixes=("backbone.feature_projections.",),
    weight_decay_norm=0.0,
    weight_decay_bias=0.0,
):
    """Build optimizer groups with an optional lightweight backbone adapter.

    Decoder/head parameters stay in DDP from iteration zero but use LR=0 for
    the first 500 steps. This is equivalent to freezing updates and avoids the
    unsupported operation of adding newly trainable parameters to DDP later.
    """
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    default_groups = get_default_optimizer_params(
        model,
        base_lr=decoder_lr,
        weight_decay_norm=weight_decay_norm,
        weight_decay_bias=weight_decay_bias,
    )
    groups = []
    seen = set()
    for default_group in default_groups:
        # Detectron2 merges parameters sharing hyperparameters. Split them
        # again because one merged group can contain both backbone and SPOR
        # parameters with different roles.
        for parameter in default_group["params"]:
            name = names[id(parameter)]
            if name.startswith("backbone."):
                use_adapter = backbone_adapter_lr is not None and any(
                    name.startswith(prefix) for prefix in backbone_adapter_prefixes
                )
                if not use_adapter:
                    parameter.requires_grad_(False)
                    continue
                role = "backbone_adapter"
            else:
                if not parameter.requires_grad:
                    continue
                role = "method" if (
                    name.startswith("sem_seg_head.ordinal_feature_aggregator.")
                    or name.startswith("sem_seg_head.predictor.qobr.")
                    or name.startswith("sem_seg_head.transformer_predictor.qobr.")
                ) else "decoder"
            group = {key: value for key, value in default_group.items() if key != "params"}
            group["params"] = [parameter]
            group["spor_role"] = role
            if role == "method":
                group["lr"] = method_lr
            elif role == "backbone_adapter":
                group["lr"] = float(backbone_adapter_lr)
            else:
                group["lr"] = 0.0
            groups.append(group)
            seen.add(id(parameter))

    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    missing = trainable - seen
    if missing:
        raise RuntimeError(f"SPOR optimizer omitted {len(missing)} trainable parameters")
    return groups


class SPORStageHook(HookBase):
    """Apply the staged LR plan and the identity-safe QOBR activation gate."""

    def __init__(
        self,
        method_lr=1e-4,
        decoder_lr=1e-5,
        backbone_adapter_lr=0.0,
        lr_warmup_iters=100,
        decoder_start_iter=500,
        qobr_warmup_iters=500,
        qobr_activation_floor=0.0,
        decay_start_iter=None,
        final_lr_scale=1.0,
    ):
        self.method_lr = float(method_lr)
        self.decoder_lr = float(decoder_lr)
        self.backbone_adapter_lr = float(backbone_adapter_lr)
        self.lr_warmup_iters = int(lr_warmup_iters)
        self.decoder_start_iter = int(decoder_start_iter)
        self.qobr_warmup_iters = int(qobr_warmup_iters)
        self.qobr_activation_floor = float(qobr_activation_floor)
        self.decay_start_iter = None if decay_start_iter is None else int(decay_start_iter)
        self.final_lr_scale = float(final_lr_scale)

    def _model(self):
        model = self.trainer.model
        return model.module if hasattr(model, "module") else model

    def _set_schedule(self):
        iteration = int(self.trainer.iter)
        warmup = min(float(iteration + 1) / max(self.lr_warmup_iters, 1), 1.0)
        lr_scale = warmup
        if self.decay_start_iter is not None and iteration >= self.decay_start_iter:
            decay_span = max(int(self.trainer.max_iter) - self.decay_start_iter, 1)
            progress = min(float(iteration - self.decay_start_iter) / decay_span, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_scale = self.final_lr_scale + (1.0 - self.final_lr_scale) * cosine
        decoder_active = iteration >= self.decoder_start_iter
        for group in self.trainer.optimizer.param_groups:
            role = group.get("spor_role")
            if role == "method":
                group["lr"] = self.method_lr * lr_scale
            elif role == "decoder":
                group["lr"] = self.decoder_lr * lr_scale if decoder_active else 0.0
            elif role == "backbone_adapter":
                group["lr"] = self.backbone_adapter_lr * lr_scale

        predictor = getattr(getattr(self._model(), "sem_seg_head", None), "predictor", None)
        qobr = getattr(predictor, "qobr", None)
        activation = max(
            self.qobr_activation_floor,
            min(float(iteration) / max(self.qobr_warmup_iters, 1), 1.0),
        )
        if qobr is not None:
            qobr.set_activation(activation)

        storage = getattr(self.trainer, "storage", None)
        if storage is not None:
            storage.put_scalar("spor/method_lr", self.method_lr * lr_scale, smoothing_hint=False)
            storage.put_scalar(
                "spor/decoder_lr", self.decoder_lr * lr_scale if decoder_active else 0.0,
                smoothing_hint=False,
            )
            storage.put_scalar(
                "spor/backbone_adapter_lr",
                self.backbone_adapter_lr * lr_scale,
                smoothing_hint=False,
            )
            storage.put_scalar("spor/qobr_activation", activation, smoothing_hint=False)

    def before_train(self):
        self._set_schedule()
        _LOGGER.info(
            "SPOR schedule active: method LR=%g, decoder LR=%g from iter %d, adapter LR=%g",
            self.method_lr,
            self.decoder_lr,
            self.decoder_start_iter,
            self.backbone_adapter_lr,
        )

    def before_step(self):
        self._set_schedule()
