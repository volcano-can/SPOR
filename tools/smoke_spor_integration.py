#!/usr/bin/env python3
"""One-batch integration smoke test for the full SPOR training path."""

import argparse

import torch
from detectron2.config import LazyConfig, instantiate

from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/Panoptic/odise_label_coco_50e_spor.py")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = LazyConfig.load(args.config)
    cfg.dataloader.train.total_batch_size = 1
    cfg.dataloader.train.num_workers = 0
    model = instantiate_odise(cfg.model).cuda().train()
    ODISECheckpointer(model).load(args.checkpoint)

    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    counts = {}
    for group in optimizer.param_groups:
        role = group.get("spor_role", "unknown")
        counts[role] = counts.get(role, 0) + sum(p.numel() for p in group["params"])
    assert counts.get("method", 0) > 0
    assert counts.get("decoder", 0) > 0
    if counts.get("backbone_adapter", 0) > 0:
        assert any(p.requires_grad for p in model.backbone.feature_projections.parameters())
        assert all(
            not p.requires_grad
            for name, p in model.backbone.named_parameters()
            if not name.startswith("feature_projections.")
        )
    else:
        assert all(not p.requires_grad for p in model.backbone.parameters())

    batch = next(iter(instantiate(cfg.dataloader.train)))
    assert batch and batch[0]["depth"] is not None and batch[0]["depth_valid"].any()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        losses = model(batch)
        total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    odfa_grad = sum(
        int(parameter.grad is not None and torch.isfinite(parameter.grad).all())
        for parameter in model.sem_seg_head.ordinal_feature_aggregator.parameters()
    )
    assert odfa_grad > 0
    print(
        {
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            "optimizer_parameter_counts": counts,
            "finite_odfa_gradient_tensors": odfa_grad,
            "qobr_activation": float(model.sem_seg_head.predictor.qobr.activation),
        }
    )


if __name__ == "__main__":
    main()
