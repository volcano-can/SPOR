#!/usr/bin/env python3
"""CPU invariants for ODFA, QOBR, ROPC, and the staged optimizer."""

import torch

from mask2former.modeling.criterion import SetCriterion
from odise.modeling.geometry.ordinal_feature_aggregation import OrdinalDepthFeatureAggregator
from odise.modeling.geometry.qobr import QueryConditionedOrdinalBoundaryRouter


def test_odfa_identity_and_ordinal_invariance():
    torch.manual_seed(0)
    module = OrdinalDepthFeatureAggregator(channels=8, hidden_channels=8)
    pyramid = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 8, 8), torch.randn(1, 8, 16, 16)]
    mask = torch.randn(1, 8, 16, 16)
    depth = torch.linspace(0.0, 1.0, 256).reshape(1, 1, 16, 16)
    valid = torch.ones_like(depth, dtype=torch.bool)

    updated, updated_mask = module(pyramid, mask, depth, valid)
    assert all(torch.equal(before, after) for before, after in zip(pyramid, updated))
    assert torch.equal(mask, updated_mask)

    with torch.no_grad():
        module.level_out[0].weight.normal_(std=0.01)
        module.mask_out.weight.normal_(std=0.01)
        baseline, baseline_mask = module(pyramid, mask, depth, valid)
        transformed, transformed_mask = module(pyramid, mask, depth * 3.7 + 5.0, valid)
    assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(baseline, transformed))
    assert torch.allclose(baseline_mask, transformed_mask, atol=1e-5, rtol=1e-5)


def test_qobr_exact_identity_gate():
    router = QueryConditionedOrdinalBoundaryRouter(
        num_layers=3,
        num_heads=2,
        start_layer=1,
        warmup_iters=0,
        continuous_bias_scale=0.0,
        activation_init=0.0,
    ).eval()
    depth = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8)
    valid = torch.ones_like(depth, dtype=torch.bool)
    encoded = router.encode_depth(depth, valid, [(4, 4), (2, 2), (1, 1)])
    logits = torch.randn(1, 3, 4, 4)
    result = router(logits, encoded, layer_index=1)
    baseline = torch.nn.functional.interpolate(
        logits, size=(2, 2), mode="bilinear", align_corners=False
    ).sigmoid() < 0.5
    expected = baseline[:, None].expand(-1, 2, -1, -1, -1).flatten(0, 1).flatten(2)
    assert torch.equal(result["hard_mask"], expected)
    assert torch.count_nonzero(result["bias"]) == 0


def test_ropc_finite_and_query_permutation_invariant():
    criterion = SetCriterion(
        num_classes=2,
        matcher=None,
        class_weight=1.0,
        mask_weight=1.0,
        dice_weight=1.0,
        num_layers=1,
        eos_coef=0.1,
        losses=[],
        num_points=16,
        oversample_ratio=1.0,
        importance_sample_ratio=1.0,
        enable_ropc=True,
        ropc_max_pairs=64,
    ).eval()
    logits = torch.randn(1, 4, 8, 8, requires_grad=True)
    masks = torch.zeros(2, 16, 16, dtype=torch.bool)
    masks[0, :, :8] = True
    masks[1, :, 8:] = True
    depth = torch.linspace(0.0, 1.0, 256).reshape(1, 1, 16, 16)
    valid = torch.ones_like(depth, dtype=torch.bool)
    outputs = {"pred_masks": logits, "depth": depth, "depth_valid": valid}
    torch.manual_seed(11)
    loss = criterion.loss_ropc(outputs, [{"masks": masks}])["loss_ropc"]
    assert torch.isfinite(loss)
    loss.backward()
    permutation = torch.tensor([2, 0, 3, 1])
    torch.manual_seed(11)
    permuted = criterion.loss_ropc(
        {"pred_masks": logits.detach()[:, permutation], "depth": depth, "depth_valid": valid},
        [{"masks": masks}],
    )["loss_ropc"]
    assert torch.allclose(loss.detach(), permuted, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_odfa_identity_and_ordinal_invariance()
    test_qobr_exact_identity_gate()
    test_ropc_finite_and_query_permutation_invariant()
    print("SPOR component tests passed")
