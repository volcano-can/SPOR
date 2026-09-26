# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, class_weight, mask_weight, dice_weight, num_layers, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 enable_ropc=False, ropc_weight=0.1, ropc_warmup_iters=500,
                 ropc_full_weight_iters=2000, ropc_max_pairs=4096,
                 ropc_temperature=1.0, ropc_reliability_sigma=0.10,
                 ropc_positive_tau=0.10, ropc_negative_tau=0.10,
                 ropc_offsets=((1, 0), (0, 1), (2, 0), (0, 2), (4, 0), (0, 4)),
                 ropc_aux_weights=(0.25, 0.5)):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if enable_ropc:
            weight_dict["loss_ropc"] = ropc_weight
        aux_weight_dict = {}
        for i in range(num_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.weight_dict = weight_dict

        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.enable_ropc = bool(enable_ropc)
        self.ropc_weight = float(ropc_weight)
        self.ropc_warmup_iters = int(ropc_warmup_iters)
        self.ropc_full_weight_iters = int(ropc_full_weight_iters)
        self.ropc_max_pairs = int(ropc_max_pairs)
        self.ropc_temperature = float(ropc_temperature)
        self.ropc_reliability_sigma = float(ropc_reliability_sigma)
        self.ropc_positive_tau = float(ropc_positive_tau)
        self.ropc_negative_tau = float(ropc_negative_tau)
        self.ropc_offsets = tuple(tuple(int(v) for v in offset) for offset in ropc_offsets)
        self.ropc_aux_weights = tuple(float(weight) for weight in ropc_aux_weights)

    @staticmethod
    def _ropc_ordinal_rank(depth, valid):
        """Tie-aware rank over valid pixels; invariant to monotone depth transforms."""
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if valid is None:
            valid = torch.ones_like(depth, dtype=torch.bool)
        elif valid.ndim == 3:
            valid = valid.unsqueeze(1)
        valid = valid.bool() & torch.isfinite(depth)
        values = torch.nan_to_num(depth[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid_2d = valid[:, 0]
        flat = values.flatten(1)
        flat_valid = valid_2d.flatten(1)
        ordered = flat.masked_fill(~flat_valid, float("inf")).sort(dim=1).values
        left = torch.searchsorted(ordered, flat, right=False)
        right = torch.searchsorted(ordered, flat, right=True)
        count = flat_valid.sum(dim=1, keepdim=True)
        denominator = (count - 1).clamp_min(1).to(values.dtype)
        rank = ((left + right - 1).to(values.dtype) * 0.5 / denominator).reshape_as(values)
        rank = rank * valid_2d.to(rank.dtype)
        rank = torch.where(count.reshape(-1, 1, 1) > 1, rank, torch.zeros_like(rank))
        return rank[:, None], valid

    @staticmethod
    def _ropc_valid_aware_pool(value, valid, size):
        valid_fraction = F.interpolate(valid.float(), size=size, mode="area")
        pooled = F.interpolate(value * valid.float(), size=size, mode="area")
        pooled = pooled / valid_fraction.clamp_min(1e-6)
        pooled_valid = valid_fraction >= 0.99
        return pooled * pooled_valid.to(pooled.dtype), pooled_valid

    def _ropc_schedule(self):
        """0->0.5 by 500 iters, then 0.5->1.0 by 2k iters."""
        if not self.training:
            return 1.0
        try:
            iteration = float(get_event_storage().iter)
        except (AssertionError, RuntimeError):
            iteration = float(self.ropc_full_weight_iters)
        half_iter = max(float(self.ropc_warmup_iters), 1.0)
        full_iter = max(float(self.ropc_full_weight_iters), half_iter + 1.0)
        if iteration <= half_iter:
            return 0.5 * iteration / half_iter
        if iteration < full_iter:
            return 0.5 + 0.5 * (iteration - half_iter) / (full_iter - half_iter)
        return 1.0

    def _ropc_sample_indices(self, indices):
        if indices.numel() <= self.ropc_max_pairs:
            return indices
        order = torch.randperm(indices.numel(), device=indices.device)[: self.ropc_max_pairs]
        return indices[order]

    @staticmethod
    def _ropc_segment_ids(target_masks, size):
        if target_masks.numel() == 0:
            return None, None
        masks = F.interpolate(target_masks[:, None].float(), size=size, mode="nearest")[:, 0] > 0.5
        valid = masks.any(dim=0)
        return masks.to(torch.int64).argmax(dim=0), valid

    def _ropc_image_loss(self, mask_logits, target_masks, depth, depth_valid):
        height, width = mask_logits.shape[-2:]
        segment_ids, gt_valid = self._ropc_segment_ids(target_masks, (height, width))
        if segment_ids is None or not gt_valid.any():
            return mask_logits.sum() * 0.0

        with torch.no_grad():
            full_rank, full_valid = self._ropc_ordinal_rank(depth[None], depth_valid[None])
            rank, valid = self._ropc_valid_aware_pool(full_rank, full_valid, (height, width))
            rank, valid = rank[0, 0], valid[0, 0]
            coarse_size = (max(1, (height + 1) // 2), max(1, (width + 1) // 2))
            coarse_rank, coarse_valid = self._ropc_valid_aware_pool(
                full_rank, full_valid, coarse_size
            )
            numerator = F.interpolate(
                coarse_rank * coarse_valid.to(coarse_rank.dtype),
                size=(height, width), mode="bilinear", align_corners=False,
            )
            denominator = F.interpolate(
                coarse_valid.to(coarse_rank.dtype),
                size=(height, width), mode="bilinear", align_corners=False,
            )
            coarse_rank = (numerator / denominator.clamp_min(1e-6))[0, 0]
            coarse_valid = denominator[0, 0] >= 0.99

            pixel_ids = torch.arange(height * width, device=mask_logits.device).reshape(height, width)
            left_ids, right_ids, edge_weights, edge_positive = [], [], [], []
            sigma = max(self.ropc_reliability_sigma, 1e-6)
            tau_positive = max(self.ropc_positive_tau, 1e-6)
            tau_negative = max(self.ropc_negative_tau, 1e-6)
            for delta_y, delta_x in self.ropc_offsets:
                if delta_y < 0 or delta_x < 0 or delta_y >= height or delta_x >= width:
                    continue
                source = (slice(0, height - delta_y), slice(0, width - delta_x))
                target = (slice(delta_y, height), slice(delta_x, width))
                usable = (
                    valid[source] & coarse_valid[source] & gt_valid[source]
                    & valid[target] & coarse_valid[target] & gt_valid[target]
                )
                if not usable.any():
                    continue
                gap = (rank[source] - rank[target]).abs()
                coarse_gap = (coarse_rank[source] - coarse_rank[target]).abs()
                reliability = torch.exp(-(gap - coarse_gap).abs() / sigma)
                positive = segment_ids[source] == segment_ids[target]
                weight = torch.where(
                    positive,
                    reliability * torch.exp(-gap / tau_positive),
                    reliability * (1.0 - torch.exp(-gap / tau_negative)),
                )
                keep = usable & (weight > 0.0)
                if keep.any():
                    left_ids.append(pixel_ids[source][keep])
                    right_ids.append(pixel_ids[target][keep])
                    edge_weights.append(weight[keep])
                    edge_positive.append(positive[keep])
            if not left_ids:
                return mask_logits.sum() * 0.0
            left_ids = torch.cat(left_ids)
            right_ids = torch.cat(right_ids)
            edge_weights = torch.cat(edge_weights)
            edge_positive = torch.cat(edge_positive)

        void_logits = mask_logits.new_zeros((1, height, width))
        partition = F.softmax(
            torch.cat((mask_logits, void_logits), dim=0) / max(self.ropc_temperature, 1e-6),
            dim=0, dtype=torch.float32,
        )[:-1].flatten(1)
        terms = []
        for is_positive in (True, False):
            candidates = torch.nonzero(edge_positive == is_positive, as_tuple=False).flatten()
            candidates = self._ropc_sample_indices(candidates)
            if candidates.numel() == 0:
                terms.append(mask_logits.sum() * 0.0)
                continue
            same = (
                partition[:, left_ids[candidates]] * partition[:, right_ids[candidates]]
            ).sum(dim=0)
            weights = edge_weights[candidates].float()
            penalty = 1.0 - same if is_positive else same
            terms.append((weights * penalty).sum() / weights.sum().clamp_min(1e-6))
        return torch.stack(terms).mean()

    def loss_ropc(self, outputs, targets, layer_weight=1.0):
        if "depth" not in outputs or "depth_valid" not in outputs:
            return {"loss_ropc": outputs["pred_masks"].sum() * 0.0}
        losses = [
            self._ropc_image_loss(
                outputs["pred_masks"][i], target["masks"], outputs["depth"][i],
                outputs["depth_valid"][i],
            )
            for i, target in enumerate(targets)
        ]
        loss = torch.stack(losses).mean() if losses else outputs["pred_masks"].sum() * 0.0
        return {"loss_ropc": loss * float(layer_weight) * self._ropc_schedule()}

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        if self.enable_ropc:
            losses.update(self.loss_ropc(outputs, targets))
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if self.enable_ropc and self.ropc_aux_weights:
                aux_outputs = outputs["aux_outputs"]
                start = max(0, len(aux_outputs) - len(self.ropc_aux_weights))
                for index, layer_weight in enumerate(self.ropc_aux_weights, start=start):
                    if layer_weight <= 0.0:
                        continue
                    ropc_outputs = dict(aux_outputs[index])
                    ropc_outputs["depth"] = outputs["depth"]
                    ropc_outputs["depth_valid"] = outputs["depth_valid"]
                    l_dict = self.loss_ropc(ropc_outputs, targets, layer_weight)
                    losses.update({key + f"_{index}": value for key, value in l_dict.items()})

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
            "enable_ropc: {}".format(self.enable_ropc),
            "ropc_weight: {}".format(self.ropc_weight),
            "ropc_max_pairs: {}".format(self.ropc_max_pairs),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
