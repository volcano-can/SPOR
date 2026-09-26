"""Query-conditioned Ordinal Boundary Routing (QOBR)."""

import torch
from torch import nn
from torch.nn import functional as F


class QueryConditionedOrdinalBoundaryRouter(nn.Module):
    """Route decoder cross-attention using a query-conditioned rank interval.

    The router only changes mask cross-attention support. Class logits, text
    embeddings, and semantic mask pooling never receive this depth signal.
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        start_layer=1,
        num_bins=16,
        max_strength=0.75,
        strength_init=0.20,
        warmup_iters=500,
        interval_mass=0.70,
        max_interval_width=0.35,
        routing_eta=0.5,
        continuous_bias_scale=1.0,
        activation_init=0.0,
    ):
        super().__init__()
        if not 0.0 < strength_init < max_strength:
            raise ValueError("strength_init must be in (0, max_strength)")
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.start_layer = int(start_layer)
        self.num_bins = int(num_bins)
        self.max_strength = float(max_strength)
        self.warmup_iters = int(warmup_iters)
        self.interval_mass = float(interval_mass)
        self.max_interval_width = float(max_interval_width)
        self.routing_eta = float(routing_eta)
        self.continuous_bias_scale = float(continuous_bias_scale)
        self.layer_schedule = (0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0, 1.0, 1.0)

        # The doc specifies the routing amplitude as a fixed, bounded scalar
        # equal to ``strength_init`` (default 0.20). We therefore store it as
        # a non-trainable buffer rather than a learnable parameter: in
        # routing-only mode (``continuous_bias_scale=0.0``) the hard mask is
        # the only signal carrying this strength, and making it constant
        # avoids re-introducing a depth-conditioned parameter outside of the
        # open-vocabulary semantic path.
        if not 0.0 <= strength_init <= max_strength:
            raise ValueError("strength_init must be in [0, max_strength]")
        self.register_buffer("strength_value", torch.tensor(float(strength_init)), persistent=True)
        # A checkpoint-compatible scalar gate keeps an official ODISE
        # checkpoint exactly identical at iteration zero. The training hook
        # ramps it to one during the first 500 iterations.
        self.register_buffer(
            "activation", torch.tensor(float(activation_init)).clamp_(0.0, 1.0), persistent=True
        )

    @property
    def strength(self):
        return float(self.strength_value.item())

    @torch.no_grad()
    def set_activation(self, value):
        self.activation.fill_(min(max(float(value), 0.0), 1.0))

    @staticmethod
    def _masked_area_pool(value, valid, size):
        valid_fraction = F.interpolate(valid.float(), size=size, mode="area")
        pooled = F.interpolate(value * valid.float(), size=size, mode="area")
        pooled = pooled / valid_fraction.clamp_min(1e-6)
        return pooled, valid_fraction >= 0.999

    @staticmethod
    def _ordinal_rank(depth, valid):
        """Tie-aware within-image mid-ranks over explicit valid pixels."""
        values = torch.nan_to_num(depth[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = valid[:, 0].bool() & torch.isfinite(depth[:, 0])
        flat = values.flatten(1)
        flat_valid = valid.flatten(1)
        sortable = flat.masked_fill(~flat_valid, float("inf"))
        ordered = sortable.sort(dim=1).values
        left = torch.searchsorted(ordered, flat, right=False)
        right = torch.searchsorted(ordered, flat, right=True)
        count = flat_valid.sum(dim=1, keepdim=True)
        denominator = (count - 1).clamp_min(1).to(values.dtype)
        rank = ((left + right - 1).to(values.dtype) * 0.5 / denominator).reshape_as(values)
        rank = rank * valid.to(rank.dtype)
        rank = torch.where(count.reshape(-1, 1, 1) > 1, rank, torch.zeros_like(rank))
        return rank[:, None].detach(), valid[:, None].detach()

    def encode_depth(self, depth, depth_valid, sizes):
        """Compute high-resolution ranks once, then valid-aware pooled maps."""
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = depth[:, None]
        if depth_valid is None:
            depth_valid = torch.ones_like(depth, dtype=torch.bool)
        elif depth_valid.ndim == 3:
            depth_valid = depth_valid[:, None]

        with torch.no_grad():
            rank, valid = self._ordinal_rank(depth.detach(), depth_valid.detach())
            pooled = [self._masked_area_pool(rank, valid, size) for size in sizes]
        reference_index = max(range(len(sizes)), key=lambda index: sizes[index][0] * sizes[index][1])
        return {"levels": pooled, "reference_index": reference_index}

    def _histogram(self, rank, valid, mask_logits):
        """Build a linear-interpolation rank histogram without gradients."""
        batch, queries, _, _ = mask_logits.shape
        probabilities = mask_logits.float().sigmoid()
        valid_map = valid[:, 0].to(probabilities.dtype)
        base_weight = F.relu((probabilities - 0.20) / 0.80).square()
        weight = base_weight * valid_map[:, None]
        flat_weight = weight.flatten(2)
        total_weight = flat_weight.sum(dim=-1).clamp_min(1e-6)

        scaled_rank = (rank[:, 0].clamp(0.0, 1.0) * (self.num_bins - 1)).flatten(1)
        lower = scaled_rank.floor().long()
        upper = (lower + 1).clamp_max(self.num_bins - 1)
        upper_fraction = scaled_rank - lower.to(scaled_rank.dtype)
        lower_fraction = 1.0 - upper_fraction

        histogram = weight.new_zeros(batch, queries, self.num_bins)
        lower_index = lower[:, None].expand(-1, queries, -1)
        upper_index = upper[:, None].expand(-1, queries, -1)
        histogram.scatter_add_(2, lower_index, flat_weight * lower_fraction[:, None])
        histogram.scatter_add_(2, upper_index, flat_weight * upper_fraction[:, None])
        histogram = histogram / total_weight[:, :, None]

        foreground = (probabilities >= 0.5) & valid_map[:, None].bool()
        foreground_count = foreground.flatten(2).sum(dim=-1)
        foreground_mean = (probabilities * foreground).flatten(2).sum(dim=-1) / foreground_count.clamp_min(1)
        c_mask = ((foreground_mean - 0.55) / 0.10).sigmoid()
        effective_support = total_weight.square() / flat_weight.square().sum(dim=-1).clamp_min(1e-6)
        c_support = (effective_support / 32.0).clamp(max=1.0)
        c_valid = flat_weight.sum(dim=-1) / base_weight.flatten(2).sum(dim=-1).clamp_min(1e-6)
        max_probability = probabilities.masked_fill(~valid_map[:, None].bool(), 0.0).flatten(2).amax(dim=-1)
        return {
            "histogram": histogram,
            "c_mask": c_mask,
            "c_support": c_support,
            "c_valid": c_valid,
            "foreground_count": foreground_count,
            "effective_support": effective_support,
            "max_probability": max_probability,
        }

    def _dominant_interval(self, histogram):
        """Form a connected interval around the dominant histogram mode."""
        _, _, bins = histogram.shape
        peak = histogram.argmax(dim=-1)
        left = peak.clone()
        right = peak.clone()
        mass = histogram.gather(-1, peak[:, :, None]).squeeze(-1)
        bin_width = 1.0 / (bins - 1)

        for _ in range(bins - 1):
            can_extend_left = left > 0
            can_extend_right = right < bins - 1
            left_index = (left - 1).clamp_min(0)
            right_index = (right + 1).clamp_max(bins - 1)
            left_mass = histogram.gather(-1, left_index[:, :, None]).squeeze(-1)
            right_mass = histogram.gather(-1, right_index[:, :, None]).squeeze(-1)
            take_right = (right_mass >= left_mass) & can_extend_right
            has_candidate = torch.where(take_right, can_extend_right, can_extend_left)
            new_left = torch.where(take_right, left, left_index)
            new_right = torch.where(take_right, right_index, right)
            new_width = (new_right - new_left + 1).to(histogram.dtype) * bin_width
            accept = has_candidate & (mass < self.interval_mass) & (new_width <= self.max_interval_width + 1e-6)
            added_mass = torch.where(take_right, right_mass, left_mass)
            left = torch.where(accept, new_left, left)
            right = torch.where(accept, new_right, right)
            mass = torch.where(accept, mass + added_mass, mass)

        centers = torch.arange(bins, device=histogram.device)[None, None, :]
        smooth = F.avg_pool1d(histogram.reshape(-1, 1, bins), kernel_size=3, stride=1, padding=1).reshape_as(histogram)
        previous = F.pad(smooth[:, :, :-1], (1, 0), value=-1.0)
        following = F.pad(smooth[:, :, 1:], (0, 1), value=-1.0)
        local_peak = (smooth >= previous) & (smooth >= following)
        outside = (centers < (left[:, :, None] - 1)) | (centers > (right[:, :, None] + 1))
        second = smooth.masked_fill(~(local_peak & outside), -1.0).amax(dim=-1).clamp_min(0.0)
        first = smooth.gather(-1, peak[:, :, None]).squeeze(-1)
        unimodality = 1.0 - second / first.clamp_min(1e-6)
        c_mode = (unimodality * (mass / self.interval_mass).clamp(max=1.0)).clamp(0.0, 1.0)

        a = ((left.to(histogram.dtype) - 0.5) * bin_width).clamp(0.0, 1.0)
        b = ((right.to(histogram.dtype) + 0.5) * bin_width).clamp(0.0, 1.0)
        return a, b, mass, c_mode

    def _warmup(self, iteration):
        if not self.training or self.warmup_iters <= 0:
            return 1.0
        if iteration is None:
            try:
                from detectron2.utils.events import get_event_storage

                iteration = get_event_storage().iter
            except (AssertionError, RuntimeError):
                iteration = self.warmup_iters
        return min(float(iteration) / max(self.warmup_iters, 1), 1.0)

    def _log_diagnostics(self, layer_index, diagnostics):
        if not self.training:
            return
        try:
            from detectron2.utils.events import get_event_storage

            storage = get_event_storage()
            prefix = f"qobr/layer_{layer_index}"
            for name, value in diagnostics.items():
                storage.put_scalar(f"{prefix}/{name}", float(value.detach().mean().item()), smoothing_hint=False)
        except (AssertionError, RuntimeError):
            pass

    def forward(self, previous_attention_logits, encoded_depth, layer_index, iteration=None):
        """Return an additive bias and protected routing mask for one layer."""
        if encoded_depth is None:
            return None
        level_rank, level_valid = encoded_depth["levels"][layer_index % len(encoded_depth["levels"])]
        reference_rank, reference_valid = encoded_depth["levels"][encoded_depth["reference_index"]]

        with torch.no_grad():
            reference_logits = F.interpolate(
                previous_attention_logits.detach().float(),
                size=reference_rank.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            statistics = self._histogram(reference_rank, reference_valid, reference_logits)
            a, b, interval_mass, c_mode = self._dominant_interval(statistics["histogram"])
            reliable = (
                (statistics["foreground_count"] >= 8)
                & (statistics["effective_support"] >= 8.0)
                & (statistics["c_valid"] >= 0.50)
                & (statistics["max_probability"] >= 0.50)
                & (interval_mass >= 0.50)
            )
            reliability = (
                statistics["c_mask"]
                * statistics["c_support"]
                * statistics["c_valid"]
                * c_mode
            ) * reliable.to(reference_logits.dtype)

            current_logits = F.interpolate(
                previous_attention_logits.detach().float(),
                size=level_rank.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            probability = current_logits.sigmoid()
            rank = level_rank[:, 0]
            distance = F.relu(a[:, :, None, None] - rank[:, None]) + F.relu(rank[:, None] - b[:, :, None, None])
            scale = (b - a + 0.05).clamp(0.08, 0.35)
            compatibility = (1.0 - distance / scale[:, :, None, None]).clamp(-1.5, 1.0)
            compatibility = compatibility * level_valid[:, 0, None].to(compatibility.dtype)

        schedule = self.layer_schedule[min(layer_index, len(self.layer_schedule) - 1)]
        layer_strength = self.strength * float(self.activation.item()) * (
            schedule * self._warmup(iteration)
        )
        raw_bias = layer_strength * reliability[:, :, None, None] * compatibility
        # QOBR routing-only uses the ordinal signal to decide which
        # uncertain boundary pixels are readable, while keeping the attention
        # logits unchanged.  The separate scale preserves that ablation and
        # still lets controlled ablations request a continuous additive bias.
        bias = raw_bias * self.continuous_bias_scale

        with torch.no_grad():
            routed_logits = current_logits + self.routing_eta * raw_bias.detach()
            routing_mask = torch.where(
                probability >= 0.70,
                torch.zeros_like(probability, dtype=torch.bool),
                torch.where(
                    probability < 0.20,
                    torch.ones_like(probability, dtype=torch.bool),
                    routed_logits.sigmoid() < 0.50,
                ),
            )
            baseline_mask = probability < 0.50
            routing_mask = torch.where(~level_valid[:, 0, None], baseline_mask, routing_mask)
            route_add = baseline_mask & ~routing_mask
            route_delete = ~baseline_mask & routing_mask
            self._log_diagnostics(
                layer_index,
                {
                    "lambda": torch.tensor(layer_strength, device=reference_logits.device),
                    "reliability": reliability,
                    "active_query_ratio": reliable.to(reliability.dtype),
                    "interval_mass": interval_mass,
                    "routing_add_ratio": route_add.to(reliability.dtype),
                    "routing_delete_ratio": route_delete.to(reliability.dtype),
                    "bias_abs_mean": bias.detach().abs(),
                },
            )

        additive_bias = bias[:, None].expand(-1, self.num_heads, -1, -1, -1).flatten(0, 1).flatten(2)
        hard_mask = routing_mask[:, None].expand(-1, self.num_heads, -1, -1, -1).flatten(0, 1).flatten(2)
        return {"bias": additive_bias, "hard_mask": hard_mask}
