"""Ordinal-depth-preserving aggregation for ODISE segmentation features."""

import torch
from torch import nn
from torch.nn import functional as F


class OrdinalDepthFeatureAggregator(nn.Module):
    """Inject ordinal geometry only into the segmentation feature pyramid.

    The module never sees text embeddings, class logits, or MaskCLIP inputs.
    It uses within-image depth ranks to restrict local feature propagation and
    to gate aggregation from adjacent pyramid levels. The output projections
    are zero initialized, giving exact baseline features before optimization.
    """

    def __init__(
        self,
        channels=256,
        num_feature_levels=3,
        hidden_channels=64,
        depth_temperature=0.08,
        num_depth_layers=3,
        max_residual=0.25,
        eps=1e-6,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_feature_levels = int(num_feature_levels)
        self.hidden_channels = int(hidden_channels)
        self.depth_temperature = float(depth_temperature)
        self.num_depth_layers = int(num_depth_layers)
        self.max_residual = float(max_residual)
        self.eps = float(eps)

        depth_channels = self.num_depth_layers + 3  # rank, edge, validity.
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden_channels, kernel_size=1),
            nn.GroupNorm(self._groups(self.hidden_channels), self.hidden_channels),
            nn.GELU(),
        )
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self._groups(self.hidden_channels), self.hidden_channels),
            nn.GELU(),
        )
        self.scale_gate = nn.ModuleList(
            [nn.Conv2d(self.hidden_channels * 2, 3, kernel_size=1) for _ in range(self.num_feature_levels)]
        )
        self.level_fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.hidden_channels * 3, self.hidden_channels, kernel_size=1),
                    nn.GroupNorm(self._groups(self.hidden_channels), self.hidden_channels),
                    nn.GELU(),
                )
                for _ in range(self.num_feature_levels)
            ]
        )
        self.level_out = nn.ModuleList(
            [nn.Conv2d(self.hidden_channels, self.channels, kernel_size=1) for _ in range(self.num_feature_levels)]
        )

        # The mask-support branch is separate from the semantic feature passed
        # to PooledMaskEmbed. It can improve mask formation without adding a
        # depth-dependent vector to ODISE's open-vocabulary text space.
        self.mask_gate = nn.Conv2d(self.hidden_channels * 2, 2, kernel_size=1)
        self.mask_fuse = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 3, self.hidden_channels, kernel_size=1),
            nn.GroupNorm(self._groups(self.hidden_channels), self.hidden_channels),
            nn.GELU(),
        )
        self.mask_out = nn.Conv2d(self.hidden_channels, self.channels, kernel_size=1)

        for module in list(self.level_out) + [self.mask_out]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _groups(channels):
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    @staticmethod
    def _ordinal_rank(depth, valid):
        """Return tie-aware rank over valid pixels of the original depth map."""
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if valid is None:
            valid = torch.ones_like(depth, dtype=torch.bool)
        elif valid.ndim == 3:
            valid = valid.unsqueeze(1)
        valid = valid.bool() & torch.isfinite(depth)
        values = torch.nan_to_num(depth[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = valid[:, 0]
        flat = values.flatten(1)
        valid_flat = valid.flatten(1)
        sortable = flat.masked_fill(~valid_flat, float("inf"))
        sorted_values = sortable.sort(dim=1).values
        left = torch.searchsorted(sorted_values, flat, right=False)
        right = torch.searchsorted(sorted_values, flat, right=True)
        valid_count = valid_flat.sum(dim=1, keepdim=True)
        denominator = (valid_count - 1).clamp_min(1).to(values.dtype)
        rank = ((left + right - 1).to(values.dtype) * 0.5 / denominator).reshape_as(values)
        rank = rank * valid.to(rank.dtype)
        rank = torch.where(valid_count.reshape(-1, 1, 1) > 1, rank, torch.zeros_like(rank))
        return rank[:, None], valid[:, None]

    @staticmethod
    def _valid_aware_pool(value, valid, size):
        """Pool rank observations without allowing padded pixels into a bin."""
        valid_fraction = F.interpolate(valid.float(), size=size, mode="area")
        pooled = F.interpolate(value * valid.float(), size=size, mode="area")
        pooled = pooled / valid_fraction.clamp_min(1e-6)
        pooled_valid = valid_fraction >= 0.999
        return pooled * pooled_valid.to(pooled.dtype), pooled_valid

    def _ordinal_map(self, full_rank, full_valid, size):
        """Pool an already-ranked valid depth map and form local ordinal cues."""
        rank, valid = self._valid_aware_pool(full_rank, full_valid, size)
        rank = rank[:, 0]
        valid_2d = valid[:, 0]
        dx = F.pad((rank[:, :, 1:] - rank[:, :, :-1]).abs(), (0, 1))
        dy = F.pad((rank[:, 1:, :] - rank[:, :-1, :]).abs(), (0, 0, 0, 1))
        edge = (dx + dy).clamp(0.0, 1.0) * valid_2d.to(rank.dtype)
        centers = torch.linspace(0.0, 1.0, self.num_depth_layers, device=rank.device, dtype=rank.dtype)
        layer_distance = (rank[:, None] - centers[None, :, None, None]).abs()
        memberships = F.softmax(-layer_distance / max(self.depth_temperature, self.eps), dim=1)
        memberships = memberships * valid.to(memberships.dtype)
        cues = torch.cat(
            (rank[:, None], edge[:, None], valid.to(rank.dtype), memberships), dim=1
        )
        return rank, valid, cues

    @staticmethod
    def _shift(tensor, dy, dx):
        shifted = torch.roll(tensor, shifts=(dy, dx), dims=(-2, -1))
        valid = torch.ones_like(tensor[:, :1], dtype=torch.bool)
        if dy > 0:
            valid[..., :dy, :] = False
        elif dy < 0:
            valid[..., dy:, :] = False
        if dx > 0:
            valid[..., :, :dx] = False
        elif dx < 0:
            valid[..., :, dx:] = False
        return shifted, valid

    def _order_preserving_pool(self, feature, rank, valid):
        """Aggregate only neighbours compatible in ordinal depth."""
        weighted = torch.zeros_like(feature)
        normalizer = torch.zeros_like(feature[:, :1])
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour_feature, in_bounds = self._shift(feature, dy, dx)
            neighbour_rank, _ = self._shift(rank[:, None], dy, dx)
            neighbour_valid, _ = self._shift(valid, dy, dx)
            affinity = torch.exp(
                -(rank[:, None] - neighbour_rank).abs() / max(self.depth_temperature, self.eps)
            )
            affinity = affinity * valid.to(affinity.dtype) * neighbour_valid.to(affinity.dtype)
            affinity = affinity * in_bounds.to(affinity.dtype)
            weighted = weighted + neighbour_feature * affinity
            normalizer = normalizer + affinity
        return (weighted + feature * self.eps) / (normalizer + self.eps)

    def _resample_and_pool(self, hidden_feature, rank, valid, size):
        if hidden_feature.shape[-2:] != size:
            hidden_feature = F.interpolate(
                hidden_feature, size=size, mode="bilinear", align_corners=False
            )
        return self._order_preserving_pool(hidden_feature, rank, valid)

    def forward(self, multi_scale_features, mask_features, depth, depth_valid=None):
        if depth is None:
            return multi_scale_features, mask_features
        if len(multi_scale_features) != self.num_feature_levels:
            raise ValueError(
                f"expected {self.num_feature_levels} pyramid levels, got {len(multi_scale_features)}"
            )

        with torch.no_grad():
            full_rank, full_valid = self._ordinal_rank(depth.detach(), depth_valid)

        hidden_features = [self.feature_proj(feature.float()) for feature in multi_scale_features]
        fused_features = []
        for level, (feature, hidden) in enumerate(zip(multi_scale_features, hidden_features)):
            rank, valid, cues = self._ordinal_map(full_rank, full_valid, feature.shape[-2:])
            depth_hidden = self.depth_proj(cues).to(hidden.dtype)
            own = self._order_preserving_pool(hidden, rank, valid)
            sources = [own]
            availability = [True]
            for neighbour in (level - 1, level + 1):
                if 0 <= neighbour < self.num_feature_levels:
                    sources.append(
                        self._resample_and_pool(hidden_features[neighbour], rank, valid, feature.shape[-2:])
                    )
                    availability.append(True)
                else:
                    sources.append(torch.zeros_like(own))
                    availability.append(False)
            gates = self.scale_gate[level](torch.cat((hidden, depth_hidden), dim=1))
            unavailable = torch.tensor(availability, device=gates.device, dtype=torch.bool)
            gates = gates.masked_fill(~unavailable[None, :, None, None], -1e4).softmax(dim=1)
            mixed = sum(gates[:, idx : idx + 1] * source for idx, source in enumerate(sources))
            fused = self.level_fuse[level](torch.cat((hidden, mixed, depth_hidden), dim=1))
            residual = self.max_residual * torch.tanh(self.level_out[level](fused)).to(feature.dtype)
            fused_features.append(feature + residual)

        # Form a depth-layer-aware mask support map from the original mask
        # feature and the highest-resolution aggregated pyramid feature.
        rank, valid, cues = self._ordinal_map(full_rank, full_valid, mask_features.shape[-2:])
        mask_hidden = self.feature_proj(mask_features.float())
        depth_hidden = self.depth_proj(cues).to(mask_hidden.dtype)
        own = self._order_preserving_pool(mask_hidden, rank, valid)
        pyramid_support = self._resample_and_pool(
            self.feature_proj(fused_features[-1].float()), rank, valid, mask_features.shape[-2:]
        )
        mask_gates = self.mask_gate(torch.cat((mask_hidden, depth_hidden), dim=1)).softmax(dim=1)
        mixed = mask_gates[:, :1] * own + mask_gates[:, 1:] * pyramid_support
        fused = self.mask_fuse(torch.cat((mask_hidden, mixed, depth_hidden), dim=1))
        mask_residual = self.max_residual * torch.tanh(self.mask_out(fused)).to(mask_features.dtype)
        return fused_features, mask_features + mask_residual
