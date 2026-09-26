#!/usr/bin/env python3
"""Create faithful depth -> ordinal-rank visualizations for ODISE.

This follows the geometry path in ``odise/data/dataset_mapper.py``:
the same sampled transform is applied to RGB, depth, and a source-validity
map.  Ordinal ranks then match
``OrdinalDepthFeatureAggregator._ordinal_rank`` (tie-aware mid-ranks over
valid finite pixels, normalized to [0, 1]).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ordinal_rank(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """NumPy equivalent of the repository's tie-aware PyTorch rank code."""
    depth = depth.astype(np.float32, copy=False)
    valid = valid.astype(bool, copy=False) & np.isfinite(depth)
    rank = np.zeros(depth.shape, dtype=np.float32)
    values = depth[valid]
    if values.size <= 1:
        return rank
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    starts = np.cumsum(np.r_[0, counts[:-1]], dtype=np.int64)
    mids = starts + (counts.astype(np.float64) - 1.0) * 0.5
    rank[valid] = (mids[inverse] / float(values.size - 1)).astype(np.float32)
    return rank


def resize_depth_to_rgb(depth: np.ndarray, rgb_shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == rgb_shape:
        return depth
    return cv2.resize(depth, (rgb_shape[1], rgb_shape[0]), interpolation=cv2.INTER_LINEAR)


def mapper_train_transform(
    rgb: np.ndarray,
    depth: np.ndarray,
    seed: int,
    target: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Apply the exact Detectron2 augmentations configured for ODISE training."""
    from detectron2.data import transforms as T

    depth = resize_depth_to_rgb(depth, rgb.shape[:2])
    source_valid = np.ones(depth.shape, dtype=np.uint8)
    augmentations = T.AugmentationList(
        [
            T.RandomFlip(horizontal=True),
            T.ResizeScale(min_scale=0.1, max_scale=2.0, target_height=target, target_width=target),
            T.FixedSizeCrop(crop_size=(target, target)),
        ]
    )
    # Detectron2 draws augmentation parameters through NumPy's RNG.
    np.random.seed(seed)
    aug_input = T.AugInput(rgb.copy())
    transforms = augmentations(aug_input)
    transformed_rgb = aug_input.image
    transformed_depth = transforms.apply_image(depth)
    transformed_valid = transforms.apply_segmentation(source_valid) == 1
    return transformed_rgb, transformed_depth, transformed_valid, str(transforms)


def choose_seed_with_padding(
    rgb: np.ndarray,
    depth: np.ndarray,
    first_seed: int,
    min_valid_fraction: float = 0.55,
    max_valid_fraction: float = 0.92,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, str]:
    """Choose a reproducible real augmentation whose validity is informative."""
    fallback = None
    for seed in range(first_seed, first_seed + 1000):
        result = mapper_train_transform(rgb, depth, seed)
        fraction = float(result[2].mean())
        fallback = (seed, *result)
        if min_valid_fraction <= fraction <= max_valid_fraction:
            return fallback
    assert fallback is not None
    return fallback


def colorize(values: np.ndarray, cmap: str, valid: np.ndarray | None = None) -> np.ndarray:
    rgba = plt.get_cmap(cmap)(np.clip(values, 0.0, 1.0), bytes=True)
    rgb = rgba[..., :3]
    if valid is not None:
        rgb = rgb.copy()
        rgb[~valid] = 0
    return rgb


def save_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def panel(ax, image, title: str, cmap=None, vmin=None, vmax=None) -> None:
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def build_sample(
    sample_id: str,
    rgb_path: Path,
    depth_path: Path,
    output_dir: Path,
    seed: int,
) -> dict:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    raw_depth = np.asarray(Image.open(depth_path).convert("L"))
    raw_depth = resize_depth_to_rgb(raw_depth, rgb.shape[:2])
    source_valid = np.ones(raw_depth.shape, dtype=bool)
    source_rank = ordinal_rank(raw_depth, source_valid)

    used_seed, aug_rgb, aug_depth, valid, transform_text = choose_seed_with_padding(
        rgb, raw_depth, seed
    )
    rank = ordinal_rank(aug_depth, valid)
    valid_fraction = float(valid.mean())

    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image(sample_dir / "01_rgb_source.png", rgb)
    save_image(sample_dir / "02_depth_source_gray.png", raw_depth.astype(np.uint8))
    save_image(sample_dir / "03_depth_source_color.png", colorize(raw_depth / 255.0, "turbo"))
    save_image(sample_dir / "04_source_rank_color.png", colorize(source_rank, "viridis"))
    save_image(sample_dir / "05_source_validity.png", source_valid.astype(np.uint8) * 255)
    save_image(sample_dir / "06_rgb_after_mapper.png", aug_rgb.astype(np.uint8))
    save_image(
        sample_dir / "07_depth_after_mapper_color.png",
        colorize(aug_depth / 255.0, "turbo", valid),
    )
    save_image(sample_dir / "08_ordinal_rank_color.png", colorize(rank, "viridis", valid))
    save_image(sample_dir / "09_validity.png", valid.astype(np.uint8) * 255)
    # Lossless numeric rank: 0..65535; invalid pixels remain zero and must be
    # interpreted together with 09_validity.png.
    save_image(sample_dir / "10_ordinal_rank_u16.png", np.round(rank * 65535.0).astype(np.uint16))
    # Display version: black is low rank.  Invalid pixels are also black, so
    # use this image together with 09_validity.png to distinguish them.
    save_image(
        sample_dir / "11_ordinal_rank_gray.png",
        np.where(valid, np.round(rank * 255.0), 0).astype(np.uint8),
    )
    np.save(sample_dir / "ordinal_rank_float32.npy", rank)
    np.save(sample_dir / "validity_bool.npy", valid)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    panel(axes[0, 0], rgb, "Source RGB")
    panel(axes[0, 1], raw_depth, "Source depth (8-bit)", "turbo", 0, 255)
    panel(axes[0, 2], source_rank, "Source ordinal rank", "viridis", 0, 1)
    panel(axes[0, 3], source_valid, "Source validity (all valid)", "gray", 0, 1)
    panel(axes[1, 0], aug_rgb, "RGB after mapper geometry")
    panel(axes[1, 1], np.ma.masked_where(~valid, aug_depth), "Depth after mapper", "turbo", 0, 255)
    panel(axes[1, 2], np.ma.masked_where(~valid, rank), "Valid-aware ordinal rank", "viridis", 0, 1)
    panel(axes[1, 3], valid, f"Validity ({valid_fraction:.1%} valid)", "gray", 0, 1)
    fig.suptitle(
        f"ODISE depth → ordinal rank | COCO {sample_id} | augmentation seed {used_seed}",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(sample_dir / "overview.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "sample_id": sample_id,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "augmentation_seed": used_seed,
        "transform": transform_text,
        "source_shape_hw": list(raw_depth.shape),
        "model_input_shape_hw": list(aug_depth.shape),
        "source_valid_fraction": 1.0,
        "model_input_valid_fraction": valid_fraction,
        "rank_definition": "tie-aware mid-rank over valid finite pixels, divided by N_valid-1",
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def score_candidates(rgb_dir: Path, depth_dir: Path, limit: int = 5000) -> list[tuple[float, str]]:
    """Rank candidates by depth entropy, spatial variation, and usable range."""
    scores = []
    for index, depth_path in enumerate(sorted(depth_dir.glob("*.png"))):
        if index >= limit:
            break
        sample_id = depth_path.stem
        if not (rgb_dir / f"{sample_id}.jpg").is_file():
            continue
        d = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if d is None:
            continue
        thumb = cv2.resize(d, (128, 128), interpolation=cv2.INTER_AREA)
        hist = np.bincount(thumb.ravel(), minlength=256).astype(np.float64)
        p = hist[hist > 0] / hist.sum()
        entropy = float(-(p * np.log2(p)).sum() / 8.0)
        gx = cv2.Sobel(thumb, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(thumb, cv2.CV_32F, 0, 1, ksize=3)
        edge = float(np.mean(np.sqrt(gx * gx + gy * gy)) / 255.0)
        spread = float((np.percentile(thumb, 95) - np.percentile(thumb, 5)) / 255.0)
        scores.append((0.50 * entropy + 0.30 * min(edge, 1.0) + 0.20 * spread, sample_id))
    return sorted(scores, reverse=True)


def candidate_sheet(ids: list[str], rgb_dir: Path, depth_dir: Path, path: Path) -> None:
    fig, axes = plt.subplots(len(ids), 2, figsize=(10, 3.2 * len(ids)), constrained_layout=True)
    if len(ids) == 1:
        axes = np.asarray([axes])
    for row, sample_id in enumerate(ids):
        rgb = np.asarray(Image.open(rgb_dir / f"{sample_id}.jpg").convert("RGB"))
        depth = np.asarray(Image.open(depth_dir / f"{sample_id}.png").convert("L"))
        panel(axes[row, 0], rgb, f"{sample_id} | RGB")
        panel(axes[row, 1], depth, "Depth", "turbo", 0, 255)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/chenduoyou/data/Datasets/coco"))
    parser.add_argument("--split", choices=("train2017", "val2017"), default="val2017")
    parser.add_argument("--ids", nargs="*", help="COCO ids; if omitted, score candidates automatically")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-only", action="store_true")
    args = parser.parse_args()

    rgb_dir = args.dataset_root / args.split
    depth_dir = args.dataset_root / ("depth_" + args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    scored = score_candidates(rgb_dir, depth_dir)
    candidate_ids = [sample_id for _, sample_id in scored[: args.candidate_count]]
    candidate_sheet(candidate_ids, rgb_dir, depth_dir, args.output / "candidate_sheet.png")
    (args.output / "candidate_scores.json").write_text(
        json.dumps([{"id": i, "score": s} for s, i in scored[: args.candidate_count]], indent=2),
        encoding="utf-8",
    )
    if args.candidates_only:
        return

    selected = args.ids if args.ids else candidate_ids[: args.num_samples]
    metadata = []
    for offset, sample_id in enumerate(selected):
        metadata.append(
            build_sample(
                sample_id,
                rgb_dir / f"{sample_id}.jpg",
                depth_dir / f"{sample_id}.png",
                args.output,
                args.seed + offset * 1000,
            )
        )
    (args.output / "selection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
