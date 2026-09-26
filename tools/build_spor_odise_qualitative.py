#!/usr/bin/env python3
"""Select and render representative COCO qualitative comparisons.

The script first selects structurally difficult COCO validation images from
ground-truth panoptic annotations, runs ODISE and SPOR on the same candidates,
scores per-image improvements, and exports six matched RGB/prediction pairs.
SPOR receives the aligned monocular depth map through the configured dataset
mapper, exactly as in evaluation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog, detection_utils as utils
from detectron2.data import get_detection_dataset_dicts
from detectron2.engine import create_ddp_model
from detectron2.evaluation import inference_context
from detectron2.utils.visualizer import ColorMode, Visualizer
from panopticapi.utils import rgb2id
from torch import nn

from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.engine.defaults import get_model_from_module


def candidate_score(record: dict, thing_ids: set[int]) -> float:
    # ODISE's registered panoptic records do not retain height/width, but
    # panoptic segments partition the image, so their areas recover it.
    image_area = float(sum(segment["area"] for segment in record["segments_info"]))
    things = [s for s in record["segments_info"] if s["category_id"] in thing_ids]
    counts = Counter(s["category_id"] for s in things)
    repeated = sum(max(0, count - 1) for count in counts.values())
    small = sum(120 <= s["area"] <= image_area * 0.018 for s in things)
    tiny = sum(40 <= s["area"] < image_area * 0.004 for s in things)
    return 3.0 * repeated + 2.5 * small + 1.0 * tiny + 0.25 * len(things)


def choose_candidates(records: list[dict], metadata, count: int) -> list[dict]:
    thing_ids = set(metadata.thing_dataset_id_to_contiguous_id.values())
    ranked = sorted(records, key=lambda r: candidate_score(r, thing_ids), reverse=True)
    selected = []
    category_use = Counter()
    for record in ranked:
        thing_categories = [
            s["category_id"] for s in record["segments_info"] if s["category_id"] in thing_ids
        ]
        if len(thing_categories) < 3:
            continue
        novelty = sum(category_use[c] == 0 for c in set(thing_categories))
        if selected and novelty == 0 and max(category_use[c] for c in set(thing_categories)) >= 5:
            continue
        selected.append(record)
        category_use.update(set(thing_categories))
        if len(selected) == count:
            break
    if len(selected) < count:
        selected.extend(r for r in ranked if r not in selected)[: count - len(selected)]
    return selected


def id_to_category_map(id_map: np.ndarray, segments: list[dict]) -> np.ndarray:
    category_map = np.full(id_map.shape, -1, dtype=np.int32)
    for segment in segments:
        category_map[id_map == segment["id"]] = int(segment["category_id"])
    return category_map


def semantic_miou(gt: np.ndarray, pred: np.ndarray) -> float:
    values = []
    for category in np.unique(gt[gt >= 0]):
        gt_mask = gt == category
        pred_mask = pred == category
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union:
            values.append(np.logical_and(gt_mask, pred_mask).sum() / union)
    return float(np.mean(values)) if values else 0.0


def instance_quality(
    gt_ids: np.ndarray,
    gt_segments: list[dict],
    pred_ids: np.ndarray,
    pred_segments: list[dict],
    thing_ids: set[int],
) -> tuple[float, float]:
    pred_by_category: dict[int, list[int]] = {}
    for segment in pred_segments:
        if segment.get("isthing", False):
            pred_by_category.setdefault(int(segment["category_id"]), []).append(int(segment["id"]))
    all_scores = []
    small_scores = []
    image_area = float(gt_ids.size)
    for segment in gt_segments:
        category = int(segment["category_id"])
        if category not in thing_ids:
            continue
        gt_mask = gt_ids == int(segment["id"])
        best = 0.0
        for pred_id in pred_by_category.get(category, []):
            pred_mask = pred_ids == pred_id
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union:
                best = max(best, float(np.logical_and(gt_mask, pred_mask).sum() / union))
        all_scores.append(best)
        if float(segment["area"]) <= image_area * 0.018:
            small_scores.append(best)
    return (
        float(np.mean(all_scores)) if all_scores else 0.0,
        float(np.mean(small_scores)) if small_scores else 0.0,
    )


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance: int = 2) -> float:
    def edges(label: np.ndarray) -> np.ndarray:
        edge = np.zeros(label.shape, dtype=np.uint8)
        edge[:-1] |= label[:-1] != label[1:]
        edge[:, :-1] |= label[:, :-1] != label[:, 1:]
        return edge

    gt_edge = edges(gt)
    pred_edge = edges(pred)
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    gt_dilated = cv2.dilate(gt_edge, kernel)
    pred_dilated = cv2.dilate(pred_edge, kernel)
    precision = float((pred_edge & gt_dilated).sum()) / max(float(pred_edge.sum()), 1.0)
    recall = float((gt_edge & pred_dilated).sum()) / max(float(gt_edge.sum()), 1.0)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def build_inference(config_path: str, checkpoint: str):
    cfg = LazyConfig.load(config_path)
    cfg.model.overlap_threshold = 0
    cfg.model.clip_head.alpha = 0.35
    cfg.model.clip_head.beta = 0.65
    mapper = instantiate(cfg.dataloader.test.mapper)
    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(checkpoint)
    wrapper_cfg = cfg.dataloader.wrapper
    while "model" in wrapper_cfg:
        wrapper_cfg = wrapper_cfg.model
    wrapper_cfg.model = get_model_from_module(model)
    inference_model = create_ddp_model(instantiate(cfg.dataloader.wrapper))
    return mapper, inference_model


def run_model(
    name: str,
    config_path: str,
    checkpoint: str,
    records: list[dict],
    metadata,
    cache_root: Path,
) -> dict[str, dict]:
    mapper, model = build_inference(config_path, checkpoint)
    output = {}
    model_dir = cache_root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        for index, record in enumerate(records, 1):
            image_id = Path(record["file_name"]).stem
            mapped = mapper(copy.deepcopy(record))
            prediction = model([mapped])[0]
            panoptic, segments = prediction["panoptic_seg"]
            panoptic = panoptic.detach().cpu().numpy().astype(np.int32)
            image = utils.read_image(record["file_name"], format="RGB")
            visualization = Visualizer(
                image,
                metadata,
                instance_mode=ColorMode.IMAGE,
            ).draw_panoptic_seg(torch.from_numpy(panoptic), segments)
            visualization.save(str(model_dir / f"{image_id}.png"))
            np.savez_compressed(
                model_dir / f"{image_id}.npz",
                panoptic=panoptic,
                segments=np.array(segments, dtype=object),
            )
            output[image_id] = {"panoptic": panoptic, "segments": segments}
            print(f"[{name}] {index}/{len(records)} {image_id}", flush=True)
    del model, mapper
    gc.collect()
    torch.cuda.empty_cache()
    return output


def compute_metrics(record: dict, prediction: dict, metadata) -> dict[str, float]:
    gt_ids = rgb2id(cv2.cvtColor(cv2.imread(record["pan_seg_file_name"]), cv2.COLOR_BGR2RGB))
    pred_ids = prediction["panoptic"]
    gt_categories = id_to_category_map(gt_ids, record["segments_info"])
    pred_categories = id_to_category_map(pred_ids, prediction["segments"])
    thing_ids = set(metadata.thing_dataset_id_to_contiguous_id.values())
    instance, small = instance_quality(
        gt_ids,
        record["segments_info"],
        pred_ids,
        prediction["segments"],
        thing_ids,
    )
    return {
        "miou": semantic_miou(gt_categories, pred_categories),
        "instance": instance,
        "small": small,
        "boundary": boundary_f1(gt_categories, pred_categories),
    }


def pick_final(scored: list[dict], count: int) -> list[dict]:
    # First cover the three requested behavior types, then fill by aggregate gain.
    selected = []
    used = set()
    for metric in ("small_gain", "instance_gain", "boundary_gain"):
        for item in sorted(scored, key=lambda x: x[metric], reverse=True):
            if item["image_id"] not in used and item[metric] > 0:
                selected.append(item)
                used.add(item["image_id"])
                break
    for item in sorted(scored, key=lambda x: x["score"], reverse=True):
        if item["image_id"] not in used:
            selected.append(item)
            used.add(item["image_id"])
        if len(selected) == count:
            break
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spor-config", default="configs/Panoptic/odise_label_coco_50e_spor.py")
    parser.add_argument("--spor-checkpoint", required=True)
    parser.add_argument("--odise-config", default="configs/Panoptic/odise_label_coco_50e.py")
    parser.add_argument("--odise-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--final-count", type=int, default=6)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    spor_dir = output_root / "SPOR"
    odise_dir = output_root / "ODISE"
    cache_root = output_root / "candidate_cache"
    spor_dir.mkdir(parents=True, exist_ok=True)
    odise_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Loading the LazyConfig imports ODISE's custom COCO registration module.
    # This must precede DatasetCatalog/MetadataCatalog access in a standalone
    # script (the training launcher normally performs this import for us).
    dataset_config = LazyConfig.load(args.spor_config)
    dataset_name = str(dataset_config.dataloader.test.dataset.names)
    metadata = MetadataCatalog.get(dataset_name)
    records = get_detection_dataset_dicts(dataset_name, filter_empty=False)
    candidates = choose_candidates(records, metadata, args.candidate_count)
    (cache_root / "candidate_ids.json").write_text(
        json.dumps([Path(r["file_name"]).stem for r in candidates], indent=2), encoding="utf-8"
    )

    odise_predictions = run_model(
        "ODISE", args.odise_config, args.odise_checkpoint, candidates, metadata, cache_root
    )
    spor_predictions = run_model(
        "SPOR", args.spor_config, args.spor_checkpoint, candidates, metadata, cache_root
    )

    scored = []
    record_by_id = {Path(r["file_name"]).stem: r for r in candidates}
    for image_id, record in record_by_id.items():
        odise = compute_metrics(record, odise_predictions[image_id], metadata)
        spor = compute_metrics(record, spor_predictions[image_id], metadata)
        gains = {f"{key}_gain": spor[key] - odise[key] for key in odise}
        score = (
            3.0 * gains["miou_gain"]
            + 2.0 * gains["small_gain"]
            + 2.0 * gains["instance_gain"]
            + gains["boundary_gain"]
        )
        scored.append(
            {
                "image_id": image_id,
                "score": score,
                "odise": odise,
                "spor": spor,
                **gains,
            }
        )

    selected = pick_final(scored, args.final_count)
    for order, item in enumerate(selected, 1):
        image_id = item["image_id"]
        record = record_by_id[image_id]
        prefix = f"{order:02d}_{image_id}"
        for destination in (spor_dir, odise_dir):
            shutil.copy2(record["file_name"], destination / f"{prefix}_rgb.jpg")
        shutil.copy2(cache_root / "SPOR" / f"{image_id}.png", spor_dir / f"{prefix}_seg.png")
        shutil.copy2(cache_root / "ODISE" / f"{image_id}.png", odise_dir / f"{prefix}_seg.png")

    manifest = {
        "selection_method": "per-image GT gains: semantic IoU, small-object matching, instance matching, boundary F1",
        "spor_checkpoint": args.spor_checkpoint,
        "odise_checkpoint": args.odise_checkpoint,
        "selected": selected,
        "all_candidates": sorted(scored, key=lambda x: x["score"], reverse=True),
    }
    (output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
