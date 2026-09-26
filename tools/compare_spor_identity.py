#!/usr/bin/env python3
"""Compare COCO AP/mIoU/PQ between official ODISE and SPOR iteration zero."""

import argparse
import json
import re
from pathlib import Path


TASK_TO_METRIC = {
    "coco_2017_val_panoptic_with_sem_seg/segm": "AP",
    "coco_2017_val_panoptic_with_sem_seg/sem_seg": "mIoU",
    "coco_2017_val_panoptic_with_sem_seg/panoptic_seg": "PQ",
}


def parse_metrics(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    found = {}
    for index, line in enumerate(lines):
        match = re.search(r"copypaste: Task: (.+)$", line)
        if not match or match.group(1) not in TASK_TO_METRIC:
            continue
        if index + 2 >= len(lines):
            continue
        values = lines[index + 2].split("copypaste:", 1)[-1].strip()
        found[TASK_TO_METRIC[match.group(1)]] = float(values.split(",", 1)[0])
    missing = set(TASK_TO_METRIC.values()) - set(found)
    if missing:
        raise RuntimeError(f"Missing metrics {sorted(missing)} in {path}")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spor", required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = parse_metrics(args.baseline)
    spor = parse_metrics(args.spor)
    delta = {name: spor[name] - baseline[name] for name in baseline}
    passed = all(abs(value) < args.threshold for value in delta.values())
    report = {
        "baseline": baseline,
        "spor_0": spor,
        "delta": delta,
        "threshold": args.threshold,
        "passed": passed,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
