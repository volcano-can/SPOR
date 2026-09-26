# SPOR

SPOR extends [ODISE](https://github.com/NVlabs/ODISE) with ordinal depth cues for open-vocabulary panoptic segmentation. This repository contains the implementation and experiment configurations.

## Components

- **ODFA** (Ordinal Depth Feature Aggregation) injects ordinal geometry into segmentation features.
- **QOBR** (Query-conditioned Ordinal Boundary Routing) uses depth order to guide decoder cross-attention.
- **ROPC** adds an ordinal consistency loss for predicted masks.

The main configuration is [`configs/Panoptic/odise_label_coco_50e_spor.py`](configs/Panoptic/odise_label_coco_50e_spor.py). Core modules are in [`odise/modeling/geometry/`](odise/modeling/geometry/) and [`odise/engine/spor.py`](odise/engine/spor.py).

## Training

Follow [`GETTING_STARTED.md`](GETTING_STARTED.md) for the ODISE environment. Provide COCO panoptic data, aligned depth maps, and an ODISE initialization checkpoint locally. The dataset root is set with `DETECTRON2_DATASETS`; the SPOR config expects depth maps under `coco/depth_train2017` and `coco/depth_val2017` within that root.

```bash
python tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e_spor.py --num-gpus 8 --init-from /path/to/odise_checkpoint.pth
```

Checkpoints, datasets, and generated outputs are not included in this repository.

## Attribution

SPOR builds on ODISE and its bundled Mask2Former code. See the included license files and upstream repositories for their terms and attribution.
