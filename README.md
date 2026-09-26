# SPOR

SPOR introduces ordinal depth cues into open-vocabulary panoptic segmentation. It combines feature aggregation, query-level boundary routing, and mask consistency to use geometric structure during prediction.

## Components

- **ODFA — Ordinal Depth Feature Aggregation:** injects depth ordering into segmentation features.
- **QOBR — Query-conditioned Ordinal Boundary Routing:** guides decoder cross-attention with query-specific depth boundaries.
- **ROPC:** encourages ordinal consistency in predicted masks.

This repository contains the SPOR implementation and experiment configurations.
