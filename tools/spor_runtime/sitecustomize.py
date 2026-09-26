"""Compatibility aliases required by the pinned Detectron2 evaluator."""

import numpy as np


if "int" not in np.__dict__:
    np.int = int
