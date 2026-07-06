"""base.utils — generic helpers shared across the base pillar.

  * ``seed_all``: global RNG seeding for reproducibility.
"""

import os
import random

import numpy as np
import torch


def seed_all(
    seed: int,
    deterministic: bool = False,
    deterministic_cudnn: bool = False,
    warn: bool = False,
) -> None:
    """Set seeds for ALL random number generators globally.

    The CENTRAL seeding function — call once at the start of a run.

    Args:
        seed: The seed value to use.
        deterministic: If True, enables strict deterministic operations
            (``torch.use_deterministic_algorithms(True)`` + CUBLAS workspace
            config). Slower, but exactly reproducible. False for production.
        deterministic_cudnn: If True AND ``deterministic``, sets cuDNN to
            deterministic mode. Ignored if ``deterministic`` is False.
        warn: If True, print the cuDNN determinism caveat.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Deterministic CUDA matmul needs the CUBLAS workspace config.
        if torch.cuda.is_available():
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=False)
        print('ensuring determinism in the torch algorithm')
        if deterministic_cudnn and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if warn:
                print(
                    "Warning: This setting is not reproducible when creating "
                    "2 models from scratch, but it is when loading pretrained models."
                )
    else:
        # Non-deterministic mode — faster for production.
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # cuDNN autotuner


__all__ = ['seed_all']
