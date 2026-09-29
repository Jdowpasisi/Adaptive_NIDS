import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA); optionally force deterministic cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required by some CUDA ops once deterministic algorithms are requested.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
