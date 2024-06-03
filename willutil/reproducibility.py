"""
some utils for reproducibility
"""

import willutil


def try_to_be_deterministic():
    import random

    random.seed(0)

    try:
        import numpy

        numpy.random.seed(0)
        did_numpy = True
    except ImportError:
        did_numpy = False

    try:
        import torch
        import os

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        did_torch = True
    except ImportError:
        did_torch = False

    return willutil.Bunch(numpy=did_numpy, torch=did_torch, _strict=True)
