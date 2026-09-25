"""Pick the fastest available torch device."""

from __future__ import annotations


def default_device() -> str:
    """CUDA (Colab/Kaggle), then Apple Silicon GPU (MPS), then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
