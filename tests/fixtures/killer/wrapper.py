"""Synthetic killer fixture: depends on nvdiffrast (NVIDIA-only)."""
import nvdiffrast.torch as dr  # noqa: F401  (no ROCm equivalent)

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = open("killer_kernel.cu").read()


def build():
    return load_inline(name="killer", cuda_sources=[CUDA_SRC],
                       functions=["warp_reduce"])
