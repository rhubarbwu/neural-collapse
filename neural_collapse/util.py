from hashlib import sha256

import torch as pt
import torch.linalg as la
from torch import Tensor

hashify = lambda O: sha256(O.cpu().numpy().tobytes()).hexdigest()
resolve = lambda A, B: B if not A or A == B else None
normalize = lambda x: x / (la.norm(x, dim=-1, keepdim=True) + pt.finfo(x.dtype).eps)


def patching(data: Tensor, kernel: callable, patch_size: int = 1) -> Tensor:
    N = len(data)
    outgrid = pt.zeros((N, N), device=data.device)
    n_patches = (N + patch_size - 1) // patch_size

    for i in range(n_patches):
        i0, i1 = i * patch_size, min((i + 1) * patch_size, N)
        patch_i = data[i0:i1]
        for j in range(n_patches):
            j0, j1 = j * patch_size, min((j + 1) * patch_size, N)
            patch_j = data[j0:j1]
            outgrid[i0:i1, j0:j1] = kernel(patch_i, patch_j)

    return outgrid


def inner_product(data: Tensor, patch_size: int = None) -> Tensor:
    if not patch_size:
        return pt.inner(data, data)

    kernel_grid = patching(data, pt.inner, patch_size)
    return kernel_grid
