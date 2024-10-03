from hashlib import sha256

import torch as pt
import torch.linalg as la
from torch import Tensor

hashify = lambda O: sha256(O.cpu().numpy().tobytes()).hexdigest()
resolve = lambda A, B: B if not A or A == B else None
normalize = lambda x: x / (la.norm(x, dim=-1, keepdim=True) + pt.finfo(x.dtype).eps)
reduce_std = lambda mu: lambda x: ((x - mu) ** 2).sum()


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


def symm_reduce(data: Tensor, reduce:callable=pt.sum) -> float:
    N = data.shape[0]
    total = 0

    assert N == data.shape[1]
    for i in range((N - 1) // 2):
        upper = data[i][i + 1 :]
        lower = data[N - i - 2][N - i - 1 :]
        folded = pt.cat((upper, lower))
        total += reduce(folded)
    if N % 2 == 0:
        row = data[N // 2 - 1][N // 2 :]
        total += reduce(row)

    mean = total / (N * (N - 1) / 2)
    return mean.item()
