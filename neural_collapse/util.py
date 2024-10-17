from hashlib import sha256

import torch as pt
import torch.linalg as la
from torch import Tensor

hashify = lambda O: sha256(O.cpu().numpy().tobytes()).hexdigest()
resolve = lambda A, B: B if not A or A == B else None
normalize = lambda x: x / (la.norm(x, dim=-1, keepdim=True) + pt.finfo(x.dtype).eps)


def tiling(data: Tensor, kernel: callable, tile_size: int = None) -> Tensor:
    """Create a grid of kernel evaluations based on tiles of the input data.

    The function divides the input tensor `data` into overlapping tiles of
    size `tile_size` and applies the specified kernel function to each pair
    of tiles. The results are stored in a 2D tensor that represents the
    evaluations of the kernel on each pair of tiles.

    Args:
        data (Tensor): Input tensor to be tiled.
        kernel (callable): Function that takes two tiles as input and
            returns a tensor representing their kernel evaluation.
        tile_size (int, optional): Size of the tiles to be extracted from the
            data. Set tile_size << K to avoid OOM. Defaults to None.

    Returns:
        Tensor: Matrix where the element at position (i, j) is the result
            of applying the kernel to the i-th and j-th tiles of the input.
    """
    N = len(data)
    outgrid = pt.zeros((N, N), device=data.device)
    if not tile_size:
        tile_size = N
    n_tiles = (N + tile_size - 1) // tile_size

    for i in range(n_tiles):
        i0, i1 = i * tile_size, min((i + 1) * tile_size, N)
        tile_i = data[i0:i1]
        for j in range(n_tiles):
            j0, j1 = j * tile_size, min((j + 1) * tile_size, N)
            tile_j = data[j0:j1]
            outgrid[i0:i1, j0:j1] = kernel(tile_i, tile_j)

    return outgrid


def symm_reduce(data: Tensor, reduce: callable = pt.sum) -> Tensor:
    """Compute a symmetric reduction of the upper triangle of a square tensor.

    This function computes a specified reduction the upper triangle of a
    square tensor `data`, ignoring the diagonal. It also handles the case of
    an even-sized tensor by including the middle row in the reduction.

    Args:
        data (Tensor): Square matrix from which to compute the reduction.
        reduce (callable, optional): A callable function to apply for the
            reduction. Defaults to `pt.sum`.

    Returns:
        Tensor: Mean of the reduction applied to the upper triangle of the
            symmetric tensor.
    """
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

    return total / (N * (N - 1) / 2)
