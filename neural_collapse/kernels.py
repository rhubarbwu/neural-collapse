from math import copysign

import torch as pt
from torch import Tensor

from .util import normalize, tiling


def class_dist_norm_var(
    V: Tensor,
    M: Tensor,
    tile_size: int = None,
) -> Tensor:
    """Compute the matrix grid of class-distance normalized variances (CDNV).
    This metric reflects pairwise variability adjusted for mean distances.
    Galanti et al. (2021): https://arxiv.org/abs/2112.15121

    Arguments:
        V (Tensor): The matrix of within-class variances for the classes.
        M (Tensor): The matrix of feature (or class mean) embeddings.
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.

    Returns:
        Tensor: A tensor representing the matrix grid of pairwise CDNVs.
    """
    V = V.view(-1, 1)
    bundled = pt.cat((M, V), dim=1)

    def kernel(tile_i, tile_j):
        vars_i, vars_j = tile_i[:, -1], tile_j[:, -1]
        var_avgs = (vars_i.unsqueeze(1) + vars_j).squeeze() / 2

        M_i, M_j = tile_i[:, :-1], tile_j[:, :-1]

        M_diff = M_i.unsqueeze(1) - M_j
        inner = pt.sum(M_diff * M_diff, dim=-1)
        return var_avgs.squeeze(0) / inner

    grid = tiling(bundled, kernel, tile_size)
    return grid


def log_kernel(data: Tensor, exponent: int = -1, tile_size: int = 1) -> Tensor:
    """Compute the matrix grid of logarithmic distances across vectors.
    Liu et al. (2023): https://arxiv.org/abs/2303.06484

    Arguments:
        data (Tensor): The input data tensor to be processed.
        exponent (int, optional): The exponent to apply to the distance norm
            before taking the logarithm. Defaults to -1 (inverse).
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.
    """
    normed = normalize(data)

    def kernel(patch_i, patch_j):
        diff = patch_i.unsqueeze(1) - patch_j
        diff_norms = diff.norm(dim=-1)
        return (diff_norms ** (exponent)).log()

    grid = tiling(normed, kernel, tile_size)
    return grid


def riesz_kernel(data: Tensor, tile_size: int = 1) -> Tensor:
    """Compute the matrix grid of Riesz distances across vectors.
    Liu et al. (2023): https://arxiv.org/abs/2303.06484

    Arguments:
        data (Tensor): The input data tensor to be processed.
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.
    """

    S = data.shape[-1] - 2
    normed = normalize(data)

    def kernel(patch_i, patch_j):
        diff = patch_i.unsqueeze(1) - patch_j
        diff_norms = diff.norm(dim=-1)
        return copysign(1, S) * diff_norms ** (-S)

    grid = tiling(normed, kernel, tile_size)
    return grid
