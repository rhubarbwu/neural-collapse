from math import copysign
from typing import Tuple

import torch as pt
from torch import Tensor

from .util import normalize, symm_reduce, tiling


def class_dist_norm_vars(
    V_norms: Tensor,
    M: Tensor,
    dist_exp: float = 1.0,
    tile_size: int = None,
) -> Tensor:
    """Compute the matrix grid of class-distance normalized variances (CDNV).
    This metric reflects pairwise variability adjusted for mean distances.
    Galanti et al. (2021): https://arxiv.org/abs/2112.15121

    Arguments:
        V_norms (Tensor): Matrix of within-class variance norms.
        M (Tensor): Matrix of feature (or class mean) embeddings.
        dist_exp (int): Power with which to exponentiate the distance
            normalizer. A greater power further diminishes the contribution of
            mutual variability between already-disparate classes. Defaults to
            1, equivalent to the CDNV introduced by Galanti et al. (2021).
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.

    Returns:
        Tensor: A tensor representing the matrix grid of pairwise CDNVs.
    """
    V_norms = V_norms.view(-1, 1)
    bundled = pt.cat((M, V_norms), dim=1)

    def kernel(tile_i, tile_j):
        vars_i, vars_j = tile_i[:, -1], tile_j[:, -1]
        var_avgs = (vars_i.unsqueeze(1) + vars_j).squeeze() / 2

        M_i, M_j = tile_i[:, :-1], tile_j[:, :-1]

        M_diff = M_i.unsqueeze(1) - M_j
        M_diff_norm_sq = pt.sum(M_diff * M_diff, dim=-1)
        return var_avgs.squeeze(0) / (M_diff_norm_sq**dist_exp)

    return tiling(bundled, kernel, tile_size)


def dist_kernel(data: Tensor, tile_size: int = None) -> Tensor:
    """Compute the grid of pairwise vector distances across a set of vectors.

    Arguments:
        data (Tensor): Input data tensor across which to apply the kernel.
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.
    """
    kernel = lambda tile_i, tile_j: (tile_i.unsqueeze(1) - tile_j).norm(dim=-1)
    return tiling(data, kernel, tile_size)


def log_kernel(data: Tensor, exponent: int = -1, tile_size: int = None) -> Tensor:
    """Compute the grid of pairwise logarithmic  distances across vectors.
    Liu et al. (2023): https://arxiv.org/abs/2303.06484

    Arguments:
        data (Tensor): Input data tensor across which to apply the kernel.
        exponent (int, optional): Power with which to exponentiate the
            distance norm before the logarithm. Defaults to -1 (inverse).
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.
    """

    def kernel(tile_i, tile_j):
        diff = tile_i.unsqueeze(1) - tile_j
        diff_norms = diff.norm(dim=-1)
        return (diff_norms ** (exponent)).log()

    return tiling(data, kernel, tile_size)


def riesz_kernel(data: Tensor, tile_size: int = None) -> Tensor:
    """Compute the grid of Riesz distances across vectors.
    Liu et al. (2023): https://arxiv.org/abs/2303.06484

    Arguments:
        data (Tensor): Input data tensor across which to apply the kernel.
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.
    """
    S = data.shape[-1] - 2

    def kernel(tile_i, tile_j):
        diff = tile_i.unsqueeze(1) - tile_j
        diff_norms = diff.norm(dim=-1)
        return copysign(1, S) * diff_norms ** (-S)

    return tiling(data, kernel, tile_size)


def kernel_grid(
    M: Tensor,
    m_G: Tensor = 0,
    kernel: callable = dist_kernel,
    tile_size: int = None,
) -> Tensor:
    """Compute the grid from the kernel function on pairwise interactions
    between embeddings. Self-interactions are excluded.

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.
        kernel (callable, optional): The kernel with which to compute
            interactions. Defaults to the inner product. Other common
            functions include the logarithmic or Riesz distance kernels.
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.

    Returns:
        float: Average of pairwise kernel interactions.
        float: Variance of pairwise kernel interactions.
    """
    M_centred_normed = normalize(M - m_G)
    return kernel(M_centred_normed, tile_size=tile_size)


def kernel_stats(
    M: Tensor,
    m_G: Tensor = 0,
    kernel: callable = dist_kernel,
    tile_size: int = None,
) -> Tuple[float, float]:
    """Compute the average and variance of a kernel function on pairwise
    interactions between embeddings. Self-interactions are excluded.
    Liu et al. (2023): https://arxiv.org/abs/2303.06484

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.
        kernel (callable): Kernel function with which to compute
            interactions. Defaults to the inner product. Other common
            functions include the logarithmic or Riesz distance kernels.
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.

    Returns:
        float: Average of pairwise kernel interactions.
        float: Variance of pairwise kernel interactions.
    """
    grid: Tensor = kernel_grid(M, m_G, kernel, tile_size)
    avg = symm_reduce(grid)
    var = symm_reduce(grid, lambda row: pt.sum((row - avg) ** 2))

    return avg.item(), var.item()
