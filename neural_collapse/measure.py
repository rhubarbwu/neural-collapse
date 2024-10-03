from math import sqrt
from typing import Tuple

import torch as pt
import torch.linalg as la
from torch import Tensor

from .util import inner_product, normalize, patching, symm_reduce


def class_dist_norm_var(
    M: Tensor,
    V: Tensor,
    patch_size: int = 1,
) -> Tensor:
    """Kernel grid of class-distance normalized variance (CDNV).
    Galanti et al. (2021): https://arxiv.org/abs/2112.15121
    """
    V = V.view(-1, 1)
    bundled = pt.cat((M, V), dim=1)

    def kernel(patch_i, patch_j):
        vars_i, vars_j = patch_i[:, -1], patch_j[:, -1]
        var_avgs = (vars_i.unsqueeze(1) + vars_j).squeeze() / 2

        M_i, M_j = patch_i[:, :-1], patch_j[:, :-1]

        M_diff = M_i.unsqueeze(1) - M_j
        inner = pt.sum(M_diff * M_diff, dim=-1)
        return var_avgs.squeeze(0) / inner

    return patching(bundled, kernel, patch_size)


def variability_cdnv(M: Tensor, V: Tensor, patch_size: int = 1) -> float:
    """Average class-distance normalized variance (CDNV).
    Galanti et al. (2021): https://arxiv.org/abs/2112.15121
    """
    kernel_grid = class_dist_norm_var(M, V, patch_size)
    avg = symm_reduce(kernel_grid, pt.sum)
    return avg


def variability_ratio(M: Tensor, V_intra: Tensor, m_G: Tensor = 0) -> float:
    """Ratio of traces of (co)variances.
    Hui et al. (2022): https://arxiv.org/abs/2202.08384
    Tirer et al. (2023): https://proceedings.mlr.press/v202/tirer23a
    """
    (_, D), M_centred = M.shape, M - m_G
    V_inter = M_centred.mT @ M_centred / D  # (D,D)
    return pt.trace(V_intra) / pt.trace(V_inter)  # (1)


def variability_pinv(M: Tensor, V_intra: Tensor, m_G: Tensor = 0) -> float:
    """Within-class variability collapse.
    Papyan, Han, & Donoho (2020): https://www.pnas.org/doi/full/10.1073/pnas.2015509117
    """
    (K, D), M_centred = M.shape, M - m_G
    V_inter = M_centred.mT @ M_centred / D  # (D,D)
    ratio_prod = la.pinv(V_inter) @ V_intra.to(V_inter.device)  # (D,D)

    return pt.trace(ratio_prod).item() / K  # (1)


def mean_norms(M: Tensor, m_G: Tensor = 0):
    M_centred = M - m_G
    return M_centred.norm(dim=-1) ** 2  # (K)


def interference(M: Tensor, m_G: Tensor = 0, patch_size: int = None) -> Tensor:
    M_centred = M - m_G
    return inner_product(normalize(M_centred), patch_size)  # (K,K)


def _structure_error(A: Tensor, B: Tensor) -> Tensor:
    (K, _) = A.shape
    ideal = (pt.eye(K) - pt.ones(K, K) / K) / sqrt(K - 1)  # (K,K)

    outer = B.to(A.device) @ A.mT  # (K,K)
    outer /= la.matrix_norm(outer)

    return la.matrix_norm(outer - ideal.to(outer.device))


def simplex_etf_error(M: Tensor, m_G: Tensor = 0) -> float:
    M_centred = M - m_G
    return _structure_error(M_centred, M_centred).item()


def self_duality_error(W: Tensor, M: Tensor, m_G: Tensor = 0) -> float:
    M_centred = M - m_G
    return _structure_error(M_centred, W).item()


def clf_ncc_agreement(Ns: Tensor, hits: Tensor = None, misses: Tensor = None) -> Tensor:
    if hits is None and misses and misses.shape == Ns.shape:
        hits = Ns - misses

    if hits and hits.shape == Ns.shape:
        return (hits / Ns).mean()

    return None
