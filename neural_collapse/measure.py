from math import sqrt
from typing import List

import numpy as np
import torch as pt
import torch.linalg as la
from scipy.sparse.linalg import svds
from torch import Tensor
from torch.nn.functional import cosine_similarity

from .accumulate import DecAccumulator
from .kernels import class_dist_norm_var
from .util import normalize, symm_reduce


def covariance_pinv(
    M: Tensor, V_intra: Tensor, m_G: Tensor = 0, svd: bool = False
) -> float:
    """Compute within-class variability collapse: trace of the product
    between the within-class (intra) variance and pseudo-inverse of the
    between-class (inter) variance.
    Papyan et al. (2020): https://doi.org/10.1073/pnas.2015509117

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        V_intra (Tensor): Matrix of within-class (co)variance.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.
        svd (bool, optional): Whether to compute Moore-Penrose pseudo-inverse
            directly. Default is False, using torch.pinv.

    Returns:
        float: Product trace of between-class/within-class (co)variances.
    """
    (K, D), M_centred = M.shape, M - m_G
    V_inter = M_centred.mT @ M_centred / D  # (D,D)

    if svd:  # compute MP-inverse directly using SVD
        V_intra, V_inter = V_intra.cpu().numpy(), V_inter.cpu().numpy()
        eig_vecs, eig_vals, _ = svds(V_inter, k=K - 1)
        inv_Sb = eig_vecs @ np.diag(eig_vals ** (-1)) @ eig_vecs.T  # (D,D)
        return float(np.trace(V_intra @ inv_Sb))

    ratio_prod = la.pinv(V_inter) @ V_intra.to(V_inter.device)  # (D,D)
    return pt.trace(ratio_prod).item() / K  # (1)


def covariance_ratio(M: Tensor, V_intra: Tensor, m_G: Tensor = 0) -> float:
    """Compute the ratio of traces of (co)variances: within-class (intra)
    variance to between-class (inter) variance.
    Hui et al. (2022): https://arxiv.org/abs/2202.08384
    Tirer et al. (2023): https://proceedings.mlr.press/v202/tirer23a

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        V_intra (Tensor): Matrix of within-class (co)variance.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.

    Returns:
        float: Ratio of traces of between-class/within-class (co)variances.
    """
    (_, D), M_centred = M.shape, M - m_G
    V_inter = M_centred.mT @ M_centred / D  # (D,D)
    ratio = pt.trace(V_intra) / pt.trace(V_inter)
    return ratio.item()


def variability_cdnv(
    M: Tensor, V: Tensor, dist_exp: float = 1.0, tile_size: int = None
) -> float:
    """Compute the average class-distance normalized variances (CDNV).
    This metric reflects pairwise variability adjusted for mean distances.
    Galanti et al. (2021): https://arxiv.org/abs/2112.15121

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        V (Tensor): Matrix of within-class variance norms for the classes.
        dist_exp (int): The power with which to exponentiate the distance
            normalizer. A greater power further diminishes the contribution of
            mutual variability between already-disparate classes. Defaults to
            1, equivalent to the CDNV introduced by Galanti et al. (2021).
        tile_size (int, optional): Size of the tile for kernel computation.
            Set tile_size << K to avoid OOM. Defaults to None.

    Returns:
        float: The average CDNVs across all class pairs.
    """
    kernel_grid = class_dist_norm_var(V, M, dist_exp, tile_size)
    avg = symm_reduce(kernel_grid, pt.sum)
    return avg.item()


def mean_norms(M: Tensor, m_G: Tensor = 0, post_funcs: List[callable] = []) -> Tensor:
    """Compute the norms of (mean) embeddings (centred).

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.
        post_funcs (List[callable], optional): Functions (Tensor -> Tensor)
            applied to the computed norms. Defaults to [].

    Returns:
        Tensor: A vector containing the norms for each class.
    """
    M_centred = M - m_G
    result = M_centred.norm(dim=-1)  # (K)
    for post_func in post_funcs:
        result = post_func(result)
    return result


def interference_grid(M: Tensor, m_G: Tensor = 0) -> Tensor:
    """Compute the pairwise interference grid between (mean) embeddings.

    Arguments:
        M (Tensor): The matrix of feature (or class mean) embeddings.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.

    Returns:
        Tensor: A matrix representing pairwise interferences.
    """
    M_centred = M - m_G
    return pt.inner(M_centred, M_centred)  # (K,K)


def similarities(M: Tensor, W: Tensor, m_G: Tensor = 0, cos: bool = False) -> Tensor:
    """Compute the (cosine or dot-product) similarities between a set of (mean)
    embeddings and classifiers vectors.

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        W (Tensor): Weight vectors of the classifiers. Computations will be
            performed on the device of W.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.
        cos (bool, optional): Whether to use cosine similarity. Defaults to
            False, using dot-product similarity.

    Returns:
        Tensor: Per-class similarities between embeddings and classifiers.
    """
    M_centred = (M - m_G).to(W.device)
    if cos:
        return cosine_similarity(W, M_centred.to(W.dtype))
    return (W * M_centred).sum(dim=1)


def distance_norms(W: Tensor, M: Tensor, m_G: Tensor = 0, norm: bool = True) -> Tensor:
    """Compute the distance between (mean) embeddings and classifier vectors.

    Arguments:
        M (Tensor): Feature (e.g. class mean) embeddings vectors.
        W (Tensor): Weight vectors of the classifiers. Computations will be
            performed on the device of W.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.
        norm (bool, optional): Whether to normalize vectors before taking
            their distances. Defaults to True, allowing two dual spaces.

    Returns:
        Tensor: Per-class distances between embeddings and classifiers.
    """
    M_centred = (M - m_G).to(W.device)
    if norm:
        W, M_centred = normalize(W), normalize(M_centred)
    return (W - M_centred).norm(dim=-1)


def structure_error(A: Tensor, B: Tensor) -> float:
    """Compute the error between the cross-coherence structure formed
    by two sets of vectors and the ideal simplex equiangular tight frame
    (ETF), expressed as the matrix norm of their difference.
    Kothapalli (2023): https://arxiv.org/abs/2206.04041

    Arguments:
        A (Tensor): First tensor for comparison.
        B (Tensor): Second tensor for comparison.

    Returns:
        float: Scalar error of excess incoherence from simplex ETF.
    """
    (K, _) = A.shape

    struct = B.to(A.device) @ A.mT  # (K,K)
    struct /= la.matrix_norm(struct)

    struct += 1 / K / sqrt(K - 1)
    struct.diagonal().sub_(1 / sqrt(K - 1))

    return la.matrix_norm(struct).item()


def simplex_etf_error(M: Tensor, m_G: Tensor = 0) -> float:
    """Compute the excess cross-class incoherence within a set of (mean)
    embeddings, relative to the ideal simplex ETF.
    Kothapalli (2023): https://arxiv.org/abs/2206.04041

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.

    Returns:
        float: Scalar error of excess incoherence from simplex ETF.
    """
    M_centred = M - m_G
    return structure_error(M_centred, M_centred)


def self_duality_error(M: Tensor, W: Tensor, m_G: Tensor = 0) -> float:
    """Compute the excess cross-class incoherence between a set of (mean)
    embeddings and classifiers, relative to the ideal simplex ETF.
    Kothapalli (2023): https://arxiv.org/abs/2206.04041

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        W (Tensor): Weight vectors of the classifiers.
        m_G (Tensor, optional): Bias (e.g. global mean) vector. Defaults to 0.

    Returns:
        float: Scalar error of excess incoherence from simplex ETF.
    """
    M_centred = M - m_G
    return structure_error(M_centred, W)


def clf_ncc_agreement(
    accum: DecAccumulator, indices: List[int] = None, weighted: bool = True
) -> float:
    """Compute the rate of agreement between the linear and the implicit
    nearest-class centre (NCC) classifiers: percentage of hits over Ns samples.

    Arguments:
        accum (DecAccumulator): Tracker of per-class sample counts and hits.
        indices (List[int], optional): Indices of specific classes to include
            in agreement analysis. Defaults to [], to include all classes.
        weighted (bool, optional): Whether to weigh class hit rates by numbers
            of samples. Defaults to True.

    Returns:
        float: The rate of agreement as a float, or None if an error occurs
            (e.g. neither hits nor misses given, or shape mismatch)
    """
    _, global_agree_rates = accum.compute(indices, weighted)
    return global_agree_rates.item()


def orthogonality_deviation(M: Tensor, m_G_ood: Tensor = 0) -> float:
    """Compute the average normalized deviation of means from the global mean
    embedding from out-of-distribution (OoD) data.
    Ammar et al. (2024): https://arxiv.org/abs/2310.06823

    Arguments:
        M (Tensor): Matrix of feature (e.g. class mean) embeddings.
        m_G_ood (Tensor, optional): Out-of-Distribution bias (e.g. global
            mean) vector. Defaults to 0.
    Returns:
        float: Average normalized deviation of means from the OoD global mean.
    """
    deviations = pt.abs(similarities(M, m_G_ood, cos=True))  # (K)
    return pt.mean(deviations).item()
