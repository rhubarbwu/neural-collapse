from math import sqrt
from typing import Union

import torch as pt
import torch.linalg as la
from torch import Tensor

from .util import inner_product, normalize


def agreement(Ns: Tensor, hits: Tensor = None, misses: Tensor = None) -> Tensor:
    if hits is None and misses and misses.shape == Ns.shape:
        hits = Ns - misses

    if hits and hits.shape == Ns.shape:
        return (hits / Ns).mean()

    return None


class NeuralCollapse:
    def __init__(self, M: Tensor, mG: Tensor, patch_size: int = None):
        self.K, self.D = M.shape
        self.M_centred = M - mG
        self.patch_size = patch_size

    def variability(self, cov_intra: Tensor) -> float:
        cov_inter = self.M_centred.mT @ self.M_centred / self.D  # (D,D)
        ratio_prod = la.pinv(cov_inter) @ cov_intra.to(cov_inter.device)  # (D,D)

        return pt.trace(ratio_prod).item()  # (1), divide by N?

    def means_norms(self) -> Tensor:
        return self.M_centred.norm(dim=-1) ** 2  # (K)

    def interference(self) -> Tensor:
        return inner_product(normalize(self.M_centred), self.patch_size)  # (K,K)

    def _structure_error(self, Q: Tensor) -> Tensor:
        outer = Q.to(self.M_centred.device) @ self.M_centred.mT  # (K,K)
        outer /= la.matrix_norm(outer)

        K = self.K
        ideal = (pt.eye(K) - pt.ones(K, K) / K) / sqrt(K - 1)  # (K,K)

        return la.matrix_norm(outer - ideal.to(outer.device))

    def simplex_etf_error(self) -> float:
        return self._structure_error(self.M_centred).item()

    def alignment_error(self, W: Tensor) -> float:
        return self._structure_error(W).item()
