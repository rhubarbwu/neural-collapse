from typing import Union

import torch as pt
import torch.linalg as la
from torch import Tensor

from .util import inner_product, normalize


def agreement_rate(
    Ns: Tensor, hits: Tensor = None, misses: Tensor = None
) -> Union[Tensor, float]:
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

    def variability(self, V: Tensor) -> Union[Tensor, float]:
        M_diff = self.M - self.mG  # (K,D)
        cov_inter = M_diff.mT @ M_diff / self.D  # (D,D)
        cov_intra = pt.outer(V, V)  # (D,D)
        ratio_prod = la.pinv(cov_inter) @ cov_intra  # (D,D)

        return pt.trace(ratio_prod)  # (1)

    def means_norms(
        self,
    ) -> Union[Tensor, float]:
        return self.M_centred.norm(dim=-1) ** 2  # (K)

    def interference(self) -> Tensor:
        return inner_product(normalize(self.M_centred), self.patch_size)  # (K,K)

    def _structure_error(self, Q: Tensor) -> Union[Tensor, float]:
        outer = Q @ self.M_centred.mT  # (K,K)
        outer /= pt.frobenius_norm(outer)

        K = self.K
        ideal = (pt.eye(K) - pt.ones(K, K) / K) / pt.sqrt(K - 1)  # (K,K)

        return pt.frobenius_norm(outer - ideal)

    def simplex_etf_error(self) -> Union[Tensor, float]:
        return self._structure_error(self.M)

    def alignment_error(self, W: Tensor) -> Union[Tensor, float]:
        return self._structure_error(self.W)
