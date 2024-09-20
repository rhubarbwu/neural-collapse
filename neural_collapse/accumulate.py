from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

import torch as pt
from torch import Tensor

from .util import hashify, resolve


class Accumulator(metaclass=ABCMeta):
    def __init__(
        self,
        n_classes: int,
        d_vectors: int,
        device: Union[str, pt.device] = "cpu",
        dtype: pt.dtype = pt.float32,
        ctype: pt.dtype = pt.int32,
    ):
        self.n_classes, self.d_vectors = n_classes, d_vectors
        self.ctype, self.dtype, self.device = ctype, dtype, device
        self.n_samples = pt.zeros(self.n_classes, dtype=ctype).to(device)  # (K)

    def filter_indices_by_n_samples(self, minimum: int = 0, maximum: int = None):
        idxs = self.n_samples.squeeze() >= minimum
        assert pt.all(minimum <= self.n_samples[idxs])
        if maximum:
            idxs &= self.n_samples.squeeze() < maximum
            assert pt.all(self.n_samples[idxs] < maximum)

        filtered = idxs.nonzero().squeeze()

        return filtered

    def class_idxs(self, X: Tensor, Y: Tensor):
        Y = Y.squeeze()
        assert X.shape[0] == Y.shape[0]
        Y_range = pt.arange(self.n_classes, dtype=self.ctype)
        idxs = (Y[:, None] == Y_range.to(Y.device)).to(self.device)
        self.n_samples += pt.sum(idxs, dim=0, dtype=self.ctype)[:, None].squeeze()
        return idxs

    def compute(
        self, idxs: List[int] = None, weighted: bool = False
    ) -> Tuple[Tensor, Tensor]:
        eps = pt.finfo(self.dtype).eps

        n_samples, totals = self.n_samples, self.totals  # (K), (K,D)
        if idxs is not None:
            n_samples, totals = n_samples[idxs], totals[idxs]  # (K'), (K',D)
        if len(self.totals.shape) > 1:
            n_samples = n_samples.unsqueeze(1)

        avg = totals / (n_samples + eps).to(self.dtype)  # (K, D)
        if weighted:
            avg_G = n_samples.to(self.dtype) @ avg / (n_samples.sum() + eps)  # (D)
        else:
            avg_G = avg.mean(dim=0)  # (D)

        return avg, avg_G

    @abstractmethod
    def accumulate(self, *args):
        pass


class MeanAccumulator(Accumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dtype, device = self.dtype, self.device
        self.totals = pt.zeros(self.n_classes, self.d_vectors, dtype=dtype).to(device)

    def accumulate(self, X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
        idxs = self.class_idxs(X, Y).mT.to(self.dtype)  # (K,B)
        self.totals += idxs @ X.to(device=self.device, dtype=self.dtype)  # (K,D)
        return self.n_samples, self.totals


class VarAccumulator(Accumulator):
    def __init__(self, *args, M: Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_M = None if M is None else hashify(M)
        self.totals = pt.zeros(self.n_classes, dtype=self.dtype).to(self.device)

    def accumulate(self, X: Tensor, Y: Tensor, M: Tensor) -> Tuple[Tensor, Tensor]:
        self.hash_M = resolve(self.hash_M, hashify(M))
        assert self.hash_M

        M = M.to(self.device, self.dtype)
        assert M.shape == (self.n_classes, self.d_vectors)

        idxs = self.class_idxs(X, Y).mT.to(self.dtype)  # (K,B)
        diffs_sq = (X.to(self.device) - M[Y]) ** 2  # (B,D)
        self.totals += (idxs @ diffs_sq).sum(dim=-1)  # (K,D)

        return self.n_samples, self.totals


class DecAccumulator(Accumulator):
    def __init__(self, *args, M: Tensor = None, W: Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_M = None if M is None else hashify(M)
        self.hash_W = None if W is None else hashify(W)
        self.totals = pt.zeros(self.n_classes, dtype=self.ctype).to(self.device)

    def accumulate(
        self, X: Tensor, Y: Tensor, M: Tensor, W: Tensor
    ) -> Tuple[Tensor, Tensor]:
        self.hash_M = resolve(self.hash_M, hashify(M))
        self.hash_W = resolve(self.hash_W, hashify(W))
        assert self.hash_M and self.hash_W

        X = X.to(self.device, self.dtype)
        M, W = M.to(self.device, self.dtype), W.to(self.device, self.dtype)
        assert M.shape == W.shape == (self.n_classes, self.d_vectors)

        # NCC classifier decisions
        dots = pt.inner(X, M)  # (B,K)
        feats, centre = pt.norm(X, dim=-1) ** 2, pt.norm(M, dim=-1) ** 2  # (B), (K)
        dists = feats.unsqueeze(1) + centre.unsqueeze(0) - 2 * dots  # (B,K)
        Y_ncc = dists.argmin(dim=-1)  # (B)

        # linear classifier decisions
        Y_lin = (X @ W.mT).argmax(dim=-1)  # (B)

        # count matches between classifiers
        matches = (Y_lin == Y_ncc).to(self.ctype)  # (B)
        self.class_idxs(X, Y)
        self.totals.scatter_add_(0, Y.to(self.device), matches)
