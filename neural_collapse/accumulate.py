from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

import torch as pt
from torch import Tensor

from .util import hashify, resolve


class Accumulator(metaclass=ABCMeta):
    """Base class for accumulators that track sample counts and totals.

    This abstract class provides the foundation for different types of
    accumulators that calculate statistics based on input tensors. It manages
    sample counts for each class and includes methods to filter indices based
    on sample counts, accumulate data, and compute averages.

    Attributes:
        n_classes (int): Number of classes.
        d_vectors (int): Dimensionality of the input vectors.
        ctype (torch.dtype): Data type for counts.
        dtype (torch.dtype): Data type for totals and computations.
        device (Union[str, torch.device]): Device on which tensors are stored.
        ns_samples (Tensor): Tensor to store per-class sample counts.

    Methods:
        filter_indices_by_n_samples(minimum=0, maximum=None):
            Filters class indices based on a minimum and maximum sample count.
        class_idxs(X, Y):
            Increment sample counts and return per-class sample indices.
        compute(idxs=None, weighted=False):
            Computes averages of accumulated samples (totals/counts).
        accumulate(*args):
            Abstract method to be implemented by subclasses to define specific
            accumulation behavior.
    """

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
        self.ns_samples = pt.zeros(self.n_classes, dtype=ctype).to(device)  # (K)

    def filter_indices_by_n_samples(
        self, minimum: int = 0, maximum: int = None
    ) -> Union[Tensor, float]:
        idxs = self.ns_samples.squeeze() >= minimum
        assert pt.all(minimum <= self.ns_samples[idxs])
        if maximum:
            idxs &= self.ns_samples.squeeze() < maximum
            assert pt.all(self.ns_samples[idxs] < maximum)

        filtered = idxs.nonzero().squeeze()

        return filtered

    def class_idxs(self, X: Tensor, Y: Tensor) -> Union[Tensor, float]:
        Y = Y.squeeze()
        assert X.shape[0] == Y.shape[0]
        Y_range = pt.arange(self.n_classes, dtype=self.ctype)
        idxs = (Y[:, None] == Y_range.to(Y.device)).to(self.device)
        self.ns_samples += pt.sum(idxs, dim=0, dtype=self.ctype)[:, None].squeeze()
        return idxs

    def compute(
        self, idxs: List[int] = None, weighted: bool = False
    ) -> Tuple[Tensor, Tensor]:
        ns_samples, totals = self.ns_samples, self.totals  # (K), (K,D)
        if idxs is not None:
            ns_samples, totals = ns_samples[idxs], totals[idxs]  # (K'), (K',D)
        if len(self.totals.shape) > 1:
            ns_samples = ns_samples.unsqueeze(1)

        eps = pt.finfo(self.dtype).eps
        avg = totals / (ns_samples + eps).to(self.dtype)  # (K, D)
        if weighted:
            avg_G = ns_samples.to(self.dtype) @ avg / (ns_samples.sum() + eps)  # (D)
        else:
            avg_G = avg.mean(dim=0)  # (D)

        return avg, avg_G

    @abstractmethod
    def accumulate(self, *args):
        pass


class MeanAccumulator(Accumulator):
    """Accumulator that computes mean vectors for each class.

    Inherits from the Accumulator class: accumulates the totals for each
    class to compute the mean of the input vectors.

    Methods:
        accumulate(X, Y):
            Increment and return per-class mean totals and sample counts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dtype, device = self.dtype, self.device
        self.totals = pt.zeros(self.n_classes, self.d_vectors, dtype=dtype).to(device)

    def accumulate(self, X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
        idxs = self.class_idxs(X, Y).mT.to(self.dtype)  # (K,B)
        self.totals += idxs @ X.to(device=self.device, dtype=self.dtype)  # (K,D)
        return self.ns_samples, self.totals


class CovarAccumulator(Accumulator):
    """Accumulator that computes covariance matrices for each class.

    Inherits from the Accumulator class: accumulates the differences between
    input vectors and class-specific mean vectors to compute covariance.

    Attributes:
        hash_M (hash): Hash of the mean vectors for ensuring consistency.
        totals (Tensor): Matrix to store the accumulated covariance totals.

    Methods:
        accumulate(X, Y, M):
            Increment covariance totals with squared differences between
            input vectors and their label-corresponding class means.
        compute(idxs=None):
            Computes the average covariance matrix from accumulated totals.
    """

    def __init__(self, *args, M: Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        D, dtype, device = self.d_vectors, self.dtype, self.device
        self.hash_M = None if M is None else hashify(M)
        self.totals = pt.zeros(D, D, dtype=dtype).to(device)

    def accumulate(self, X: Tensor, Y: Tensor, M: Tensor) -> Tuple[Tensor, Tensor]:
        self.hash_M = resolve(self.hash_M, hashify(M))
        assert self.hash_M

        M = M.to(self.device, self.dtype)
        assert M.shape == (self.n_classes, self.d_vectors)

        self.class_idxs(X, Y)
        diff = X.to(self.device) - M[Y]  # (B,D)
        self.totals += diff.mT @ diff  # (D,D)

        return self.ns_samples, self.totals

    def compute(self, idxs: List[int] = None) -> Tuple[Tensor, Tensor]:
        ns_samples, totals = self.ns_samples, self.totals  # (K), (K,D)
        if idxs is not None:
            ns_samples, totals = ns_samples[idxs], totals[idxs]

        eps = pt.finfo(self.dtype).eps
        return totals / (ns_samples.sum() + eps).to(self.dtype)


class VarNormAccumulator(Accumulator):
    """Accumulator that computes the variance norms for each class.

    Inherits from the Accumulator class and calculates the variance of input
    vectors relative to class-specific mean vectors.

    Attributes:
        hash_M (hash): Hash of the mean vectors for ensuring consistency.
        totals (Tensor): A tensor to hold the accumulated variance totals.

    Methods:
        accumulate(X, Y, M):
            Increment per-class totals with norms of squared differences
            between input vectors and their label-corresponding class means.
    """

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

        return self.ns_samples, self.totals


class DecAccumulator(Accumulator):
    """Accumulator that computes decision hits from multiple classifiers.

    Inherits from the Accumulator class and integrates results from
    nearest-class center and linear classifiers to track hits for each class.

    Attributes:
        hash_M (hash): Hash of the mean vectors for ensuring consistency.
        hash_W (hash): Hash of the linear classifier weights for consistency.
        index (IndexFlatL2): FAISS index for efficient nearest neighbors.
        totals (Tensor): Tensor of per-class hit counts.

    Methods:
        create_index(M):
            Initializes a FAISS (if installed) index with the provided mean
                vectors. If FAISS not found, do nothing.
        accumulate(X, Y, W, M=None):
            Updates the totals based on hits between the nearest-class center
            and linear classifiers.
        compute(idxs=None, weighted=True):
            Computes per-class and global classifier decision agreement rates.
    """

    def __init__(self, *args, M: Tensor = None, W: Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_M = None if M is None else hashify(M)
        self.hash_W = None if W is None else hashify(W)
        self.index = None
        self.totals = pt.zeros(self.n_classes, dtype=self.ctype).to(self.device)

    def create_index(self, M: Tensor):
        self.hash_M = resolve(self.hash_M, hashify(M))
        assert self.hash_M

        try:
            from faiss import IndexFlatL2

            self.index = IndexFlatL2(self.d_vectors)
            self.index.add(M.cpu().numpy())
        except:
            self.index = None

    def accumulate(
        self,
        X: Tensor,
        Y: Tensor,
        W: Tensor,
        M: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:

        self.hash_W = resolve(self.hash_W, hashify(W))
        assert self.hash_W
        assert W.shape == (self.n_classes, self.d_vectors)
        X, W = X.to(self.device, self.dtype), W.to(self.device, self.dtype)

        # NCC classifier decisions
        if self.index:  # using FAISS index
            _, I = self.index.search(X.cpu().numpy(), 1)
            Y_ncc = pt.tensor(I).to(self.device).squeeze()  # (B)
        else:  # manual near-class centre, using given means
            assert type(M) == Tensor
            self.hash_M = resolve(self.hash_M, hashify(M))
            assert self.hash_M
            assert M.shape == (self.n_classes, self.d_vectors)
            M = M.to(self.device, self.dtype)

            dots = pt.inner(X, M)  # (B,K)
            feats, centre = pt.norm(X, dim=-1) ** 2, pt.norm(M, dim=-1) ** 2  # (B), (K)
            dists = feats.unsqueeze(1) + centre.unsqueeze(0) - 2 * dots  # (B,K)
            Y_ncc = dists.argmin(dim=-1)  # (B)

        # linear classifier decisions
        Y_lin = (X @ W.mT).argmax(dim=-1)  # (B)

        # count matches between classifiers
        matches = (Y_lin == Y_ncc).to(self.ctype)  # (B)
        self.class_idxs(X, Y)
        self.totals.scatter_add_(0, Y.to(self.device, pt.int64), matches)

        return self.ns_samples, self.totals
