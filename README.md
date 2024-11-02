# Neural Collapse

This package aims to be a reference implementation for the analysis of
[Neural Collapse (NC) (Papyan et al., 2020)](https://www.pnas.org/doi/full/10.1073/pnas.2015509117).
We provide,

1. Accumulators to collect embeddings from output representations from your
   pre-trained model.
2. Measurement (kernel) functions for several canonical and modern NC metrics.
3. Tiling support for memory-bound settings arising from large embeddings,
   many classes and/or limited parallel accelerator (e.g. GPU) memory.

## Installation

```sh
# install from remote
pip install git+https://github.com/rhubarbwu/neural-collapse.git

# with FAISS
pip install git+https://github.com/rhubarbwu/neural-collapse.git#egg=neural_collapse[faiss]

# install locally from a repository clone [with FAISS]
git clone https://github.com/rhubarbwu/neural-collapse.git
pip install -e neural-collapse[faiss]
```

## Usage

```py
import neural_collapse as nc
```

We assume that you,

- Already pre-trained your model or are in the training process with a
  programmable loop, where the top-layer classifier weights are available.
- Have your iterable dataloader(s) available. Make sure your training data is
  the same as that with which you trained your model.
- Have model evaluation functions or results; technically optional but ideal.

For use cases with large embeddings or many classes, we recommend using a
hardware accelerator (e.g. `cuda`).

### Accumulators

You'll need to collect (e.g. "accumulate") statistics from you learned
representations. Here we outline a
[basic example on the MNIST dataset](./examples/mnist.py) with `K=10`
classes and embeddings of size `D=512`.

```py
from neural_collapse.accumulate import (CovarAccumulator, DecAccumulator,
                                        MeanAccumulator, VarNormAccumulator)
```

#### Mean Embedding Accumulators (for NC\* in general)

```py
mean_accum = MeanAccumulator(10, 512, "cuda")
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    mean_accum.accumulate(Features.value, labels)
means, mG = mean_accum.compute()
```

#### Variance Accumulators (for NC1)

For measuring within-class variability collapse (NC1), you would typically
collect within-class covariances (`covar_accum` below); note that this might
be memory-intensive at order `K*D*D`.

```py
covar_accum = CovarAccumulator(10, 512, "cuda", M=means)
var_norms_accum = VarNormAccumulator(10, 512, "cuda", M=means) # for CDNV
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    covar_accum.accumulate(Features.value, labels, means)
    var_norms_accum.accumulate(Features.value, labels, means)
covar_within = covar_accum.compute()
var_norms, _ = var_norms_accum.compute() # for CDNV
```

NC1 can also be empirically measured using the class-distance normalized
variance [(CDNV) (Galanti et. al, 2021)](https://arxiv.org/abs/2112.15121),
which only requires collecting within-class variance norms at order `K`.

#### Decision Agreement Accumulators (for NC4)

Measuring the convergence of the linear classifier's behaviour to that of the
implicit near-class centre (NCC) classifier has since been extended to
generalizing to unseen (e.g. validation or test) data.

```py
dec_accum = DecAccumulator(10, 512, "cuda", M=means, W=weights)
dec_accum.create_index(means) # optionally use FAISS index for NCC
for i, (images, labels) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)

    # mean embeddings (only) necessary again if not using FAISS index
    if dec_accum.index is None:
        dec_accum.accumulate(Features.value, labels, weights, means)
    else:
        dec_accum.accumulate(Features.value, labels, weights)
```

#### Out-of-Distribution (OoD) Means (for NC5)

For OoD detection
[(NC5) (Ammar et al., 2024)](https://arxiv.org/abs/2310.06823), collect
class-mean embeddings from an out-of-distribution dataset for OoD detection.

```py
ood_mean_accum = MeanAccumulator(10, 512, "cuda")
for i, (images, labels) in enumerate(ood_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    ood_mean_accum.accumulate(Features.value, labels)
_, mG_ood = ood_mean_accum.compute()
```

### Measurements

Here's our snippet for an [example on the MNIST dataset](./examples/mnist.py).

```py
from neural_collapse.measure import (clf_ncc_agreement, covariance_pinv,
                                     covariance_ratio, orthogonality_deviation,
                                     self_duality_error, simplex_etf_error,
                                     variability_cdnv)

results = {
    "nc1_pinv": covariance_pinv(covar_within, means, mG, svd=True),
    "nc1_svd": covariance_pinv(covar_within, means, mG, svd=True),
    "nc1_quot": covariance_ratio(covar_within, means, mG),
    "nc1_cdnv": variability_cdnv(var_norms, means),
    "nc2_etf_err": simplex_etf_error(means, mG),
    "nc2g_dist": kernel_stats(means, mG)[1],
    "nc2g_log": kernel_stats(means, mG, kernel=log_kernel)[1],
    "nc3_self_dual": self_duality_error(weights, means, mG),
    "nc3u_uni_dual": similarities(weights, means, mG).var().item(),
    "nc4_agree": clf_ncc_agreement(dec_accum),
    "nc5_ood_dev": orthogonality_deviation(means, mG_ood),
}
```

#### Pre-Centring Means

Where centring is required for `means`, you can include the global mean `mG`
as a bias argument (as above), or pre-centre them (as below).

```py
means_centred = means - mG
results = {
    "nc1_pinv": covariance_pinv(covar_within, means_centred),
    "nc1_svd": covariance_pinv(covar_within, means_centred, svd=True),
    "nc1_quot": covariance_ratio(covar_within, means_centred),
    "nc1_cdnv": variability_cdnv(var_norms, means),
    # ...
    "nc5_ood_dev": orthogonality_deviation(means, mG_ood),
}
```

Note that since the uncentred means are still needed for some measurements
(such as CDNV) (and therefore cannot be discarded), storing pre-centred means
may not be economical memory-wise if `K` and/or `D` are large.

#### Tiling & Reductions

For many of the NC measurement functions, we implement kernel tiling if large
embeddings or many classes are straining your hardware memory. You may want to
tune the tile square size to maximize accelerator throughput.

```py
results = {
    # ...
    "nc1_cdnv": variability_cdnv(var_norms, means, tile_size=64),
    "nc2g_dist": kernel_stats(means, mG, tile_size=64)[1], # var
    "nc2g_log": kernel_stats(means, mG, kernel=log_kernel, tile_size=64)[1], # var
    # ...
}
```

After `kernel_grid` produces a symmetric measurement matrix, `kernel_stats`
computes the mean (`[0]`) and variance (`[1]`) using triangle row folding.

## Development

This project is under active development. Feel free to open issues for bugs,
features, optimizations, or papers you would like (us) to implement.

## References

- [Prevalence of neural collapse during the terminal phase of deep learning training](https://www.pnas.org/doi/full/10.1073/pnas.2015509117)
- [On the Role of Neural Collapse in Transfer Learning](https://arxiv.org/abs/2112.15121)
- [Neural Collapse: A Review on Modelling Principles and Generalization](https://arxiv.org/abs/2206.04041)
- [Perturbation Analysis of Neural Collapse](https://proceedings.mlr.press/v202/tirer23a)
- [Generalizing and Decoupling Neural Collapse via Hyperspherical Uniformity Gap](https://arxiv.org/abs/2303.06484)
- [NECO: NEural Collapse Based Out-of-distribution detection](https://arxiv.org/abs/2310.06823)
- [Linguistic Collapse: Neural Collapse in (Large) Language Models](https://arxiv.org/abs/2405.17767)
