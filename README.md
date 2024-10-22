# Neural Collapse

## Installation

```sh
# install from remote
pip install git+https://github.com/rhubarbwu/neural-collapse.git

# with FAISS
pip install git+https://github.com/rhubarbwu/neural-collapse.git#egg=neural_collapse[faiss]

# install locally [with faiss]
pip install -e .[faiss]
```

## Import

```py
import neural_collapse as nc
```

## Usage

We assume that, you,

- Already trained your model or are in the process with a programmable loop,
  where the top-layer classifier weights are available.
- Have your iterable dataloader(s) available.
- Have model evaluation functions or results; technically optional but ideal to have.

### Accumulators

Firstly, you'll need to accumulate the,

1. Class-mean embeddings from the train set.
2. Within-class covariances or variance norms from the train set.
3. Decision agreement from unseen data (validation and/or test).

Here's our snippet for an [example on the MNIST dataset](./examples/mnist.py).

```py
from neural_collapse.accumulate import (CovarAccumulator, DecAccumulator,
                                        MeanAccumulator, VarNormAccumulator)

mean_accum = MeanAccumulator(10, 512, "cuda")
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    mean_accum.accumulate(Features.value, labels)
means, mG = mean_accum.compute()

var_norms_accum = VarNormAccumulator(10, 512, "cuda", M=means) # for CDNV
covar_accum = CovarAccumulator(10, 512, "cuda", M=means)
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    var_norms_accum.accumulate(Features.value, labels, means)
    covar_accum.accumulate(Features.value, labels, means)
var_norms, _ = var_norms_accum.compute() # for CDNV
covar_within = covar_accum.compute()

dec_accum = DecAccumulator(10, 512, "cuda", M=means, W=model.fc.weight)
dec_accum.create_index(means) # index makes means unnecessary in accumulation
for i, (images, labels) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    dec_accum.accumulate(Features.value, labels, model.fc.weight)
```

#### Class-Distance Normalized Variance (CDNV)

NC1 can also be empirically measured by the
[CDNV (Galanti et. al, 2021)](https://arxiv.org/abs/2112.15121)
based on the variance norms from the second pass over the train set.

#### Out-of-Distribution (OoD) Means

Optionally, collect class-mean embeddings from an out-of-distribution dataset
[NC5 (Ammar et al., 2024)](https://arxiv.org/abs/2310.06823).

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
results = {
    "nc1_covar_pinv": covariance_pinv(covar_within, means, mG, svd=True),
    "nc1_covar_ratio": covariance_ratio(covar_within, means, mG),
    "nc1_var_cdnv": variability_cdnv(var_norms, means),
    "nc2_simplex_etf_error": simplex_etf_error(means, mG),
    "nc3_self_duality": self_duality_error(weights, means, mG),
    "nc4_decs_agreement": clf_ncc_agreement(dec_accum),
    "nc5_ood_deviation": orthogonality_deviation(means, mG_ood),
}
```

Where centring is required for `means`, you can pre-centre them or include the global mean `mG` as a bias argument, up to you!

```py
means_centred = means - mG
results = {
    "nc1_covar_pinv": covariance_pinv(covar_within, means_centred, svd=True),
    "nc1_covar_ratio": covariance_ratio(covar_within, means_centred),
    "nc1_var_cdnv": variability_cdnv(means, var_norms),
    "nc2_simplex_etf_error": simplex_etf_error(means_centred),
    "nc3_self_duality": self_duality_error(model.fc.weight, means_centred),
    "nc4_decs_agreement": clf_ncc_agreement(dec_accum),
    "nc5_ood_deviation": orthogonality_deviation(means, mG_ood),
}
```

## References

- [Prevalence of neural collapse during the terminal phase of deep learning training](https://www.pnas.org/doi/full/10.1073/pnas.2015509117)
- [On the Role of Neural Collapse in Transfer Learning](https://arxiv.org/abs/2112.15121)
- [Neural Collapse: A Review on Modelling Principles and Generalization](https://arxiv.org/abs/2206.04041)
- [Perturbation Analysis of Neural Collapse](https://proceedings.mlr.press/v202/tirer23a)
- [Generalizing and Decoupling Neural Collapse via Hyperspherical Uniformity Gap](https://arxiv.org/abs/2303.06484)
- [NECO: NEural Collapse Based Out-of-distribution detection](https://arxiv.org/abs/2310.06823)
- [Linguistic Collapse: Neural Collapse in (Large) Language Models](https://arxiv.org/abs/2405.17767)
