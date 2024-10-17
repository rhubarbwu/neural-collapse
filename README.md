# Neural Collapse

## Installation

```sh
# install from remote
pip install git+https://github.com/rhubarbwu/neural-collapse.git

# install locally
pip install -e .
```

## Import

```py
import neural_collapse as nc
```

## Usage

Here's an snippet for an [example on the MNIST dataset](./examples).

```py
from neural_collapse.accumulate import (CovarAccumulator, DecAccumulator,
                                        MeanAccumulator, VarNormAccumulator)
from neural_collapse.measure import (clf_ncc_agreement, self_duality_error,
                                     simplex_etf_error, variability_cdnv,
                                     variability_pinv, variability_ratio)

mean_accum = MeanAccumulator(n_classes=10, d_vectors=512, device="cuda")
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    h = Features.value
    mean_accum.accumulate(h, labels)
means, mG = mean_accum.compute()

var_accum = VarNormAccumulator(n_classes=10, d_vectors=512, device="cuda")
var_mat_accum = CovarAccumulator(n_classes=10, d_vectors=512, device="cuda")
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    h = Features.value
    var_accum.accumulate(h, labels, means)
    var_mat_accum.accumulate(h, labels, means)
var, _ = var_accum.compute()
covar = var_mat_accum.compute()

dec_accum = DecAccumulator(n_classes=10, d_vectors=512, device="cuda")
for i, (images, labels) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    h = Features.value
    dec_accum.accumulate(h, labels, means, model.fc.weight)

results = {
    "nc1_variability_pinv": variability_pinv(covar, means, mG, svd=True),
    "nc1_variability_ratio": variability_ratio(covar, means, mG),
    "nc1_variability_cdnv": variability_cdnv(var, means, 1),
    "nc2_simplex_etf_error": simplex_etf_error(means, mG),
    "nc3_self_duality": self_duality_error(model.fc.weight, means, mG),
    "nc4_agreement": clf_ncc_agreement(dec_accum.ns_samples, dec_accum.totals),
}
```

## References

- [Prevalence of neural collapse during the terminal phase of deep learning training](https://www.pnas.org/doi/full/10.1073/pnas.2015509117)
- [On the Role of Neural Collapse in Transfer Learning](https://arxiv.org/abs/2112.15121)
- [Neural Collapse: A Review on Modelling Principles and Generalization](https://arxiv.org/abs/2206.04041)
- [Perturbation Analysis of Neural Collapse](https://proceedings.mlr.press/v202/tirer23a)
- [Generalizing and Decoupling Neural Collapse via Hyperspherical Uniformity Gap](https://arxiv.org/abs/2303.06484)
- [Linguistic Collapse: Neural Collapse in (Large) Language Models](https://arxiv.org/abs/2405.17767)
