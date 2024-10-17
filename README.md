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

model.eval()

mean_accum = MeanAccumulator(10, 512, "cuda")
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    mean_accum.accumulate(Features.value, labels)
means, mG = mean_accum.compute()

var_accum = VarNormAccumulator(10, 512, "cuda", M=means)
covar_accum = CovarAccumulator(10, 512, "cuda", M=means)
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    var_accum.accumulate(Features.value, labels, means)
    covar_accum.accumulate(Features.value, labels, means)
var, _ = var_accum.compute()
covar = covar_accum.compute()

dec_accum = DecAccumulator(10, 512, "cuda", M=means, W=model.fc.weight)
for i, (images, labels) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    dec_accum.accumulate(Features.value, labels, means, model.fc.weight)

ood_mean_accum = MeanAccumulator(10, 512, "cuda")
for i, (images, labels) in enumerate(ood_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    ood_mean_accum.accumulate(Features.value, labels)
_, mG_ood = ood_mean_accum.compute()

results = {
    "nc1_variability_pinv": variability_pinv(covar, means, mG, svd=True),
    "nc1_variability_ratio": variability_ratio(covar, means, mG),
    "nc1_variability_cdnv": variability_cdnv(var, means),
    "nc2_simplex_etf_error": simplex_etf_error(means, mG),
    "nc3_self_duality": self_duality_error(model.fc.weight, means, mG),
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
