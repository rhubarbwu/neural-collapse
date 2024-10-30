import torch as pt
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import wandb

from neural_collapse.accumulate import (
    CovarAccumulator,
    DecAccumulator,
    MeanAccumulator,
    VarNormAccumulator,
)
from neural_collapse.measure import (
    clf_ncc_agreement,
    covariance_pinv,
    covariance_ratio,
    orthogonality_deviation,
    self_duality_error,
    simplex_etf_error,
    variability_cdnv,
)


def replace_layers(model, max_channels=16):
    for name, module in model.named_children():
        # Replace Conv2d layers with in_channels > max_channels
        if isinstance(module, nn.Conv2d) and module.in_channels >= max_channels:
            new_conv = nn.Conv2d(
                in_channels=max_channels,
                out_channels=max_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            setattr(model, name, new_conv)

        # Replace BatchNorm2d layers with num_features > max_channels
        elif isinstance(module, nn.BatchNorm2d) and module.num_features >= max_channels:
            new_bn = nn.BatchNorm2d(
                num_features=max_channels,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
            )
            setattr(model, name, new_bn)

        # Recursively apply the function to child modules
        replace_layers(module, max_channels)


# Device configuration
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

# Hyperparameters
n_epochs = 200
batch_size = 128
lr, epochs_lr_decay, lr_decay = 0.0679, [n_epochs // 3, n_epochs * 2 // 3], 0.1
momentum = 0.9
weight_decay = 5e-4

# MNIST dataset
transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_dataset = MNIST("./data", True, transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNIST("./data", False, transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
# OoD dataset (Fashion MNIST), for NC5
ood_dataset = FashionMNIST("./data", False, transform, download=True)
ood_loader = DataLoader(dataset=ood_dataset, batch_size=batch_size, shuffle=True)


# ResNet model
model = models.resnet18(num_classes=10, weights=None).to(device)
model.conv1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
model.fc = nn.Linear(in_features=16, out_features=10)
replace_layers(model)

print(model)


class ModifiedResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()
        self.features = nn.ModuleList(original_model.children())

    def forward(self, x):
        outputs = []
        for i, module in enumerate(self.features):
            if isinstance(module, nn.Sequential):
                x = module(x)
                outputs.append(x.flatten(start_dim=1))
            elif isinstance(module, nn.Linear):
                x = x.flatten(start_dim=1)
                outputs.append(x)
                x = module(x)
            else:
                x = module(x)
        return x, outputs


model = ModifiedResNet(model).to(device=device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr, momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, epochs_lr_decay, lr_decay)


wandb.init(project="neural-collapse")


with pt.no_grad():
    latents = model(pt.ones(1, 1, 28, 28).cuda())[-1]

# Train the model
total_step = len(train_loader)
log_line = lambda epoch, i: f"Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{total_step}]"
for epoch in range(n_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"{log_line(epoch, i)}, Loss: {loss.item():.4f}")
    lr_scheduler.step()

    with pt.no_grad():
        model.eval()

        # NC collections
        mean_accums = [MeanAccumulator(10, t.shape[-1], "cuda") for t in latents]
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, hiddens = model(images)
            for mean_accum, hidden in zip(mean_accums, hiddens):
                mean_accum.accumulate(hidden, labels)
        mean_stats = [mean_accum.compute() for mean_accum in mean_accums]
        # means, mG = mean_accum.compute()

        var_norms_accums = [
            VarNormAccumulator(10, t.shape[-1], "cuda", M=mean)
            for ((mean, _), t) in zip(mean_stats, latents)
        ]
        covar_accums = [
            CovarAccumulator(10, t.shape[-1], "cuda", M=mean)
            for ((mean, _), t) in zip(mean_stats, latents)
        ]
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, hiddens = model(images)

            for (mean, _), var_norms_accum, covar_accum, hidden in zip(
                mean_stats, var_norms_accums, covar_accums, hiddens
            ):
                var_norms_accum.accumulate(hidden, labels, mean)
                covar_accum.accumulate(hidden, labels, mean)
        vars_norms = [
            var_norms_accum.compute()[0] for var_norms_accum in var_norms_accums
        ]
        covars_within = [covar_accum.compute() for covar_accum in covar_accums]

        dec_accum = DecAccumulator(
            10,
            latents[-1].shape[-1],
            "cuda",
            M=mean_stats[-1][0],
            W=model.features[-1].weight,
        )
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, hiddens = model(images)
            dec_accum.accumulate(
                hiddens[-1], labels, model.features[-1].weight, mean_stats[-1][0]
            )

        ood_mean_accums = [MeanAccumulator(10, t.shape[-1], "cuda") for t in latents]
        for i, (images, labels) in enumerate(ood_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, hiddens = model(images)
            for ood_mean_accum, hidden in zip(ood_mean_accums, hiddens):
                ood_mean_accum.accumulate(hidden, labels)
        ood_mean_stats = [
            ood_mean_accum.compute() for ood_mean_accum in ood_mean_accums
        ]

        # NC measurements

        for i in range(len(latents)):
            means, mG = mean_stats[i]
            covar_within = covars_within[i]
            var_norms = vars_norms[i]
            _, mG_ood = ood_mean_stats[i]

            results = {
                "nc1_covariance_pinv": covariance_pinv(
                    means, covar_within, mG, svd=True
                ),
                "nc1_covariance_ratio": covariance_ratio(means, covar_within, mG),
                "nc1_variability_cdnv": variability_cdnv(means, var_norms),
                "nc2_simplex_etf_error": simplex_etf_error(means, mG),
                "nc5_ood_deviation": orthogonality_deviation(means, mG_ood),
            }
            if i == (len(latents) - 1):
                results["nc4_decs_agreement"] = clf_ncc_agreement(dec_accum)
                results["nc3_self_duality"] = self_duality_error(
                    means, model.features[-1].weight, mG
                )

            wandb.log({f"layer {i + 1}/{k}": v for k, v in results.items()})

pt.save(model.state_dict(), "collapsed_network.pt")
