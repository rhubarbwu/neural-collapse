import torch as pt
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from neural_collapse.accumulate import (CovarAccumulator, DecAccumulator,
                                        MeanAccumulator, VarNormAccumulator)
from neural_collapse.kernels import kernel_stats, log_kernel
from neural_collapse.measure import (clf_ncc_agreement, covariance_pinv,
                                     covariance_ratio, orthogonality_deviation,
                                     self_duality_error, similarities,
                                     simplex_etf_error, variability_cdnv)
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

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
model.conv1 = nn.Conv2d(1, model.conv1.weight.shape[0], 3, 1, 1, bias=False)
model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
model.to(device)


class Features:
    pass


def hook(self, input, output):
    Features.value = input[0].clone()


# register hook that saves last-layer input into features
classifier = model.fc
classifier.register_forward_hook(hook)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr, momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, epochs_lr_decay, lr_decay)

try:
    import wandb

    wandb.init(project="neural-collapse")
    WANDB = True
except:
    WANDB = False


# Train the model
total_step = len(train_loader)
log_line = lambda epoch, i: f"Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{total_step}]"
for epoch in range(n_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"{log_line(epoch, i)}, Loss: {loss.item():.4f}")
    lr_scheduler.step()

    with pt.no_grad():
        model.eval()
        weights = model.fc.weight

        # NC collections
        mean_accum = MeanAccumulator(10, 512, "cuda")
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            mean_accum.accumulate(Features.value, labels)
        means, mG = mean_accum.compute()

        var_norms_accum = VarNormAccumulator(10, 512, "cuda", M=means)
        covar_accum = CovarAccumulator(10, 512, "cuda", M=means)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            var_norms_accum.accumulate(Features.value, labels, means)
            covar_accum.accumulate(Features.value, labels, means)
        var_norms, _ = var_norms_accum.compute()
        covar_within = covar_accum.compute()

        dec_accum = DecAccumulator(10, 512, "cuda", M=means, W=weights)
        dec_accum.create_index(means)  # optionally use FAISS index for NCC
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # mean embeddings (only) necessary again if not using FAISS index
            if dec_accum.index is None:
                dec_accum.accumulate(Features.value, labels, weights, means)
            else:
                dec_accum.accumulate(Features.value, labels, weights)

        ood_mean_accum = MeanAccumulator(10, 512, "cuda")
        for i, (images, labels) in enumerate(ood_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            ood_mean_accum.accumulate(Features.value, labels)
        _, mG_ood = ood_mean_accum.compute()

        # NC measurements
        results = {
            "nc1_pinv": covariance_pinv(covar_within, means, mG),
            "nc1_svd": covariance_pinv(covar_within, means, mG, svd=True),
            "nc1_quot": covariance_ratio(covar_within, means, mG),
            "nc1_cdnv": variability_cdnv(var_norms, means, tile_size=64),
            "nc2_etf_err": simplex_etf_error(means, mG),
            "nc2g_dist": kernel_stats(means, mG, tile_size=64)[1],
            "nc2g_log": kernel_stats(means, mG, kernel=log_kernel, tile_size=64)[1],
            "nc3_self_dual": self_duality_error(weights, means, mG),
            "nc3u_uni_dual": similarities(weights, means, mG).var().item(),
            "nc4_agree": clf_ncc_agreement(dec_accum),
            "nc5_ood_dev": orthogonality_deviation(means, mG_ood),
        }

        if WANDB:
            wandb.log(results)
        else:
            print(results)
