import torch as pt
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from neural_collapse.accumulate import (CovarAccumulator, DecAccumulator,
                                        MeanAccumulator, VarNormAccumulator)
from neural_collapse.measure import (clf_ncc_agreement, self_duality_error,
                                     simplex_etf_error, variability_cdnv,
                                     variability_pinv, variability_ratio)

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

        if WANDB:
            wandb.log(results)
        else:
            print(results)
