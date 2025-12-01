from models.baseline import BaselineCNN
from models.batchdrop import BatchDropCNN
from models.batchnorm import BatchnormCNN
from models.deepbatchdrop import DeepBatchDropCNN
from models.dropout import DropoutCNN

from datasets import get_cifar100, get_cifar100_transfer
from models.deepwide import DeepWideCNN
from models.resnet_transfer import ResNet18TransferCNN
from models.svm_baseline import PCASVMClassifier
from models.widebatchdrop import WideBatchDropCNN
from utils import train, set_seed, save_model, find_latest_checkpoint, train_classical

import torch
import torch.nn as nn
import torch.optim as optim

import settings

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(settings.SEED)

# Loadstate is unused. It was planned for use in final evaluation, but that ended up in a separate script

experiments = [
    {
        "name": "ModelA",
        "augment": False,
        "model_class": BaselineCNN,
        "save_state": False,
        "load_state": False,
        "skip": True
    },
    {
        "name": "ModelB",
        "augment": True,
        "model_class": BaselineCNN,
        "save_state": False,
        "load_state": False,
        "skip": True
    },
    {
        "name": "ModelC",
        "augment": False,
        "model_class": BatchnormCNN,
        "save_state": False,
        "load_state": False,
        "skip": True
    },
    {
        "name": "ModelD",
        "augment": False,
        "model_class": DropoutCNN,
        "save_state": False,
        "load_state": False,
        "skip": True
    },
    {
        "name": "ModelE",
        "augment": True,
        "model_class": BatchnormCNN,
        "save_state": False,
        "load_state": False,
        "skip": True
    },
    {
        "name": "ModelF",
        "augment": True,
        "model_class": BatchDropCNN,
        "save_state": False,
        "load_state": False,
        "skip": True
    },
    {
        "name": "ModelG",
        "augment": True,
        "model_class": DeepBatchDropCNN,
        "save_state": False,
        "load_state": False,
        "skip": True
    },
    {
        "name": "ModelH",
        "augment": True,
        "model_class": WideBatchDropCNN,
        "save_state": False,
        "load_state": False,
        "skip": True
    },
    {
        "name": "ModelMML",
        "augment": False,
        "model_class": BaselineCNN,
        "save_state": False,
        "load_state": False,
        "loss_fn": nn.MultiMarginLoss(),
        "skip": True
    },
    {
        "name": "ModelT",
        "augment": True,
        "model_class": ResNet18TransferCNN,
        "save_state": False,
        "load_state": False,
        "transfer": True,
        "skip": True
    },
    {
        "name": "ModelX",
        "augment": True,
        "model_class": DeepWideCNN,
        "save_state": False,
        "load_state": False,
        "skip": True
    },
    {
        "name": "ModelCL",
        "augment": False,
        "model_class": PCASVMClassifier,
        "save_state": True,
        "load_state": True,
        "classical": True,
        "skip": True
    },
]

for exp in experiments:
    if exp.get("skip", False):
        continue

    if exp.get("classical", False):
        train_loader, val_loader, _ = get_cifar100(seed=settings.SEED, augment=False)
        train_classical(exp, train_loader, val_loader)
        continue
    elif exp.get("transfer", False):
        train_loader, val_loader, _ = get_cifar100_transfer(seed=settings.SEED, augment=exp["augment"])
    else:
        train_loader, val_loader, _ = get_cifar100(seed=settings.SEED, augment=exp["augment"])

    model = exp["model_class"]().to(device)

    if exp.get("load_state", False):
        ckpt_path = find_latest_checkpoint(exp["name"])
        if ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            continue

    print(f"[Info] Training {exp["name"]}")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = exp.get("loss_fn", nn.CrossEntropyLoss())

    best_state, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=settings.EPOCHS,
        device=device
    )

    model.load_state_dict(best_state)

    if exp["save_state"]:
        save_paths = save_model(
            model,
            exp["name"],
            history
        )
