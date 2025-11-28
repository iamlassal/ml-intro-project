from models.baseline import BaselineCNN
from models.batchdrop import BatchDropCNN
from models.batchnorm import BatchnormCNN
from models.deepbatchdrop import DeepBatchDropCNN
from models.dropout import DropoutCNN

from datasets import get_cifar100, get_cifar100_transfer
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

experiments = [
    {
        "name": "ModelA",
        "augment": False,
        "model_class": BaselineCNN,
        "save_state": True,
        "load_state": True,
    },
    {
        "name": "ModelB",
        "augment": True,
        "model_class": BaselineCNN,
        "save_state": True,
        "load_state": True,
    },
    {
        "name": "ModelC",
        "augment": False,
        "model_class": BatchnormCNN,
        "save_state": True,
        "load_state": True,
    },
    {
        "name": "ModelD",
        "augment": False,
        "model_class": DropoutCNN,
        "save_state": True,
        "load_state": True,
    },
    {
        "name": "ModelE",
        "augment": True,
        "model_class": BatchnormCNN,
        "save_state": True,
        "load_state": True,
    },
    {
        "name": "ModelF",
        "augment": True,
        "model_class": BatchDropCNN,
        "save_state": True,
        "load_state": True,
    },
    {
        "name": "ModelG",
        "augment": True,
        "model_class": DeepBatchDropCNN,
        "save_state": True,
        "load_state": True,
    },
    {
        "name": "ModelH",
        "augment": True,
        "model_class": WideBatchDropCNN,
        "save_state": True,
        "load_state": True,
    },
    {
        "name": "ModelT",
        "augment": True,
        "model_class": ResNet18TransferCNN,
        "save_state": True,
        "load_state": False,
        "transfer": True
    },
    {
        "name": "ModelCL",
        "augment": False,
        "model_class": PCASVMClassifier,
        "save_state": True,
        "load_state": False,
        "classical": True
    },
]

for exp in experiments:
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
    loss_fn = nn.CrossEntropyLoss()

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
