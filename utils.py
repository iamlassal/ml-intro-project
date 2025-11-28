import torch
import random
import os
import json
import glob
import numpy as np
import settings
from datetime import datetime
from time import time
import copy

def set_seed(seed=42):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total

def evaluate_epoch(model, dataloader, loss_fn, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct, total = accuracy(logits, labels)
            total_correct += correct
            total_samples += total

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc

def train_one_epoch(model, dataloader, optimizer, loss_fn, device="cpu"):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs,
    device="cpu"
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "total_time": None
    }

    best_val_acc = 0.0
    best_state = None

    t_start = time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Elapsed Time: {(time() - t_start):.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    history["total_time"] = time() - t_start
    history["seed"] = settings.SEED

    return best_state, history

def train_classical(exp, train_loader, val_loader):
    name = exp["name"]
    model_class = exp["model_class"]

    if exp.get("load_state", False):
        prefix = find_latest_svm_checkpoint(name)
        if prefix is not None:
            print(f"[INFO] Loaded classical model checkpoint for {name}")
            model = model_class.load(prefix)
            return None

    print(f"[INFO] Training classical model: {name}")
    model = model_class()
    history = model.fit(train_loader, val_loader)

    if exp.get("save_state", False):
        os.makedirs("checkpoints", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        prefix = f"checkpoints/{name}_{timestamp}"

        saved_paths = model.save(prefix)

        history_path = prefix + "_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

        print("[INFO] Saved classical model components:")
        print(f"       PCA → {saved_paths['pca']}")
        print(f"       SVM → {saved_paths['svm']}")
        print(f"[INFO] Saved history → {history_path}")

    return history


def save_model(model, model_name, history=None):
    os.makedirs("checkpoints", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{model_name}_{timestamp}"

    model_path = os.path.join("checkpoints", filename + ".pth")

    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Saved model to {model_path}")

    history_path = None
    if history is not None:
        history_path = os.path.join("checkpoints", filename + "_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
            print(f"[INFO] Saved history to {history_path}")

    return {
        "model": model_path,
        "history": history_path
    }


def load_model(model_class, path, device="cpu"):
    model = model_class().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded model from {path}")
    return model

def get_all_checkpoints(model_name):
    pairs = []
    for f in os.listdir("checkpoints"):
        if f.startswith(model_name) and f.endswith(".pth"):
            base = f[:-4]  # strip ".pth"
            checkpoint = os.path.join("checkpoints", f)
            history = os.path.join("checkpoints", base + "_history.json")
            pairs.append((checkpoint, history))

    print([p[0] for p in pairs])
    print([p[1] for p in pairs])
    return pairs

def get_classic_checkpoints(model_name):
    folder = "checkpoints"
    prefixes = set()

    for filename in os.listdir(folder):
        if not filename.startswith(model_name + "_"):
            continue

        if filename.endswith("_history.json"):
            stem = filename[:-len("_history.json")]
        elif filename.endswith("_pca.joblib"):
            stem = filename[:-len("_pca.joblib")]
        elif filename.endswith("_scaler.joblib"):
            stem = filename[:-len("_scaler.joblib")]
        elif filename.endswith("_svm.joblib"):
            stem = filename[:-len("_svm.joblib")]
        else:
            continue

        parts = stem.split("_")
        prefix = parts[0] + "_" + parts[1]
        prefixes.add(prefix)

    return prefixes

def find_latest_checkpoint(model_name: str):
    pattern = os.path.join("checkpoints", f"{model_name}_*.pth")
    files = glob.glob(pattern)

    if not files:
        print(f"[INFO] No checkpoint found for {model_name}")
        return None

    files.sort()

    latest = files[-1]
    print(f"[INFO] Latest checkpoint for {model_name}: {latest}")

    return latest

def find_latest_svm_checkpoint(model_name: str):
    pattern = os.path.join("checkpoints", f"*_{model_name}_*_pca.joblib")
    pca_files = glob.glob(pattern)

    if not pca_files:
        print(f"[INFO] No SVM checkpoint found for {model_name}")
        return None

    pca_files.sort(reverse=True)

    for pca_file in pca_files:
        prefix = pca_file.replace("_pca.joblib", "")
        svm_file = prefix + "_svm.joblib"

        if os.path.exists(svm_file):
            print(f"[INFO] Latest SVM checkpoint for {model_name}: {prefix}")
            return prefix

    print(f"[INFO] Found PCA files but no matching SVM for {model_name}")
    return None

def confusion_matrix(model, dataloader, num_classes=100, device="cpu"):
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for t, p in zip(labels, preds):
                matrix[t.long(), p.long()] += 1

    return matrix

def get_history_seed(path):
    with open(path) as f:
        history = json.load(f)
    if "seed" in history:
        return history["seed"]
    else:
        print(f"[ERROR]: \"seed\" key not found in {path}! Aborting!")
