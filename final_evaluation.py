import argparse
import json
import torch
import torch.nn as nn
import numpy as np

from utils import (
    load_model,
    confusion_matrix as util_confusion_matrix,
    get_all_checkpoints,
    get_classic_checkpoints,
    get_history_seed,
)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from datasets import get_cifar100, get_cifar100_transfer
from models.baseline import BaselineCNN
from models.batchdrop import BatchDropCNN
from models.batchnorm import BatchnormCNN
from models.deepbatchdrop import DeepBatchDropCNN
from models.dropout import DropoutCNN
from models.widebatchdrop import WideBatchDropCNN
from models.resnet_transfer import ResNet18TransferCNN
from models.svm_baseline import PCASVMClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

def require_confirmation(model_name, confirm_flag):
    if not confirm_flag:
        print("WARNING: You are about to evaluate the TEST SET.")
        print("Run again with '--confirm' to proceed.")
        exit(1)

    print("IMPORTANT: This will evaluate your model on the TEST SET.")
    print("You should ONLY run this once per model in the final experiment.")
    answer = input("Type 'yes' to continue: ").strip().lower()
    if answer != "yes":
        print("Aborting test evaluation.")
        exit(1)

    confirm_name = input(f"Type the model name ({model_name}) to confirm: ").strip()
    if confirm_name != model_name:
        print("Model name does not match. Aborting.")
        exit(1)

    print("\nTest evaluation confirmed.\n")

def topk_accuracy(logits, labels, k=5):
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
    return correct / labels.size(0)

def evaluate_model(model, dataloader, loss_fn, device="cpu"):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_top5 += topk_accuracy(logits, labels, k=5) * labels.size(0)
            total_samples += labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    test_conf = util_confusion_matrix(model, dataloader, num_classes=100, device=device)

    per_class_acc = (test_conf.diag() / test_conf.sum(dim=1)).tolist()

    eps = 1e-9
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for i in range(100):
        tp = test_conf[i, i].item()
        fn = test_conf[i, :].sum().item() - tp
        fp = test_conf[:, i].sum().item() - tp

        precision = tp / (tp + fp + eps)

        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_scores.append(f1)

    macro_precision = float(np.mean(precision_scores))
    macro_recall = float(np.mean(recall_scores))
    macro_f1 = float(np.mean(f1_scores))


    return {
        "test_loss": float(total_loss / total_samples),
        "test_acc": float(total_correct / total_samples),
        "test_top5_acc": float(total_top5 / total_samples),
        "per_class_accuracy": per_class_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "test_conf_matrix": test_conf.tolist(),
    }

def evaluate_classical(model, dataloader):
    X = []
    y = []

    for images, labels in dataloader:
        arr = images.numpy().reshape(images.shape[0], -1)
        X.append(arr)
        y.append(labels.numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    X_scaled = model.scaler.transform(X)
    X_pca = model.pca.transform(X_scaled)

    preds = model.svm.predict(X_pca)

    acc = accuracy_score(y, preds)

    decision = model.svm.decision_function(X_pca)
    top5 = np.argsort(decision, axis=1)[:, -5:]
    top5_correct = sum(1 for i in range(len(y)) if y[i] in top5[i])
    top5_acc = top5_correct / len(y)

    conf = sklearn_confusion_matrix(y, preds).tolist()

    conf_np = np.array(conf, dtype=np.int64)
    per_class_acc = (conf_np.diagonal() / np.maximum(conf_np.sum(axis=1), 1)).tolist()

    eps = 1e-9
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for i in range(100):
        tp = conf_np[i, i]
        fn = conf_np[i, :].sum() - tp
        fp = conf_np[:, i].sum() - tp

        precision = tp / (tp + fp + eps)

        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    macro_precision = float(np.mean(precision_scores))
    macro_recall = float(np.mean(recall_scores))
    macro_f1 = float(np.mean(f1_scores))

    return {
        "test_acc": float(acc),
        "test_top5_acc": float(top5_acc),
        "per_class_accuracy": per_class_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "test_conf_matrix": conf,
    }

def save_and_print_results(results, model_name, i, seed):
    save_path = f"results/{model_name}_{i}_SEED_{seed}_EVAL_RESULTS.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    return save_path

MODEL_MAP = {
    "ModelA": BaselineCNN,
    "ModelB": BaselineCNN,
    "ModelC": BatchnormCNN,
    "ModelD": DropoutCNN,
    "ModelE": BatchnormCNN,
    "ModelF": BatchDropCNN,
    "ModelG": DeepBatchDropCNN,
    "ModelH": WideBatchDropCNN,
    "ModelT": ResNet18TransferCNN,
    "ModelCL": PCASVMClassifier,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model name, e.g. ModelA, ModelH, ModelT")
    parser.add_argument("--confirm", action="store_true",
                        help="Required to allow test-set evaluation.")

    args = parser.parse_args()

    if args.model not in MODEL_MAP:
        raise ValueError(f"Unknown model: {args.model}")

    require_confirmation(args.model, args.confirm)

    model_class = MODEL_MAP[args.model]

    print(f"[INFO] Starting evaliation of model {args.model}")

    if args.model == "ModelCL":
        for i, (model_path) in enumerate(get_classic_checkpoints(args.model)):
            prefix = f"checkpoints/{model_path}"

            seed = get_history_seed(f"{prefix}_history.json")
            _, _, test_loader = get_cifar100(seed=seed, augment=False)

            model = PCASVMClassifier.load(prefix)
            ckpt_path = prefix
            print(f"[INFO] Loaded classical PCA+SVM model: {prefix}")

            results = evaluate_classical(model, test_loader)
            save_path = save_and_print_results(results, args.model, i, seed)

            print(f"[INFO] Saved evaluation results to {save_path}")

            print("\n==== Evaluation Summary ====")
            print(f"Top-1 Accuracy:         {results['test_acc']:.4f}")
            print(f"Top-5 Accuracy:         {results['test_top5_acc']:.4f}")
            print(f"Macro Precision Score:  {results['macro_precision']:.4f}")
            print(f"Macro Recall Score:     {results['macro_recall']:.4f}")
            print(f"Macro F1 Score:         {results['macro_f1']:.4f}")

    else:
        for i, (model_path, history_path) in enumerate(get_all_checkpoints(args.model)):
            seed = get_history_seed(history_path)

            if args.model == "ModelT":
                _, _, test_loader = get_cifar100_transfer(seed=seed, augment=False)
            else:
                _, _, test_loader = get_cifar100(seed=seed, augment=False)

            model = load_model(model_class, model_path, device=device)
            print(f"[INFO] Loaded CNN checkpoint: {model_path}")

            loss_fn = nn.CrossEntropyLoss()
            results = evaluate_model(model, test_loader, loss_fn, device=device)
            save_path = save_and_print_results(results, args.model, i, seed)

            print(f"[INFO] Saved evaluation results to {save_path}")

            print("\n==== Evaluation Summary ====")
            print(f"Top-1 Accuracy:         {results['test_acc']:.4f}")
            print(f"Top-5 Accuracy:         {results['test_top5_acc']:.4f}")
            print(f"Macro Precision Score:  {results['macro_precision']:.4f}")
            print(f"Macro Recall Score:     {results['macro_recall']:.4f}")
            print(f"Macro F1 Score:         {results['macro_f1']:.4f}")
            print(f"Test Loss:              {results['test_loss']:.4f}")
