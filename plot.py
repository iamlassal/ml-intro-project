import json
import argparse
import os
import matplotlib.pyplot as plt

def load_history(path):
    path = os.path.join("checkpoints", f"{path}_history.json")
    with open(path, "r") as f:
        history = json.load(f)
    return history

def plot_histories(histories, labels=None, title="Training Curves"):
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(histories))]

    epochs = range(1, len(histories[0]["train_loss"]) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for hist, label in zip(histories, labels):
        plt.plot(epochs, hist["train_loss"], linestyle="--", alpha=0.6)
        plt.plot(epochs, hist["val_loss"], label=f"{label}", alpha=0.9)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    for hist, label in zip(histories, labels):
        plt.plot(epochs, hist["train_acc"], linestyle="--", alpha=0.6)
        plt.plot(epochs, hist["val_acc"], label=f"{label}", alpha=0.9)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_single_history(history, title="Training Curve"):
    plot_histories([history], labels=["Run"], title=title)

def list_history_files():
    checkpoint_dir = "checkpoints"
    return [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith("_history.json")
    ]

def extract_model_prefix(filename):
    parts = filename.split("_")
    return parts[0]


def extract_timestamp(filename):
    parts = filename.split("_")
    return parts[1]

def plot_multiple_histories(histories, labels):
    if labels is None:
            labels = [f"Model {i+1}" for i in range(len(histories))]

    plt.figure(figsize=(12, 5))

    epochs = range(1, len(histories[0]["train_loss"]) + 1)

    for history, label in zip(histories, labels):
        plt.plot(epochs, history["val_acc"], label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Model Comparison (Validation Accuracy)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=str, help="Path to history json")
    parser.add_argument("--multiple", nargs="+", help="List of history paths")
    parser.add_argument("--labels", nargs="+", help="List of labels")
    parser.add_argument("--list", action="store_true", help="List all history files in checkpoints/")
    parser.add_argument("--compare-latest", action="store_true", help="Compare the newest run of every model type")
    parser.add_argument("--latest", nargs="?", const=True, help="Show newest run for a model name, or list all model names if empty")
    parser.add_argument("--compare-all", type=str, help="Compare ALL runs for a given model name (e.g., BaselineCNN_ModelA)")
    args = parser.parse_args()

    if args.list:
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            print("[INFO] No checkpoints directory found.")
            exit(0)

        files = os.listdir(checkpoint_dir)
        history_files = [f for f in files if f.endswith("_history.json")]

        if not history_files:
            print("[INFO] No history files found in checkpoints/.")
        else:
            print("[INFO] History files found:")
            for f in history_files:
                m = f.replace("_history.json", "")
                print( f"  - {m}")

        exit(0)

    if args.compare_latest:
        files = list_history_files()
        if not files:
            print("[INFO] No history files found.")
            exit(0)

        groups = {}
        for f in files:
            prefix = extract_model_prefix(f)
            groups.setdefault(prefix, []).append(f)

        selected_paths = []
        for prefix, items in groups.items():
            newest = sorted(items, key=extract_timestamp)[-1]
            selected_paths.append(os.path.join("checkpoints", newest))

        print("[INFO] Comparing newest runs of all model types:")
        for p in selected_paths:
            print("  -", p)

        histories = [load_history(p) for p in selected_paths]
        labels = [os.path.basename(p).replace("_history.json", "") for p in selected_paths]

        plot_multiple_histories(histories, labels)
        exit(0)

    if args.latest is not None:
        files = list_history_files()
        if not files:
            print("[INFO] No history files found.")
            exit(0)

        if args.latest is True:
            prefixes = sorted(set(extract_model_prefix(f) for f in files))
            print("[INFO] Available model names:")
            for p in prefixes:
                print("  -", p)
            exit(0)

        model_name = args.latest
        matching = [f for f in files if extract_model_prefix(f) == model_name]

        if not matching:
            print(f"[ERROR] No history files match model '{model_name}'.")
            print("Use '--latest' with no arguments to list model names.")
            exit(1)

        newest = sorted(matching, key=extract_timestamp)[-1]
        path = os.path.join("checkpoints", newest)

        print(f"[INFO] Showing latest run for model '{model_name}':")
        print("  -", path)

        history = load_history(path)
        plot_single_history(history, title=model_name)
        exit(0)

    if args.single:
        history = load_history(args.single)
        plot_single_history(history, title=args.labels[0] if args.labels else "Model")

    elif args.multiple:
        histories = [load_history(p) for p in args.multiple]
        plot_multiple_histories(histories, args.labels)

    if args.compare_all:
        model_prefix = args.compare_all

        files = list_history_files()

        matching = [f for f in files if extract_model_prefix(f) == model_prefix]

        if not matching:
            print(f"[ERROR] No history files found for model name '{model_prefix}'.")
            print("Use --list to view available prefixes.")
            exit(1)

        matching = sorted(matching, key=extract_timestamp)

        print(f"[INFO] Found {len(matching)} runs for model '{model_prefix}':")
        for m in matching:
            print("  -", m.replace("_history.json", ""))

        histories = [load_history(m.replace("_history.json", "")) for m in matching]
        labels = [f"Run {i+1}" for i in range(len(histories))]

        plot_histories(histories, labels, title=f"All runs for {model_prefix}")
        exit(0)
