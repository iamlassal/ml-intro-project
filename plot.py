import json
import argparse
import os
import re
import matplotlib.pyplot as plt

def load_history(path):
    path = os.path.join("checkpoints", f"{path}_history.json")
    with open(path, "r") as f:
        history = json.load(f)
    return history

def load_results(model_name):
    files = list_result_files()
    results = []
    for filename in files:
        if model_name in filename.split("_"):
            path = os.path.join("results", filename)
            m = re.search(r"SEED_(\d+)", filename)
            seed = m.group(1) # pyright: ignore
            print(seed)
            with open(path, "r") as f:
                result = json.load(f)
                results.append((seed, result))
    return results

def cm_heatmap(model_name, filename=""):
    fig, ax = plt.subplots(2,2, figsize=(12, 8))
    ax = ax.flatten()
    results = load_results(model_name)

    for i, (seed, data) in enumerate(results):
        heatmap = data["test_conf_matrix"]
        ax[i].imshow(heatmap)
        ax[i].set_title(f"Seed: {seed}")

    avg = [[0.0 for j in range(100)] for i in range(100)]

    for _, data in results:
        for i in range(100):
            for j in range(100):
                avg[i][j] += data["test_conf_matrix"][i][j]

    for i in range(100):
        for j in range(100):
            avg[i][j] /= 3

    ax[3].imshow(avg)
    ax[3].set_title("Average")
    plt.tight_layout()
    plt.suptitle(model_name)
    if filename != "":
        plt.savefig(f"{filename}.pdf")
    else:
        plt.show()

def plot_histories(histories, title="Training Curves", test_set=False, filename=""):
    epochs = range(1, len(histories[0]["train_loss"]) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for i, hist in enumerate(histories):
        if test_set:
            plt.plot(epochs, hist["train_loss"], linestyle="--", alpha=0.6, color=f"C{i}")
        plt.plot(epochs, hist["val_loss"], label=f"Seed {hist["seed"]}", alpha=0.9,color=f"C{i}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)


    plt.subplot(1, 2, 2)
    for i, hist in enumerate(histories):
        if test_set:
            plt.plot(epochs, hist["train_acc"], linestyle="--", alpha=0.6, color=f"C{i}")
        plt.plot(epochs, hist["val_acc"], label=f"Seed {hist["seed"]}", alpha=0.9, color=f"C{i}")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.grid(True)
    if filename != "":
        plt.savefig(f"{filename}.pdf")
    else:
        plt.show()

def plot_single_history(history, title="Training Curve", filename=""):
    plot_histories([history], title=title, filename=filename if args.save else "")

def list_history_files():
    checkpoint_dir = "checkpoints"
    return [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith("_history.json")
    ]

def list_result_files():
    return [
        f for f in os.listdir("results")
        if f.endswith("_EVAL_RESULTS.json")
    ]

def extract_model_prefix(filename):
    parts = filename.split("_")
    return parts[0]


def extract_timestamp(filename):
    parts = filename.split("_")
    return parts[1]

def plot_multiple_histories(histories, labels, filename=""):
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
    plt.grid(True)
    if filename != "":
       plt.savefig(f"{filename}.pdf")
    else:
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
    parser.add_argument("--test-set", action="store_true", help="Include test history when comparing all runs for a given model name.")
    parser.add_argument("--save", type=str, help="Save plot to a PDF.")
    parser.add_argument("--cm", type=str, help="Show the confusion matrices of an evaluated model.")
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

        plot_multiple_histories(histories, labels=None, filename=args.save if args.save else "")
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
        plot_single_history(history, title=model_name, filename=args.save if args.save else "")
        exit(0)

    if args.single:
        history = load_history(args.single)
        plot_single_history(history, title=args.labels[0] if args.labels else "Model", filename=args.save if args.save else "")

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

        plot_histories(histories, title=f"All runs for {model_prefix}", test_set=args.test_set, filename=args.save if args.save else "")
        exit(0)

    if args.cm:
        cm_heatmap(args.cm, filename=args.save if args.save else "")
