import os
import json
import argparse
import matplotlib.pyplot as plt

metric_dict = {
    "train_loss": "Training Loss",
    "train_acc": "Training Accuracy",
    "val_loss": "Validation Loss",
    "val_acc": "Validation Accuracy"
}

model_desc_dict = {
    "ModelA": "Model A - Baseline CNN",
    "ModelB": "Model B - Baseline CNN + Data Augmentation",
    "ModelC": "Model C - Baseline CNN + Batch Normalization",
    "ModelD": "Model D - Baseline CNN + Dropout p=0.5",
    "ModelE": "Model E - Baseline CNN + Batch Normalization + Data Augmentation",
    "ModelF": "Model F - Baseline CNN + Batch Normalization + Dropout + Data Augmentation",
    "ModelG": "Model G - Deep CNN",
    "ModelH": "Model H - Wide CNN",
    "ModelMML": "Model MML - BaselineCNN \\w MultiMarginLoss",
    "ModelT": "Model T - ResNet-18 Transfer Model",
    "ModelX": "Model X - DeepWide CNN"
}

def stderr(values):
    n = len(values)
    mean = sum(values) / n
    sum_sq = 0
    for v in values:
        sum_sq += (v - mean) ** 2
    sd = (sum_sq / (n - 1)) ** 0.5 if n > 1 else 0
    se = sd / (n ** 0.5) if n > 0 else 0
    return mean, se

def load_all_histories(model_name):
    checkpoint_dir = "checkpoints"
    all_filenames =  [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith("_history.json")
    ]

    histories = []
    for filename in all_filenames:
        if model_name in filename.split("_"):
            path = os.path.join("checkpoints", filename)
            with open(path, "r") as f:
                history = json.load(f)
                histories.append(history)
    return histories

def average_histories(histories, metric):
    timelines = []
    for history in histories:
        timelines.append(history[metric])

    n = len(histories[0][metric])

    mean_line = []
    se_line = []

    for i in range(n):
        values = []
        for curve in timelines:
            values.append(curve[i])

        mean, se = stderr(values)
        mean_line.append(mean)
        se_line.append(se)

    return mean_line, se_line

def plot(line_pairs, labels, metric, save=False):
    plt.figure(figsize=(12, 5))

    x = list(range(len(line_pairs[0][0])))
    desc = []
    for label in labels:
        desc.append(model_desc_dict[label])

    for (mean_line, se_line), label in zip(line_pairs, labels):
        plt.fill_between(
            x,
            [mean_line[i] - se_line[i] for i in range(len(x))],
            [mean_line[i] + se_line[i] for i in range(len(x))],
            alpha=0.3,
        )
        plt.plot(mean_line, label=model_desc_dict[label])

    plt.xlabel("Epoch")
    plt.ylabel(metric_dict[metric])
    plt.title(f"Comparison of {metric_dict[metric]} between {", ".join(desc)}" if len(desc) <= 2 else f"{metric_dict[metric]} comparison of multiple CNN Models")
    plt.legend(
        ncol=1,
        loc="lower right",
        fontsize=7,
        borderpad=0.3,
        framealpha=1,
    )
    plt.grid(True)
    if save:
        dir = "plot_avg"
        if not os.path.exists(dir):
            os.mkdir(dir)
        plt.savefig(os.path.join(dir, f"{metric}_{"_".join(labels)}.pdf"), bbox_inches="tight", dpi=300)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="List of models")
    parser.add_argument("--metric", type=str, help="Metric to compare")
    parser.add_argument("--save", action="store_true", help="Save the graph to hard drive")
    args = parser.parse_args()

    if args.models and args.metric:
        pairs = []
        labels = []
        for model in args.models:
            hist = load_all_histories(model)
            p = average_histories(hist, args.metric)
            pairs.append(p)
            labels.append(model)

        plot(pairs, labels, args.metric, args.save)
