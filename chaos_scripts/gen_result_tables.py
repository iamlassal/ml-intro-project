import os
import json

model_names = [
# "ModelA", "ModelB", "ModelC", "ModelD", "ModelE",
# "ModelF", "ModelG", "ModelH", "ModelT", "ModelCL",
# "ModelX",
"ModelMML"
]

metric_keys = {
    "loss": "test_loss",
    "acc": "test_acc",
    "top5_acc": "test_top5_acc",
    "macro_f1": "macro_f1",
    "macro_precision": "macro_precision",
    "macro_recall": "macro_recall"
}

def sterr(values):
    n = len(values)
    mean = sum(values) / n
    sum_sq = 0
    for v in values:
        sum_sq += (v - mean) ** 2
    sd = (sum_sq / (n - 1)) ** 0.5 if n > 1 else 0
    se = sd / (n ** 0.5) if n > 0 else 0
    return mean, se

if __name__ == "__main__":
    filenames = [f for f in os.listdir("results") if f.endswith("_EVAL_RESULTS.json")]
    summary_by_model = {}

    for name in model_names:
        model_results = []
        for filename in filenames:
            parts = filename.split("_")
            if name in parts:
                path = os.path.join("results", filename)
                with open(path, "r") as f:
                    model_results.append(json.load(f))

        if not model_results:
            summary_by_model[name] = {}
            continue

        field_map = {}
        for key, json_key in metric_keys.items():
            field_map[key] = []
            for r in model_results:
                if json_key in r and not (name == "ModelCL" and key == "loss"):
                    field_map[key].append(r[json_key])

        metric_summary = {}
        for key, values in field_map.items():
            clean = []
            for v in values:
                if v is not None:
                    clean.append(v)

            if clean:
                mean, se = sterr(clean)
                metric_summary[key] = (mean, se)
            else:
                metric_summary[key] = ("N/A", 0)

        summary_by_model[name] = metric_summary

    t1 = {
        "Acc (±SE)": [],
        "Top-5 (±SE)": [],
        "Loss (±SE)": []
    }

    for model, sm in summary_by_model.items():
        def cell(k, show_loss=True): # pyright: ignore
            val = sm.get(k)
            if not val or val[0] == "N/A":
                return "N/A"
            mean, se = val
            return f"${mean:.3f} \\pm {se:.3f}$"

        t1["Acc (±SE)"].append((model, cell("acc", False)))
        t1["Top-5 (±SE)"].append((model, cell("top5_acc", False)))
        t1["Loss (±SE)"].append((model, cell("loss", False)))

    # Table 2
    t2 = {
        "macro_f1": [],
        "macro_precision": [],
        "macro_recall": []
    }

    for model, sm in summary_by_model.items():
        def cell(k):
            val = sm.get(k)
            if not val or val[0] == "N/A":
                return "N/A"
            mean, se = val
            return f"${mean:.3f} \\pm {se:.3f}$"

        t2["macro_f1"].append((model, cell("macro_f1")))
        t2["macro_precision"].append((model, cell("macro_precision")))
        t2["macro_recall"].append((model, cell("macro_recall")))

    print("\\subsection*{Test Results (Accuracy & Loss)}")
    print("\\small\\setlength{\\tabcolsep}{3pt}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Model & Accuracy (±SE) & Top-5 (±SE) & Loss (±SE)\\\\")
    print("\\midrule")
    for model, acc_cell in [x for x in t1["Acc (±SE)"]]:
        top5 = next(val for m, val in t1["Top-5 (±SE)"] if m == model)
        loss = next(val for m, val in t1["Loss (±SE)"] if m == model)
        print(f"{model} & {acc_cell} & {top5} & {loss}\\\\")
    print("\\bottomrule")
    print("\\end{tabular}\\normalsize\n")

    print("\\subsection*{Test Results (Macro Metrics)}")
    print("\\small\\setlength{\\tabcolsep}{3pt}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Model & Macro F1 (±SE) & Precision (±SE) & Recall (±SE)\\\\")
    print("\\midrule")
    for model, _ in t2["macro_f1"]:
        f1   = next(val for m, val in t2["macro_f1"] if m == model)
        prec = next(val for m, val in t2["macro_precision"] if m == model)
        rec  = next(val for m, val in t2["macro_recall"] if m == model)
        print(f"{model} & {f1} & {prec} & {rec}\\\\")
    print("\\bottomrule")
    print("\\end{tabular}\\normalsize\n")
