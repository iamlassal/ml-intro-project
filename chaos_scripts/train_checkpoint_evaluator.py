import json
import argparse
import os

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Model")
    args = parser.parse_args()

    if args.m:
        checkpoint_dir = "checkpoints"

        files = [f for f in os.listdir(checkpoint_dir) if f.endswith("_history.json")]
        print("\n# ", args.m)

        histories = []
        for filename in files:
            parts = filename.split("_")
            if args.m in parts and filename.endswith("_history.json"):
                path = os.path.join(checkpoint_dir, filename)
                with open(path, "r") as f:
                    histories.append(json.load(f))

        grouped = {}
        for h in histories:
            seed = h.get("seed", "unknown")
            grouped.setdefault(seed, []).append(h)

        for seed, seed_histories in grouped.items():
            print(f"## Seed {seed}\n### Training")

            for history in seed_histories:
                training_time = history.get("total_time")
                a_buf = 0
                l_buf = 1e9

                best_acc = None
                best_loss = None

                val_losses = history.get("val_loss", [])
                val_accs = history.get("val_acc", [])

                epochs = min(len(val_losses), len(val_accs))

                for i in range(epochs):
                    loss = val_losses[i]
                    acc = val_accs[i]

                    if loss < l_buf:
                        l_buf = loss
                        best_loss = {
                            "epoch": i+1,
                            "loss": loss,
                            "acc_at_loss": acc
                        }

                    if acc > a_buf:
                        a_buf = acc
                        best_acc = {
                            "epoch": i+1,
                            "acc": acc,
                            "loss_at_acc": loss
                        }

                print(f"  Training time: {history.get("total_time")}")
                if best_loss:
                    print(f"- Lowest loss epoch {best_loss['epoch']}:")
                    print(f"\t- loss         : {best_loss['loss']:.4f}")
                    print(f"\t- val acc      : {best_loss['acc_at_loss']:.4f}")

                if best_acc:
                    print(f"- Highest acc epoch {best_acc['epoch']}:")
                    print(f"\t- val acc      : {best_acc['acc']:.4f}")
                    print(f"\t- loss         : {best_acc['loss_at_acc']:.4f}")


            print("\n### Testing\n")
        print("\n### Thoughts")

        best_seed_accs = []
        best_seed_losses = []
        total_times = [] # Initialized new list

        for seed, seed_histories in grouped.items():
            seed_best_acc = 0
            seed_best_loss = 1e9

            # Collect total_time for the seed (using the time from the first history)
            if seed_histories and "total_time" in seed_histories[0]:
                total_times.append(seed_histories[0]["total_time"])

            for history in seed_histories:
                val_accs = history.get("val_acc", [])
                val_losses = history.get("val_loss", [])

                if val_accs:
                    m_acc = max(val_accs)
                    if m_acc > seed_best_acc:
                        seed_best_acc = m_acc

                if val_losses:
                    m_loss = min(val_losses)
                    if m_loss < seed_best_loss:
                        seed_best_loss = m_loss

            best_seed_accs.append(seed_best_acc)
            best_seed_losses.append(seed_best_loss)

        n = len(best_seed_accs)
        mean_acc = sum(best_seed_accs) / n
        sum_sq = 0
        for a in best_seed_accs:
            sum_sq += (a - mean_acc) ** 2
        sd_acc = (sum_sq / (n - 1)) ** 0.5 if n > 1 else 0
        se_acc = sd_acc / (n ** 0.5)

        mean_loss = sum(best_seed_losses) / n
        sum_sq = 0
        for l in best_seed_losses:
            sum_sq += (l - mean_loss) ** 2
        sd_loss = (sum_sq / (n - 1)) ** 0.5 if n > 1 else 0
        se_loss = sd_loss / (n ** 0.5)

        # Calculate mean and SE for total_time
        n_time = len(total_times)
        if n_time > 0:
            mean_time = sum(total_times) / n_time
            sum_sq = 0
            for t in total_times:
                sum_sq += (t - mean_time) ** 2
            sd_time = (sum_sq / (n_time - 1)) ** 0.5 if n_time > 1 else 0
            se_time = sd_time / (n_time ** 0.5)
        else:
            mean_time, se_time = 0.0, 0.0

        print("### Combined across all seeds")
        print(f"\t- Best val accuracy: {mean_acc:.4f} ± {se_acc:.4f}")
        print(f"\t- Lowest val loss  : {mean_loss:.4f} ± {se_loss:.4f}")
        if n_time > 0:
             print(f"\t- Training time    : {mean_time:.2f} ± {se_time:.2f}")
