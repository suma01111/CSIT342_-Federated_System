import csv
import json
import os
import matplotlib.pyplot as plt

# Secure federated results visualization.
# This script reads the secure FL results CSV and JSON summary,
# then plots accuracy growth and privacy configuration across rounds.

RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "federated_results.csv")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "federated_results_summary.json")


def load_csv_results(path):
    rounds = []
    accuracies = []
    local_dp = []
    global_dp = []
    secure_agg = []
    encryption = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            accuracies.append(float(row["accuracy"]))
            local_dp.append(float(row.get("local_dp_std", 0.0)))
            global_dp.append(float(row.get("global_dp_std", 0.0)))
            secure_agg.append(row.get("secure_aggregation", "False") in ["True", "true", "1"])
            encryption.append(row.get("encryption", "False") in ["True", "true", "1"])

    return rounds, accuracies, local_dp, global_dp, secure_agg, encryption


def load_summary(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_accuracy(rounds, accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracies, marker="o", linestyle="-", color="#1f77b4")
    plt.title("Secure Federated Learning: Accuracy by Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig("secure_federated_accuracy.png")
    plt.close()
    print("Saved secure_federated_accuracy.png")


def plot_privacy_parameters(rounds, local_dp, global_dp):
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, local_dp, marker="o", label="Local DP sigma", color="#ff7f0e")
    plt.plot(rounds, global_dp, marker="s", label="Global DP sigma", color="#2ca02c")
    plt.title("Privacy Parameters Across Rounds")
    plt.xlabel("Round")
    plt.ylabel("Noise Sigma")
    plt.legend()
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig("secure_federated_privacy_params.png")
    plt.close()
    print("Saved secure_federated_privacy_params.png")


def plot_secure_flags(rounds, secure_agg, encryption):
    secure_agg_numeric = [1 if flag else 0 for flag in secure_agg]
    encryption_numeric = [1 if flag else 0 for flag in encryption]

    plt.figure(figsize=(8, 3.5))
    plt.step(rounds, secure_agg_numeric, where="mid", label="Secure Aggregation Enabled", color="#d62728")
    plt.step(rounds, encryption_numeric, where="mid", label="Encryption Enabled", color="#9467bd")
    plt.yticks([0, 1], ["Disabled", "Enabled"])
    plt.title("Secure Mechanism Flags by Round")
    plt.xlabel("Round")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("secure_federated_flags.png")
    plt.close()
    print("Saved secure_federated_flags.png")


def main():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV}")

    rounds, accuracies, local_dp, global_dp, secure_agg, encryption = load_csv_results(RESULTS_CSV)
    summary = load_summary(SUMMARY_JSON)

    plot_accuracy(rounds, accuracies)
    plot_privacy_parameters(rounds, local_dp, global_dp)
    plot_secure_flags(rounds, secure_agg, encryption)

    if summary is not None:
        best = summary.get("best_round")
        if best:
            print(f"Best round: {best['round']} with accuracy {best['accuracy']}")
        print("Privacy configuration:")
        print(json.dumps(summary.get("privacy_config", {}), indent=2))


if __name__ == "__main__":
    main()
