import csv
import matplotlib.pyplot as plt

# ==========================
# FEDERATED GRAPH
# ==========================
rounds = []
accs = []

with open("results/federated_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rounds.append(int(row["round"]))
        accs.append(float(row["accuracy"]))

plt.figure()
plt.plot(rounds, accs, marker="o")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Federated Accuracy vs Round")
plt.savefig("federated_curve.png")
plt.close()

print("Federated graph saved!")
