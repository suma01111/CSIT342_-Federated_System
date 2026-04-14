import matplotlib.pyplot as plt

# YOUR RESULTS
local_acc = 0.8125
fed_acc = 0.8875   # best from your JSON

labels = ["Local", "Federated"]
values = [local_acc, fed_acc]

plt.figure()
plt.bar(labels, values)
plt.ylabel("Accuracy")
plt.title("Local vs Federated Performance")
plt.savefig("comparison.png")
plt.close()

print("Comparison graph saved!")
