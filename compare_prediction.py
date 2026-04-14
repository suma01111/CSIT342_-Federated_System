import matplotlib.pyplot as plt

# values from your terminal
local_conf = 0.7166
fed_conf = 0.5823

labels = ["Local", "Federated"]
values = [local_conf, fed_conf]

plt.figure()
plt.bar(labels, values)
plt.ylim(0, 1)
plt.ylabel("Confidence")
plt.title("Prediction Confidence Comparison")
plt.savefig("prediction_compare.png")
plt.close()

print("Prediction comparison graph saved!")
