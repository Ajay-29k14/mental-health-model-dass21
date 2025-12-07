import matplotlib.pyplot as plt

# Replace these with your actual accuracies
acc_depression = 0.8409
acc_anxiety = 0.7614
acc_stress = 0.7727

labels = ["Depression", "Anxiety", "Stress"]
accuracies = [acc_depression, acc_anxiety, acc_stress]

plt.figure(figsize=(6, 4))
plt.bar(labels, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # scale from 0 to 1 (0% to 100%)

# Show values on top of bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
