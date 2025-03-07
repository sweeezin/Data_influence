# Plot the Precision-Recall curve in isolation
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Sample data to test plotting
precision = [0.1, 0.2, 0.3, 0.4, 0.5]
recall = [0.5, 0.4, 0.3, 0.2, 0.1]
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()  # Explicitly show the plot
