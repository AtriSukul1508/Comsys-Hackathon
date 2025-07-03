import torch
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, loader,device, type='train',class_names=None, plot_cm=True,save_dir='plots'):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Class labels
    if class_names is None:
        class_names = [str(i) for i in sorted(np.unique(y_true))]

    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    print(f"Accuracy:, {acc:.4f}")
    print(f"Precision:, {prec:.4f}")
    print(f"Recall:, {rec:.4f}")
    print(f"F1 Score:, {f1:.4f}")
    
    # Confusion Matrix
    if plot_cm:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        # plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    cm_plot_path = os.path.join(save_dir, f'{type}_confusion_matrix.png')
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix saved to {cm_plot_path}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "y_true": y_true,
        "y_pred": y_pred
    }
