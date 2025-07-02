import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import os

def test_model(model, test_loader, device, criterion=nn.CrossEntropyLoss(), save_path='best_model.pth'):

    if not os.path.exists(save_path):
        print(f"Error: Model weights not found at {save_path}. Please ensure training completed successfully.")
        return None, None

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()

    test_loss = 0
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing : ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"\n--- Test Results ---")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    return test_loss, test_acc
