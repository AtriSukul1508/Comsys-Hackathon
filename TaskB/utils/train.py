# src/train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_siamese_model(
    model, train_loader, val_loader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=20,
    patience=7,
    max_grad_norm=1.0,
    use_amp=True,
    cosine_threshold=0.5, # Threshold for similarity-based prediction
    save_path="best_model.pth"
):
    model.to(device)
    scaler = GradScaler() if use_amp else None

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    trigger_times = 0

    print(f"Training started on device: {device}")

    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        for img1, img2, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device.type):
                    emb1, emb2 = model(img1, img2)
                    loss = criterion(emb1, emb2, labels)
                scaler.scale(loss).backward()
                if max_grad_norm:
                    # Unscale the gradients before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                emb1, emb2 = model(img1, img2)
                loss = criterion(emb1, emb2, labels)
                loss.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            train_loss += loss.item() * img1.size(0)

            # Cosine similarity-based prediction for accuracy
            cosine_sim = F.cosine_similarity(emb1, emb2)
            preds = (cosine_sim > cosine_threshold).float()

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, np.round(train_preds))

        # ---------- Validation ----------
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []

        with torch.no_grad():
            for img1, img2, labels in tqdm(val_loader, desc="Validating", leave=False):
                img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)
                emb1, emb2 = model(img1, img2)
                loss = criterion(emb1, emb2, labels)

                val_loss += loss.item() * img1.size(0)

                # Embeddings are already normalized
                cosine_sim = F.cosine_similarity(emb1, emb2)
                preds = (cosine_sim > cosine_threshold).float()

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, np.round(val_preds))

        # ---------- Logging ----------
        print(f" Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if scheduler:
            scheduler.step(val_loss)

        # ---------- Early Stopping ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            print(f"Model saved to {save_path} with best validation loss: {best_val_loss:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"Early Stopping Warning: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("Early stopping activated.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return history

def plot_training_curves(history, save_dir='plots'):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")
    # plt.show()

def evaluate_siamese_model(model, loader, threshold=0.5, plot_cm=True,save_dir='plots'):

    model.to(device)
    model.eval()
    y_true, y_pred = [], []

    print(f"Evaluating model on device: {device}")

    with torch.no_grad():
        for img1, img2, labels in tqdm(loader, desc="Evaluating", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

            emb1, emb2 = model(img1, img2)

            cosine_sim = F.cosine_similarity(emb1, emb2)
            preds = (cosine_sim > threshold).int()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    print("\nClassification Report:\n", classification_report(
        y_true, y_pred, target_names=["Different", "Same"]))
    print(f"Accuracy:, {acc:.4f}")
    print(f"Precision:, {prec:.4f}")
    print(f"Recall:, {rec:.4f}")
    print(f"F1 Score:, {f1:.4f}")

    if plot_cm:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Different", "Same"],
                    yticklabels=["Different", "Same"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        # plt.show()
        plot_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(plot_path)
        print(f"Training curves saved to {plot_path}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "y_true": y_true,
        "y_pred": y_pred
    }
