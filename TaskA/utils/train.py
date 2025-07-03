import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import os

def train_model(
    model, train_loader, val_loader, device,
    criterion=None,
    optimizer=None,
    scheduler=None,
    epochs=20,
    patience=7,
    max_grad_norm=1.0,
    label_smoothing=0.1,
    use_amp=True,
    save_path='best_model.pth'
):
    
    model.to(device)
    scaler = GradScaler() if use_amp else None

    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    trigger_times = 0

    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training : ", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        # ---------- Validation ----------
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validating : ", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f" Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if scheduler:
            scheduler.step(val_loss)

        # ----------- Early Stopping ---------- 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            trigger_times = 0
            # Saving the best model
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} with best validation loss: {best_val_loss:.4f}")
        else:
            trigger_times += 1
            print(f"Early Stopping Warning: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("Early stopping activated.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return history
