from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def test_model(model, dataloader, device, threshold=0.5,save_path='best_model.pth'):

    
    if not os.path.exists(save_path):
        print(f"Error: Model weights not found at {save_path}. Please ensure training completed successfully.")
        return None, None

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc="Testing"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()

            emb1,emb2 = model(img1,img2)

            similarity = F.cosine_similarity(emb1, emb2)

            preds = (similarity > threshold).float()

            all_scores.extend(similarity.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\nTest Accuracy : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")

    return accuracy, precision, recall, f1
