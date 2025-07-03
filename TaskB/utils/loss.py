import torch.nn as nn
import torch.nn.functional as F

class SiameseHybridLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, emb1, emb2, label):
        # Normalize embeddings
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        # ----------- Contrastive Loss -----------
        euclidean_dist = F.pairwise_distance(emb1, emb2, p=2)
        contrastive_loss = label * euclidean_dist.pow(2) + \
                           (1 - label) * F.relu(self.margin - euclidean_dist).pow(2)
        contrastive_loss = contrastive_loss.mean()

        # ----------- Cosine BCEWithLogits Loss -----------
        cosine_sim = F.cosine_similarity(emb1, emb2)  # [-1, 1]
        cosine_logits = cosine_sim * 5  # Stretch to make logits more confident
        cosine_loss = self.bce_logits(cosine_logits, label)

        # ----------- Final Weighted Loss -----------
        total_loss = self.alpha * contrastive_loss + (1 - self.alpha) * cosine_loss
        return total_loss
