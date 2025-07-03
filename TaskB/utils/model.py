import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.backbone = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, embedding_dim)
        )

    def forward_once(self, x):
        emb = self.backbone(x)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)