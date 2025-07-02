import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):

    model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.head.in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    return model

if __name__ == '__main__':
    model = get_model(num_classes=2)
    print(model)
