import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from collections import Counter
import os


train_dir ='../data/Task_A/train'
val_dir = '../data/Task_A/val'
test_dir = ''
def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ], p=0.2),

        transforms.RandomApply([transforms.RandomRotation(20)], p=0.2),
        
        transforms.RandomApply([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15))
        ], p=0.2),
        
        transforms.RandomGrayscale(p=0.1),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ], p=0.1),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])



    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform ,test_transform

def get_datasets_and_loaders(train_transform, val_transform,test_transform, train_data_dir=train_dir,val_data_dir=val_dir, test_data_dir=test_dir,batch_size=32):

    train_dataset = ImageFolder(train_data_dir, transform=train_transform)
    val_dataset = ImageFolder(val_data_dir, transform=val_transform)
    test_dataset = ImageFolder(test_data_dir,transform=test_transform)
    sampler, class_weights = compute_sampler_and_weights(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total validation samples: {len(val_loader.dataset)}")
    print(f"Total testing samples: {len(test_loader.dataset)}")

    return train_loader, val_loader, train_dataset, val_dataset ,test_loader,test_dataset

def compute_sampler_and_weights(train_dataset):


    targets = [label for _, label in train_dataset]
    class_counts = Counter(targets)
    print(f"Class counts: {class_counts}")  # e.g., Counter({0: 1500, 1: 300})

    class_weights = 1. / torch.tensor([class_counts[i] for i in sorted(class_counts)], dtype=torch.float)
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return sampler, class_weights

