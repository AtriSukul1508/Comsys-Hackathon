import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder



train_dir = "../data/Task_B/train"
val_dir = "../data/Task_B/val"
test_dir = ""

def get_transforms():

    train_transform =  transforms.Compose([
        transforms.Resize((256, 256)),  # Upscale slightly
        transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),  # Random scale variation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform =  transforms.Compose([
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
    return train_transform, val_transform, test_transform



class SiameseFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_pairs_per_identity=5):
        self.root_dir = root_dir
        self.transform = transform
        self.num_pairs_per_identity = num_pairs_per_identity

        self.identity_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                                 if os.path.isdir(os.path.join(root_dir, d))]

        # Map: identity_name -> list of all images (clean + distorted)
        self.identity_images = self._load_identity_images()

        # Generate all pairs (balanced)
        self.pairs = self._create_pairs()

    def _load_identity_images(self):
        identity_images = {}
        for identity_path in self.identity_folders:
            images = []

            # Clean images
            for file in os.listdir(identity_path):
                file_path = os.path.join(identity_path, file)
                if file.endswith('.jpg') or file.endswith('.png'):
                    images.append(file_path)

            # Distorted images (if present)
            dist_path = os.path.join(identity_path, 'distortion')
            if os.path.exists(dist_path):
                for file in os.listdir(dist_path):
                    if file.endswith('.jpg') or file.endswith('.png'):
                        images.append(os.path.join(dist_path, file))

            if images:
                identity_images[identity_path] = images
        return identity_images

    def _create_pairs(self):
        positive_pairs = []
        negative_pairs = []

        identities = list(self.identity_images.keys())

        for identity in identities:
            imgs = self.identity_images[identity]
            if len(imgs) < 2:
                continue

            # Create positive pairs
            for _ in range(self.num_pairs_per_identity):
                img1, img2 = random.sample(imgs, 2)
                positive_pairs.append((img1, img2, 1))

            # Create negative pairs
            for _ in range(self.num_pairs_per_identity):
                other_identity = random.choice([i for i in identities if i != identity])
                img1 = random.choice(imgs)
                img2 = random.choice(self.identity_images[other_identity])
                negative_pairs.append((img1, img2, 0))

        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        return all_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2, label = self.pairs[idx]
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

def get_datasets_and_loaders(train_transform, val_transform,test_transform, train_data_dir=train_dir,val_data_dir=val_dir, test_data_dir=test_dir):

    train_dataset = SiameseFaceDataset(train_data_dir, transform=train_transform)
    val_dataset = SiameseFaceDataset(val_data_dir, transform=val_transform)
    test_dataset = SiameseFaceDataset(test_data_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total validation samples: {len(val_loader.dataset)}")
    print(f"Total testing samples: {len(test_loader.dataset)}")
    return train_loader,val_loader,test_loader
