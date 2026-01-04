"""
Utilitaires pour gérer le dataset d'images
"""
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


class DocumentDataset(Dataset):
    """
    Dataset personnalisé pour les documents
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Dossier racine (ex: data/processed/)
            transform: Transformations à appliquer
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Classes = noms des sous-dossiers
        self.classes = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Lister tous les fichiers
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            
            # Parcourir TOUS les sous-dossiers et fichiers
            for root, dirs, files in os.walk(class_dir):
                for img_name in files:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"📊 Dataset chargé:")
        print(f"  {len(self.samples)} images")
        print(f"  {len(self.classes)} classes: {self.classes}")
        
        for cls in self.classes:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[cls])
            print(f"    {cls}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True):
    """
    Transformations pour le dataset
    """
    if train:
        # Augmentation de données pour l'entraînement
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Pas d'augmentation pour validation/test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(data_dir, batch_size=8, train_split=0.7):
    """
    Crée les dataloaders train/val/test
    
    Args:
        data_dir: Dossier contenant les images organisées par classe
        batch_size: Taille des batches
        train_split: Proportion pour l'entraînement (0.7 = 70%)
        
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    from sklearn.model_selection import train_test_split
    
    # Charger dataset complet
    full_dataset = DocumentDataset(data_dir, transform=None)
    
    # Split indices
    indices = list(range(len(full_dataset)))
    labels = [label for _, label in full_dataset.samples]
    
    # Train / (Val + Test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - train_split),
        stratify=labels,
        random_state=42
    )
    
    # Val / Test
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )
    
    # Créer datasets avec transformations
    train_dataset = DocumentDataset(data_dir, transform=get_transforms(train=True))
    val_dataset = DocumentDataset(data_dir, transform=get_transforms(train=False))
    test_dataset = DocumentDataset(data_dir, transform=get_transforms(train=False))
    
    # Subsets
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)
    test_dataset = Subset(test_dataset, test_idx)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Important pour Windows
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"\n📦 DataLoaders créés:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader, full_dataset.classes


# Test
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        "../data/processed/",
        batch_size=8
    )
    
    print(f"\nClasses: {classes}")
    print(f"\n✅ Dataset prêt pour l'entraînement!")