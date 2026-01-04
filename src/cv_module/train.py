"""
Script d'entraînement du modèle hybride - Optimisé pour CPU
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
from datetime import datetime
import sys

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv_module.hybrid_model import HybridDocumentClassifier
from cv_module.dataset_utils import create_dataloaders
from cv_module.gabarit_detector import GabaritDetector
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss et optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=3, 
            factor=0.5
        )
        
        # Gabarit detector
        self.gabarit_detector = GabaritDetector()
        
        # Historique
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
    
    def extract_gabarit_features(self, images):
        """
        Extrait features de gabarits pour un batch d'images
        
        Args:
            images: Tensor (batch, 3, 224, 224) NORMALISÉ
            
        Returns:
            Tensor (batch, 9) avec features de gabarits
        """
        batch_features = []
        
        for img_tensor in images:
            # Dénormaliser l'image
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = (img_np * 255).astype('uint8')
            
            # Extraire features
            features_dict = self.gabarit_detector.extract_features(img_np)
            
            # Convertir en liste ordonnée (9 features)
            features_list = [
                features_dict['ratio_carte'],
                features_dict['ratio_a4'],
                features_dict['has_face'],
                features_dict['table_score'],
                features_dict['blue_dominant'],
                features_dict['green_dominant'],
                features_dict['text_density'],
                features_dict['signature_zone'],
                features_dict['numbers_density']
            ]
            
            batch_features.append(features_list)
        
        return torch.FloatTensor(batch_features).to(self.device)
    
    def train_epoch(self):
        """Une epoch d'entraînement"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"  Entraînement sur {len(self.train_loader)} batches...")
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Extraire features gabarits
            gabarit_features = self.extract_gabarit_features(images)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images, gabarit_features)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress (tous les 3 batches)
            if (batch_idx + 1) % 3 == 0 or (batch_idx + 1) == len(self.train_loader):
                print(f"    Batch {batch_idx+1}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Acc={100.*correct/total:.1f}%")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validation"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"  Validation sur {len(self.val_loader)} batches...")
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Features gabarits
                gabarit_features = self.extract_gabarit_features(images)
                
                # Forward
                outputs = self.model(images, gabarit_features)
                loss = self.criterion(outputs, labels)
                
                # Métriques
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs=10, save_dir="models/cv"):
        """
        Boucle d'entraînement complète
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🚀 Début de l'entraînement - {num_epochs} epochs")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Estimated time: ~{num_epochs * 5} minutes")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Entraînement
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Historique
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduler
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Temps
            epoch_time = time.time() - epoch_start
            
            # Affichage
            print(f"\n📊 Résultats Epoch {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Temps: {epoch_time:.1f}s")
            
            # Sauvegarder le meilleur modèle
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, save_path)
                print(f"  ✅ Meilleur modèle sauvegardé: {val_acc:.2f}%")
        
        # Temps total
        total_time = time.time() - start_time
        
        # Sauvegarder l'historique
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ Entraînement terminé!")
        print(f"  Meilleure val accuracy: {self.best_val_acc:.2f}%")
        print(f"  Temps total: {total_time/60:.1f} minutes")
        print(f"  Modèle sauvegardé dans: {save_dir}/")
        print(f"{'='*60}\n")
        
        return self.history


# Script principal
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "C:/Users/mabrchaouen/Documents/projet_classification/data/processed/"
    BATCH_SIZE = 4  # Petit batch pour CPU
    NUM_EPOCHS = 10  # Réduit pour CPU
    DEVICE = 'cpu'
    
    print(f"🔧 Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Data dir: {DATA_DIR}")
    
    # Créer dataloaders
    print(f"\n📦 Chargement du dataset...")
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Créer modèle
    print(f"\n🧠 Création du modèle...")
    model = HybridDocumentClassifier(
        num_classes=len(classes),
        num_gabarit_features=9,
        pretrained=True
    )
    
    print(f"  Classes: {classes}")
    print(f"  Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entraîner
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE)
    history = trainer.train(num_epochs=NUM_EPOCHS)
    
    print("\n✅ Tout est terminé ! Le modèle est prêt.")