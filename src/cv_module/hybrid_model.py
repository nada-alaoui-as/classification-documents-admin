"""
Modèle hybride: ResNet50 + Features de gabarits
"""
import torch
import torch.nn as nn
import torchvision.models as models


class HybridDocumentClassifier(nn.Module):
    """
    Modèle qui combine:
    - Features visuelles (ResNet50)
    - Features de gabarits (9 features numériques)
    """
    
    def __init__(self, num_classes=5, num_gabarit_features=9, pretrained=True):
        """
        Args:
            num_classes: Nombre de catégories (5)
            num_gabarit_features: Nombre de features gabarits (9)
            pretrained: Utiliser ResNet50 pré-entraîné
        """
        super(HybridDocumentClassifier, self).__init__()
        
        # Branche 1: ResNet50 pour features visuelles
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Remplacer la dernière couche FC
        # ResNet50 original: ... → fc(2048 → 1000)
        # On veut: ... → fc(2048 → 128) pour avoir un embedding
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 128)
        
        # Branche 2: MLP pour features gabarits
        self.gabarit_mlp = nn.Sequential(
            nn.Linear(num_gabarit_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion: Concaténation des deux branches
        # 128 (ResNet) + 32 (Gabarits) = 160
        self.fusion = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, image, gabarit_features):
        """
        Forward pass
        
        Args:
            image: Tensor (batch, 3, 224, 224)
            gabarit_features: Tensor (batch, 9)
            
        Returns:
            logits: Tensor (batch, num_classes)
        """
        # Branche visuelle
        visual_features = self.resnet(image)  # (batch, 128)
        
        # Branche gabarits
        gabarit_embedding = self.gabarit_mlp(gabarit_features)  # (batch, 32)
        
        # Fusion
        combined = torch.cat([visual_features, gabarit_embedding], dim=1)  # (batch, 160)
        logits = self.fusion(combined)  # (batch, 5)
        
        return logits


# Test
if __name__ == "__main__":
    print("🧪 Test du modèle hybride...")
    
    model = HybridDocumentClassifier(num_classes=5)
    
    # Test forward pass
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_gabarits = torch.randn(batch_size, 9)
    
    output = model(dummy_image, dummy_gabarits)
    print(f"✅ Output shape: {output.shape}")  # Should be (4, 5)
    
    # Nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Statistiques du modèle:")
    print(f"  Total de paramètres: {total_params:,}")
    print(f"  Paramètres entraînables: {trainable_params:,}")
    print(f"  Taille estimée: {total_params * 4 / (1024**2):.2f} MB")
    
    print("\n✅ Modèle hybride prêt pour l'entraînement!")