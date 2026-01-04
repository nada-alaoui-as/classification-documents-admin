import torch
import torchvision.models as models
import os

def download_resnet50():
    """
    Télécharge et sauvegarde ResNet50 localement
    """
    print("📥 Téléchargement de ResNet50...")
    
    # Créer le dossier models/cv s'il n'existe pas
    os.makedirs("models/cv", exist_ok=True)
    
    # Télécharger ResNet50 pré-entraîné
    model = models.resnet50(pretrained=True)
    
    # Sauvegarder le modèle
    model_path = "models/cv/resnet50.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"✓ ResNet50 sauvegardé dans {model_path}")
    print(f"✓ Taille du fichier : {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    return model_path


def load_resnet50():
    """
    Charge ResNet50 depuis le disque local
    """
    print("📂 Chargement de ResNet50 depuis le disque...")
    
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load("models/cv/resnet50.pth"))
    model.eval()  # Mode évaluation
    
    print("✓ Modèle chargé avec succès !")
    return model


# TEST
if __name__ == "__main__":
    # Télécharger une seule fois
    download_resnet50()
    
    # Tester le chargement
    model = load_resnet50()
    print("\n✓ Tout fonctionne !")