"""
Pipeline CV complet: PDF → Images → Prédiction
"""
import torch
import sys
import os
import numpy as np
from pathlib import Path


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pdf_to_images import pdf_to_images
from cv_module.hybrid_model import HybridDocumentClassifier
from cv_module.gabarit_detector import GabaritDetector
from cv_module.dataset_utils import get_transforms


class CVPipeline:
    def __init__(self, model_path="models/cv/best_model.pth", device='cpu'):
        """
        Pipeline complet de classification visuelle
        
        Args:
            model_path: Chemin vers le modèle entraîné
            device: 'cuda' ou 'cpu'
        """
        self.device = torch.device(device)
        
        # Classes (IMPORTANT: même ordre que l'entraînement)
        self.classes = [
            'document_employeur',
            'facture_eau',
            'facture_electricite',
            'identite',
            'releve_bancaire'
        ]
        
        # Charger modèle
        print(f"📂 Chargement du modèle depuis {model_path}...")
        self.model = HybridDocumentClassifier(
            num_classes=len(self.classes),
            num_gabarit_features=9,
            pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Autres modules
        self.gabarit_detector = GabaritDetector()
        self.transform = get_transforms(train=False)
        
        print(f"✅ Pipeline CV initialisé (device: {self.device})")
        print(f"   Classes: {self.classes}")
    
    def extract_gabarit_features(self, image_tensor):
        """
        Extrait features de gabarits d'une image
        
        Args:
            image_tensor: Tensor (3, 224, 224) normalisé
            
        Returns:
            Tensor (9,) avec features
        """
        # Dénormaliser
        img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = (img_np * 255).astype('uint8')
        
        # Extraire features
        features_dict = self.gabarit_detector.extract_features(img_np)
        
        # Convertir en liste ordonnée
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
        
        return torch.FloatTensor(features_list)
    
    def predict_image(self, image_path):
        """
        Prédit la classe d'une image
        
        Returns:
            dict: {
                'category': catégorie prédite,
                'confidence': niveau de confiance,
                'all_scores': scores pour toutes les classes,
                'gabarit_scores': scores gabarits
            }
        """
        # Charger et transformer image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Features gabarits
        gabarit_features = self.extract_gabarit_features(self.transform(image))
        gabarit_features = gabarit_features.unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(image_tensor, gabarit_features)
            probs = torch.softmax(outputs, dim=1)[0]
        
        # Résultats
        predicted_idx = torch.argmax(probs).item()
        confidence = probs[predicted_idx].item()
        category = self.classes[predicted_idx]
        
        # Tous les scores CNN
        all_scores = {
            self.classes[i]: probs[i].item() 
            for i in range(len(self.classes))
        }
        
        # Scores gabarits
        gabarit_scores = self.gabarit_detector.compute_gabarit_scores(
            self.gabarit_detector.extract_features(image_path)
        )
        
        return {
            'category': category,
            'confidence': confidence,
            'all_scores': all_scores,
            'gabarit_scores': gabarit_scores
        }
    
    def process_pdf(self, pdf_path, output_dir="data/images"):
        """
        Traite un PDF complet
        
        Returns:
            dict: Résultat pour la première page (ou agrégé)
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        print(f"\n{'='*60}")
        print(f"🖼  Traitement CV: {pdf_path}")
        print(f"{'='*60}")
        
        # Convertir PDF en images
        print("🔄 Conversion PDF → Images...")
        image_paths = pdf_to_images(pdf_path, output_dir)
        
        if not image_paths:
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'all_scores': {},
                'gabarit_scores': {}
            }
        
        print(f"✅ {len(image_paths)} page(s) convertie(s)")
        
        # Prédire sur la première page
        print("🤖 Classification en cours...")
        result = self.predict_image(image_paths[0])
        
        # Affichage
        print(f"\n📊 Résultats CV:")
        print(f"  Catégorie: {result['category']}")
        print(f"  Confiance: {result['confidence']:.2%}")
        print(f"\n  Scores CNN:")
        for cat, score in sorted(result['all_scores'].items(), 
                                 key=lambda x: x[1], reverse=True):
            print(f"    {cat}: {score:.2%}")
        print(f"\n  Scores Gabarits:")
        for cat, score in sorted(result['gabarit_scores'].items(), 
                                 key=lambda x: x[1], reverse=True):
            print(f"    {cat}: {score:.2%}")
        
        return result


# Test
if __name__ == "__main__":
    # Initialiser le pipeline
    pipeline = CVPipeline()
    
    # Tester sur un PDF
    test_pdf = r".\SRM.pdf"
    
    if os.path.exists(test_pdf):
        result = pipeline.process_pdf(test_pdf)
    else:
        print(f"❌ Fichier de test non trouvé: {test_pdf}")
        print("Modifie le chemin dans le script pour tester.")
    
    print("\n✅ Pipeline CV prêt à l'emploi!")