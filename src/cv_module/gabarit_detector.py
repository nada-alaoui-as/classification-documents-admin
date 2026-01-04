"""
Détecteur de gabarits structurels - Version corrigée et optimisée
"""
import cv2
import numpy as np
from typing import Dict
from PIL import Image


class GabaritDetector:
    """
    Détecte des caractéristiques structurelles dans les documents
    pour aider à la classification
    """
    
    def __init__(self):
        # Charger détecteur de visage (pour CNIE)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def extract_features(self, image_path: str) -> Dict[str, float]:
        """
        Extrait les features de gabarit d'une image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            dict: Features avec scores 0-1
        """
        # Charger image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            # Si c'est déjà une image PIL ou numpy array
            image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        
        if image is None:
            return self._empty_features()
        
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # Feature 1: RATIO D'ASPECT
        ratio = w / h
        features['ratio_carte'] = 1.0 if 1.5 < ratio < 1.7 else 0.0
        features['ratio_a4'] = 1.0 if 1.3 < ratio < 1.5 else 0.0
        
        # Feature 2: DÉTECTION VISAGE (pour CNIE)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )
        features['has_face'] = 1.0 if len(faces) > 0 else 0.0
        
        # Feature 3: DÉTECTION LIGNES HORIZONTALES (tableaux)
        features['table_score'] = self._detect_table_structure(gray, w)
        
        # Feature 4: COULEUR DOMINANTE
        avg_color = cv2.mean(image)[:3]  # BGR
        features['blue_dominant'] = 1.0 if avg_color[0] > 100 else 0.0
        features['green_dominant'] = 1.0 if avg_color[1] > 100 else 0.0
        
        # Feature 5: DENSITÉ DE TEXTE
        features['text_density'] = self._compute_text_density(gray, h, w)
        
        # Feature 6: ZONE SIGNATURE (bas peu dense)
        features['signature_zone'] = self._detect_signature_zone(gray, h)
        
        # Feature 7: DENSITÉ DE CHIFFRES (nouveau)
        features['numbers_density'] = self._detect_numbers_density(gray)
        
        return features
    
    def _detect_table_structure(self, gray: np.ndarray, width: int) -> float:
        """
        Détecte la présence de tableaux (lignes horizontales)
        """
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Détecter lignes avec Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=width//4,
            maxLineGap=20
        )
        
        if lines is None:
            return 0.0
        
        # Compter lignes horizontales
        horizontal_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Ligne quasi horizontale ET suffisamment longue
            if abs(y2 - y1) < 10 and abs(x2 - x1) > width//5:
                horizontal_lines += 1
        
        # Normaliser (max 15 lignes = score 1.0)
        return min(1.0, horizontal_lines / 15.0)
    
    def _compute_text_density(self, gray: np.ndarray, h: int, w: int) -> float:
        """
        Calcule la densité de texte dans l'image
        """
        # Seuillage pour isoler le texte
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Proportion de pixels de texte
        text_pixels = np.sum(thresh == 255)
        total_pixels = h * w
        
        return text_pixels / total_pixels
    
    def _detect_signature_zone(self, gray: np.ndarray, height: int) -> float:
        """
        Détecte une zone potentielle de signature (bas de page peu dense)
        """
        # Analyser le tiers inférieur
        bottom_third = gray[2*height//3:, :]
        _, thresh = cv2.threshold(bottom_third, 127, 255, cv2.THRESH_BINARY_INV)
        
        density = np.sum(thresh == 255) / thresh.size
        
        # Zone de signature = faible densité
        return 1.0 if density < 0.1 else 0.0
    
    def _detect_numbers_density(self, gray: np.ndarray) -> float:
        """
        Détecte la densité de chiffres (utile pour relevés/factures)
        """
        try:
            import pytesseract
            text = pytesseract.image_to_string(gray, config='--psm 6')
            digits = sum(c.isdigit() for c in text)
            total_chars = len([c for c in text if c.isalnum()])
            if total_chars == 0:
                return 0.0
            return min(1.0, digits / total_chars)
        except:
            # Si pytesseract pas installé ou erreur, on estime via contours
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Approximation basique
            return min(1.0, len(contours) / 500.0)
    
    def _empty_features(self) -> Dict[str, float]:
        """Retourne des features vides en cas d'erreur"""
        return {
            'ratio_carte': 0.0,
            'ratio_a4': 0.0,
            'has_face': 0.0,
            'table_score': 0.0,
            'blue_dominant': 0.0,
            'green_dominant': 0.0,
            'text_density': 0.0,
            'signature_zone': 0.0,
            'numbers_density': 0.0
        }
    
    def compute_gabarit_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {}
        
        scores['identite'] = (
            features['ratio_carte'] * 0.5 +
            features['has_face'] * 0.4 +
            features['green_dominant'] * 0.1
        )
        
        scores['releve_bancaire'] = (
            features['ratio_a4'] * 0.2 +
            features['table_score'] * 0.5 +
            features['text_density'] * 0.3
        )
        
        scores['facture_electricite'] = (
            features['ratio_a4'] * 0.2 +
            features['table_score'] * 0.3 +
            (1.0 - features['text_density']) * 0.2 +
            features['text_density'] * 0.3
        )
        
        scores['facture_eau'] = (
            features['ratio_a4'] * 0.2 +
            features['table_score'] * 0.4 +
            features['blue_dominant'] * 0.1 +
            features['text_density'] * 0.2 +
            features['numbers_density'] * 0.1
        )
        
        scores['document_employeur'] = (
            features['ratio_a4'] * 0.3 +
            features['signature_zone'] * 0.3 +
            features['text_density'] * 0.4
        )
        
        return scores
    
    def predict_from_image(self, image_path: str) -> tuple:
        """
        Prédit la catégorie d'une image basée sur les gabarits
        
        Returns:
            (categorie, confiance, all_scores)
        """
        features = self.extract_features(image_path)
        scores = self.compute_gabarit_scores(features)
        
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        return best_category, confidence, scores


# Test
if __name__ == "__main__":
    detector = GabaritDetector()
    
    # Test sur une image
    # category, conf, scores = detector.predict_from_image("data/images/test.png")
    # print(f"Catégorie: {category} ({conf:.2%})")
    # print(f"Scores: {scores}")
    
    print("✅ Module gabarit_detector.py prêt!")