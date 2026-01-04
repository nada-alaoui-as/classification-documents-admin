"""
Pipeline NLP complet: PDF → Texte → Classification
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pdf_to_text import PDFTextExtractor
from nlp.keyword_classifier import KeywordClassifier

class NLPPipeline:
    def __init__(self, tesseract_path=r'C:\Users\alaou_5lgerz1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe', use_camembert=False):
        self.extractor = PDFTextExtractor(tesseract_path)
        
        if use_camembert:
            from nlp.camembert_classifier import CamemBERTClassifier
            self.classifier = CamemBERTClassifier()
        else:
            from nlp.keyword_classifier import KeywordClassifier
            self.classifier = KeywordClassifier()
        
        self.use_camembert = use_camembert
    
    def process_pdf(self, pdf_path):
        """
        Traite un PDF de bout en bout
        
        Returns:
            dict: {
                'text': texte extrait,
                'category': catégorie prédite,
                'confidence': niveau de confiance,
                'all_scores': scores pour toutes les catégories
            }
        """
        # 1. Extraire le texte
        print(f"\n{'='*60}")
        print(f"Traitement de: {pdf_path}")
        print(f"{'='*60}")
        
        text = self.extractor.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            return {
                'text': '',
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'all_scores': {}
            }
        
        # 2. Classifier
        all_scores = self.classifier.classify(text)
        category, confidence = self.classifier.predict(text)
        
        # 3. Afficher résultats
        print(f"\n📊 Résultats NLP:")
        print(f"   Catégorie: {category}")
        print(f"   Confiance: {confidence:.2%}")
        print(f"\n   Scores détaillés:")
        for cat, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"      {cat}: {score:.2%}")
        
        return {
            'text': text,
            'category': category,
            'confidence': confidence,
            'all_scores': all_scores
        }

# Test
if __name__ == "__main__":
    # Ajuster le chemin Tesseract si besoin (Windows)
    pipeline = NLPPipeline()
    
    # Tester sur un PDF
    result = pipeline.process_pdf("data/raw/test.pdf")