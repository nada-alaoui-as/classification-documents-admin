"""
Module d'extraction de texte depuis PDF
Utilise Tesseract OCR pour extraire le texte de chaque page
"""
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

class PDFTextExtractor:
    def __init__(self, tesseract_path=None):
        """
        Args:
            tesseract_path: Chemin vers tesseract.exe (Windows uniquement)
                           Ex: r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def extract_text_from_pdf(self, pdf_path, lang='fra'):
        """
        Extrait le texte d'un PDF
        
        Args:
            pdf_path: Chemin vers le PDF
            lang: Langue pour OCR (défaut: français)
        
        Returns:
            str: Texte extrait de toutes les pages
        """
        print(f"🔍 Extraction de texte depuis: {pdf_path}")
        
        # Convertir PDF en images
        try:
            images = convert_from_path(pdf_path, 
                                       dpi=300,
                                       poppler_path=r'C:\poppler-25.12.0\Library\bin'
                                       )
        except Exception as e:
            print(f"❌ Erreur conversion PDF: {e}")
            return ""
        
        # Extraire texte de chaque page
        full_text = ""
        for i, image in enumerate(images):
            print(f"  Page {i+1}/{len(images)}...")
            
            # Prétraitement image (optionnel - améliore OCR)
            # image = self._preprocess_image(image)
            
            # OCR
            text = pytesseract.image_to_string(image, lang=lang)
            full_text += f"\n--- PAGE {i+1} ---\n{text}\n"
        
        print(f"✅ {len(images)} pages traitées")
        return full_text
    
    def _preprocess_image(self, image):
        """
        Prétraitement pour améliorer l'OCR (OPTIONNEL - à faire si tu as le temps)
        """
        import cv2
        import numpy as np
        
        # Convertir en array numpy
        img_array = np.array(image)
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Débruitage
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Améliorer contraste
        enhanced = cv2.equalizeHist(denoised)
        
        # Reconvertir en PIL Image
        return Image.fromarray(enhanced)

# Test du module
if __name__ == "__main__":
    # WINDOWS
    extractor = PDFTextExtractor(r'C:\Users\alaou_5lgerz1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe')
    
    # Test sur un PDF
    text = extractor.extract_text_from_pdf("data/raw/test.pdf")
    print(text[:500])  # Affiche les 500 premiers caractères
    
    # MAC/LINUX
    extractor = PDFTextExtractor()
    
    # Test sur un PDF (tu dois créer un fichier test)
    # text = extractor.extract_text_from_pdf("data/raw/test.pdf")
    # print(text[:500])  # Affiche les 500 premiers caractères