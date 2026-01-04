import os
from pathlib import Path
from preprocessing.pdf_to_text import PDFTextExtractor

def clean_text(text):
    """Nettoie le texte pour le format TSV"""
    # Remplacer les sauts de ligne par des espaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remplacer les tabulations par des espaces
    text = text.replace('\t', ' ')
    # Supprimer les espaces multiples
    text = ' '.join(text.split())
    return text

def create_nlp_dataset():
    extractor = PDFTextExtractor(
        tesseract_path=r'C:\Users\alaou_5lgerz1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    )
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "dataset_nlp"
    
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "train_data.txt"
    
    print(f"📂 Lecture depuis: {data_dir}")
    print(f"📝 Écriture vers: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for class_folder in data_dir.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                print(f"\n🔍 Traitement classe: {class_name}")
                
                pdf_count = 0
                for pdf_file in class_folder.glob("*.pdf"):
                    print(f"  📄 {pdf_file.name}...")
                    text = extractor.extract_text_from_pdf(str(pdf_file))
                    
                    # NETTOYER LE TEXTE ← AJOUT ICI
                    text = clean_text(text)
                    
                    # Vérifier que le texte n'est pas vide
                    if len(text) > 50:
                        f.write(f"{class_name}\t{text}\n")
                        pdf_count += 1
                
                print(f"  ✅ {pdf_count} PDFs traités")
    
    print(f"\n✅ Dataset créé: {output_file}")

if __name__ == "__main__":
    create_nlp_dataset()