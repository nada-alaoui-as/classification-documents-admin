"""
Module de conversion PDF vers images
"""
from pdf2image import convert_from_path
from pathlib import Path

POPPLER_PATH = r'C:\poppler-25.12.0\Library\bin'


def pdf_to_images(pdf_path, output_folder=None):
    """
    Convertit un PDF en images
    
    Args:
        pdf_path: chemin vers le PDF (str ou Path)
        output_folder: dossier de sortie (str ou Path), si None utilise le parent du PDF
        
    Returns:
        list: Liste des chemins vers les images générées
    """
    # Convertir en Path si c'est une string
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    
    # Si pas de dossier de sortie, utiliser le parent du PDF
    if output_folder is None:
        output_folder = pdf_path.parent
    elif isinstance(output_folder, str):
        output_folder = Path(output_folder)
    
    # Créer le dossier de sortie
    output_folder.mkdir(parents=True, exist_ok=True)

    # Convertir le PDF
    images = convert_from_path(
        str(pdf_path),
        poppler_path=POPPLER_PATH
    )

    # Sauvegarder et collecter les chemins
    image_paths = []
    
    for i, image in enumerate(images):
        # nom unique = nom_du_pdf + page
        image_name = f"{pdf_path.stem}_page_{i+1}.jpg"
        image_path = output_folder / image_name
        image.save(image_path, "JPEG")

        print(f"✓ {pdf_path.name} → {image_name}")
        
        # Ajouter le chemin à la liste (en string pour compatibilité)
        image_paths.append(str(image_path))
    
    return image_paths  # ⬅️ AJOUT CRUCIAL


def convert_all_pdfs(raw_dir: Path, processed_dir: Path):
    """
    Convertit tous les PDFs d'un dossier
    """
    for pdf in raw_dir.rglob("*.pdf"):
        # le dossier parent du pdf = la classe
        class_name = pdf.parent.relative_to(raw_dir)

        # data/processed/identite/  OU  data/processed/factures/
        output_dir = processed_dir / class_name

        print(f"\n📄 Conversion : {pdf}")
        pdf_to_images(pdf, output_dir)


# ▶️ MAIN
if __name__ == "__main__":
    BASE_DIR = Path(r"C:\Users\alaou_5lgerz1\projet_classification")

    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    convert_all_pdfs(RAW_DIR, PROCESSED_DIR)

    print("\n🎉 Conversion de tous les PDFs terminée")