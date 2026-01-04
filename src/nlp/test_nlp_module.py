"""
Script de test pour que Salma puisse tester ton module NLP
"""
from src.nlp.nlp_pipeline import NLPPipeline

def test_nlp():
    pipeline = NLPPipeline()
    
    # Tester sur différents PDFs
    test_files = [
        "data/raw/cnie_test.pdf",
        "data/raw/releve_test.pdf",
        "data/raw/facture_elec_test.pdf",
    ]
    
    for pdf_path in test_files:
        try:
            result = pipeline.process_pdf(pdf_path)
            print(f"\n✅ {pdf_path}: {result['category']} ({result['confidence']:.0%})")
        except Exception as e:
            print(f"\n❌ Erreur sur {pdf_path}: {e}")

if __name__ == "__main__":
    test_nlp()