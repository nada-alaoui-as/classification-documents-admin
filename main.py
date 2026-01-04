import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from nlp.nlp_pipeline import NLPPipeline
from fusion.multimodal_fusion import MultimodalFusion
from cv_module.cv_pipeline import CVPipeline

class DocumentClassificationPipeline:
    def __init__(self, tesseract_path=r'C:\Users\alaou_5lgerz1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe', cv_model_path="models/cv/best_model.pth", use_camembert=False):
        self.nlp_pipeline = NLPPipeline(tesseract_path, use_camembert)
        self.cv_pipeline = CVPipeline(model_path=cv_model_path, device='cpu')
        self.fusion = MultimodalFusion()
    
    def classify_document(self, pdf_path: str):
        print(f"\n{'='*80}")
        print(f"🔍 CLASSIFICATION: {pdf_path}")
        print(f"{'='*80}")
        
        nlp_result = self.nlp_pipeline.process_pdf(pdf_path)
        nlp_prediction = (nlp_result['category'], nlp_result['confidence'])
        
        cv_result = self.cv_pipeline.process_pdf(pdf_path)
        cv_prediction = (cv_result['category'], cv_result['confidence'])
        gabarit_scores = cv_result['gabarit_scores']
        
        fusion_result = self.fusion.fuse(
            cv_prediction=cv_prediction,
            gabarit_scores=gabarit_scores,
            nlp_prediction=nlp_prediction
        )
        
        should_review = self.fusion.should_reject(fusion_result)
        
        result = {
            'file': os.path.basename(pdf_path),
            'category': fusion_result['category'],
            'confidence': fusion_result['confidence'],
            'method': fusion_result['method'],
            'should_review': should_review,
            'nlp_result': nlp_result,
            'cv_result': cv_result
        }
        
        print(f"\n{'='*80}")
        print(f"📋 RÉSULTAT FINAL")
        print(f"{'='*80}")
        print(f"   Catégorie: {result['category']}")
        print(f"   Confiance: {result['confidence']:.2%}")
        print(f"   Méthode: {result['method']}")
        print(f"   À vérifier: {'⚠️  OUI' if should_review else '✅ NON'}")
        print(f"{'='*80}\n")
        
        return result

def main():
    tesseract_path = r'C:\Users\alaou_5lgerz1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

    USE_CAMEMBERT = True 
    
    pipeline = DocumentClassificationPipeline(tesseract_path=tesseract_path, use_camembert=USE_CAMEMBERT)
    
    input_folder = "data/raw"
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_files = list(Path(input_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ Aucun PDF trouvé dans {input_folder}")
        return
    
    print(f"📁 {len(pdf_files)} PDFs à traiter\n")
    
    results = []
    for pdf_path in pdf_files:
        try:
            result = pipeline.classify_document(str(pdf_path))
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur sur {pdf_path}: {e}\n")
    
    print(f"\n{'='*80}")
    print(f"📊 RÉSUMÉ")
    print(f"{'='*80}")
    print(f"Total traités: {len(results)}")
    print(f"À vérifier: {sum(1 for r in results if r['should_review'])}")
    print(f"\nRépartition par catégorie:")
    
    from collections import Counter
    categories = Counter(r['category'] for r in results)
    for cat, count in categories.most_common():
        print(f"   {cat}: {count}")
    
    import json
    
    results_serializable = []
    for r in results:
        r_copy = r.copy()
        r_copy['should_review'] = bool(r_copy['should_review'])
        results_serializable.append(r_copy)
    
    with open(f"{output_folder}/results.json", "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ Résultats sauvegardés dans {output_folder}/results.json")

if __name__ == "__main__":
    main()