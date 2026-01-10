"""
Classification par règles logiques + fallback NLP
"""
import sys
import os
from pathlib import Path
import re
import json
from collections import Counter

sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing.pdf_to_text import PDFTextExtractor
from nlp.camembert_classifier import CamemBERTClassifier


class RuleBasedClassifier:
    """
    Classificateur basé sur règles logiques (AND/OR/XOR)
    Court-circuite l'inférence lourde quand des mots-clés discriminants sont présents
    """
    
    def __init__(self):
        # Mots-clés ULTRA-DISCRIMINANTS (quasi-certitude)
        self.strong_keywords = {
            'facture_electricite': {
                'critical': ['kwh', 'kilowatt', 'kw/h'],  # AND implicite (un suffit)
                'supporting': ['électricité', 'one', 'lydec', 'redal', 'compteur électrique']
            },
            'facture_eau': {
                'critical': ['m³', 'm3', 'mètre cube', 'metre cube'],
                'supporting': ['eau', 'onep', 'radeema', 'potable', 'assainissement']
            },
            'identite': {
                'critical': ['cnie', 'carte nationale', "carte d'identité"],
                'supporting': ['identité nationale', 'né le', 'née le', 'nationalité marocaine']
            },
            'releve_bancaire': {
                'critical': ['rib', 'iban'],
                'supporting': ['solde', 'débit', 'crédit', 'virement', 'relevé de compte']
            },
            'document_employeur': {
                'critical': ['cnss', 'bulletin de paie', 'bulletin de salaire'],
                'supporting': ['salaire brut', 'salaire net', 'cotisations', 'employeur']
            }
        }
        
        # Mots-clés faibles (nécessitent combinaison)
        self.weak_keywords = {
            'facture_electricite': ['facture', 'consommation', 'abonnement', 'compteur'],
            'facture_eau': ['facture', 'consommation', 'abonnement', 'compteur'],
            'identite': ['royaume', 'maroc', 'prénom', 'nom'],
            'releve_bancaire': ['banque', 'compte', 'date', 'opération'],
            'document_employeur': ['attestation', 'travail', 'fonction']
        }
    
    def normalize_text(self, text):
        """Normalise le texte pour matching robuste"""
        text = text.lower()
        text = text.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
        text = text.replace('à', 'a').replace('â', 'a')
        text = text.replace('ô', 'o')
        text = text.replace('î', 'i')
        text = text.replace('ù', 'u').replace('û', 'u')
        text = text.replace('ç', 'c')
        return text
    
    def apply_rule_1_critical_keyword(self, text_normalized):
        """
        RÈGLE 1 (AND implicite) : Mot-clé critique présent
        Si un mot-clé ultra-discriminant est trouvé → Classification immédiate
        """
        for category, keywords in self.strong_keywords.items():
            for keyword in keywords['critical']:
                if keyword in text_normalized:
                    return {
                        'category': category,
                        'confidence': 0.95,
                        'method': 'rule_1_critical_keyword',
                        'matched_keyword': keyword
                    }
        return None
    
    def apply_rule_2_critical_and_supporting(self, text_normalized):
        """
        RÈGLE 2 (AND) : Mot-clé critique + mot-clé support
        Renforce la confiance si les deux sont présents
        """
        for category, keywords in self.strong_keywords.items():
            critical_found = any(kw in text_normalized for kw in keywords['critical'])
            supporting_found = any(kw in text_normalized for kw in keywords['supporting'])
            
            if critical_found and supporting_found:
                return {
                    'category': category,
                    'confidence': 0.98,
                    'method': 'rule_2_critical_and_supporting'
                }
        return None
    
    def apply_rule_3_multiple_supporting(self, text_normalized):
        """
        RÈGLE 3 (OR multiple) : Au moins 3 mots-clés supporting
        Si 3+ mots-clés support présents → Forte probabilité
        """
        for category, keywords in self.strong_keywords.items():
            supporting_matches = sum(1 for kw in keywords['supporting'] if kw in text_normalized)
            
            if supporting_matches >= 3:
                return {
                    'category': category,
                    'confidence': 0.85,
                    'method': 'rule_3_multiple_supporting',
                    'matches': supporting_matches
                }
        return None
    
    def apply_rule_4_weak_combination(self, text_normalized):
        """
        RÈGLE 4 (AND combiné) : Combinaison mots faibles
        Facture électricité : "facture" AND "consommation" AND "compteur"
        """
        # Facture électricité
        if ('facture' in text_normalized and 
            'consommation' in text_normalized and 
            'compteur' in text_normalized and
            'kwh' not in text_normalized and 'm3' not in text_normalized):
            # Heuristique : si "électrique/électricité" présent
            if 'electr' in text_normalized:
                return {
                    'category': 'facture_electricite',
                    'confidence': 0.70,
                    'method': 'rule_4_weak_elec'
                }
        
        # Facture eau (même logique)
        if ('facture' in text_normalized and 
            'consommation' in text_normalized and 
            'compteur' in text_normalized and
            'kwh' not in text_normalized):
            if 'eau' in text_normalized or 'potable' in text_normalized:
                return {
                    'category': 'facture_eau',
                    'confidence': 0.70,
                    'method': 'rule_4_weak_eau'
                }
        
        return None
    
    def apply_rule_5_xor_disambiguation(self, text_normalized):
        """
        RÈGLE 5 (XOR) : Désambiguïsation facture eau vs électricité
        Si les deux semblent possibles, départager par mot-clé exclusif
        """
        has_elec_weak = any(kw in text_normalized for kw in ['electr', 'one', 'lydec'])
        has_eau_weak = any(kw in text_normalized for kw in ['eau', 'onep', 'radeema'])
        
        # XOR : exactement un des deux
        if has_elec_weak and not has_eau_weak:
            return {
                'category': 'facture_electricite',
                'confidence': 0.75,
                'method': 'rule_5_xor_elec'
            }
        elif has_eau_weak and not has_elec_weak:
            return {
                'category': 'facture_eau',
                'confidence': 0.75,
                'method': 'rule_5_xor_eau'
            }
        
        return None
    
    def apply_rule_6_negative_exclusion(self, text_normalized):
        """
        RÈGLE 6 (AND NOT) : Exclusion par mots-clés contradictoires
        Si "kwh" présent → NE PEUT PAS être facture eau
        """
        if 'kwh' in text_normalized:
            # Exclure facture_eau, releve, identite, employeur
            return {
                'category': 'facture_electricite',
                'confidence': 0.90,
                'method': 'rule_6_exclusion_kwh'
            }
        
        if 'm3' in text_normalized or 'metre cube' in text_normalized:
            return {
                'category': 'facture_eau',
                'confidence': 0.90,
                'method': 'rule_6_exclusion_m3'
            }
        
        return None
    
    def classify(self, text):
        """
        Applique les règles en cascade (par ordre de priorité)
        Retourne dès qu'une règle match
        """
        text_normalized = self.normalize_text(text)
        
        # Ordre de priorité décroissant
        rules = [
            self.apply_rule_1_critical_keyword,
            self.apply_rule_2_critical_and_supporting,
            self.apply_rule_6_negative_exclusion,
            self.apply_rule_3_multiple_supporting,
            self.apply_rule_5_xor_disambiguation,
            self.apply_rule_4_weak_combination
        ]
        
        for rule in rules:
            result = rule(text_normalized)
            if result:
                return result
        
        # Aucune règle ne match → Fallback sur modèle
        return None


class HybridClassificationPipeline:
    """
    Pipeline hybride : Règles logiques + Fallback NLP
    """
    
    def __init__(self, tesseract_path, nlp_model_path="models/nlp/camembert_finetuned"):
        self.ocr = PDFTextExtractor(tesseract_path)
        self.rule_classifier = RuleBasedClassifier()
        self.nlp_classifier = CamemBERTClassifier(model_path=nlp_model_path)
        
        self.stats = {
            'total': 0,
            'rules': 0,
            'fallback_nlp': 0
        }
    
    def classify_document(self, pdf_path):
        print(f"\n{'='*80}")
        print(f"🔍 CLASSIFICATION HYBRIDE: {Path(pdf_path).name}")
        print(f"{'='*80}")
        
        # Étape 1 : OCR
        print("📄 Extraction texte (OCR)...")
        text = self.ocr.extract_text_from_pdf(pdf_path)
        
        if len(text) < 50:
            print("⚠️  Texte trop court, pas d'OCR fiable")
            text = ""
        
        # Étape 2 : Règles logiques
        print("🔧 Application règles logiques...")
        rule_result = self.rule_classifier.classify(text)
        
        if rule_result:
            print(f"✅ RÈGLE MATCHÉE: {rule_result['method']}")
            print(f"   Catégorie: {rule_result['category']}")
            print(f"   Confiance: {rule_result['confidence']:.2%}")
            
            self.stats['rules'] += 1
            
            return {
                'file': Path(pdf_path).name,
                'category': rule_result['category'],
                'confidence': rule_result['confidence'],
                'method': rule_result['method'],
                'source': 'rules'
            }
        
        # Étape 3 : Fallback NLP
        print("🤖 Fallback → CamemBERT fine-tuné...")
        category, confidence = self.nlp_classifier.predict(text)
        
        self.stats['fallback_nlp'] += 1
        
        print(f"   Catégorie: {category}")
        print(f"   Confiance: {confidence:.2%}")
        
        return {
            'file': Path(pdf_path).name,
            'category': category,
            'confidence': confidence,
            'method': 'camembert_finetuned',
            'source': 'nlp_model'
        }


def main():
    tesseract_path = r'C:\Users\alaou_5lgerz1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    
    pipeline = HybridClassificationPipeline(tesseract_path)
    
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
            pipeline.stats['total'] += 1
        except Exception as e:
            print(f"❌ Erreur sur {pdf_path}: {e}\n")
    
    # Statistiques finales
    print(f"\n{'='*80}")
    print(f"📊 STATISTIQUES FINALES")
    print(f"{'='*80}")
    print(f"Total traités: {pipeline.stats['total']}")
    print(f"Règles logiques: {pipeline.stats['rules']} ({pipeline.stats['rules']/pipeline.stats['total']*100:.1f}%)")
    print(f"Fallback NLP: {pipeline.stats['fallback_nlp']} ({pipeline.stats['fallback_nlp']/pipeline.stats['total']*100:.1f}%)")
    
    print(f"\nRépartition par catégorie:")
    categories = Counter(r['category'] for r in results)
    for cat, count in categories.most_common():
        print(f"   {cat}: {count}")
    
    # Sauvegarder
    with open(f"{output_folder}/results_rules.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Résultats sauvegardés dans {output_folder}/results_rules.json")


if __name__ == "__main__":
    main()