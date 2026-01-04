import re
from typing import Dict, List

class KeywordClassifier:
    def __init__(self):
        self.keywords = {
            'identite': [
                'identité', 'nationale', 'carte', 'cnie', 'né(e)', 'naissance',
                'nationalité', 'marocaine', 'royaume', 'maroc', 'numéro',
                'date émission', 'validité'
            ],
            'releve_bancaire': [
                'solde', 'débit', 'crédit', 'compte', 'banque', 'opération',
                'virement', 'retrait', 'dépôt', 'rib', 'iban', 'swift',
                'relevé', 'transaction', 'balance'
            ],
            'facture_electricite': [
                'kwh', 'électricité', 'puissance', 'abonnement', 'consommation',
                'one', 'lydec', 'redal', 'amendis', 'facture', 'compteur',
                'index', 'watt', 'tarif'
            ],
            'facture_eau': [
                'm³', 'm3', 'eau', 'consommation', 'facture', 'compteur',
                'redal', 'lydec', 'radeema', 'onep', 'index', 'potable',
                'assainissement', 'mètre cube'
            ],
            'document_employeur': [
                'salaire', 'employeur', 'embauche', 'cotisations', 'bulletin',
                'paie', 'cnss', 'attestation', 'travail', 'contrat',
                'rémunération', 'brut', 'net', 'impôt', 'ir'
            ]
        }
        
        self.keyword_weights = {
            'identite': {'cnie': 2.0, 'identité nationale': 2.0},
            'releve_bancaire': {'solde': 1.5, 'rib': 1.5, 'relevé': 1.5},
            'facture_electricite': {'kwh': 3.0, 'one': 1.5},
            'facture_eau': {'m³': 3.0, 'm3': 3.0, 'eau': 1.2},
            'document_employeur': {'bulletin de paie': 2.0, 'cnss': 2.0, 'salaire': 1.5}
        }
    
    def classify(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.keywords.items():
            score = 0.0
            matches = 0
            
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                
                if count > 0:
                    weight = self.keyword_weights.get(category, {}).get(keyword, 1.0)
                    score += count * weight
                    matches += 1
            
            if matches > 0:
                scores[category] = matches / len(keywords)
            else:
                scores[category] = 0.0
        
        return scores
    
    def predict(self, text: str) -> tuple:
        scores = self.classify(text)
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        return best_category, confidence

if __name__ == "__main__":
    classifier = KeywordClassifier()
    
    text_cnie = """
    ROYAUME DU MAROC
    CARTE NATIONALE D'IDENTITÉ ÉLECTRONIQUE
    Numéro: AB123456
    Né(e) le: 01/01/1990
    Nationalité: MAROCAINE
    """
    print("Test CNIE:", classifier.predict(text_cnie))
    
    text_elec = """
    LYDEC - Facture d'électricité
    Consommation: 250 kWh
    Puissance: 6 kVA
    Index compteur: 12345
    """
    print("Test Électricité:", classifier.predict(text_elec))