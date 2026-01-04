from typing import Dict, Tuple

class MultimodalFusion:
    def __init__(self):
        self.weights = {
            'cv': 0.4,
            'gabarits': 0.3,
            'nlp': 0.3
        }
        
        self.high_confidence_threshold = 0.85
        self.medium_confidence_threshold = 0.60
    
    def fuse(self, cv_prediction: Tuple[str, float], gabarit_scores: Dict[str, float], nlp_prediction: Tuple[str, float]) -> Dict:
        cv_cat, cv_conf = cv_prediction
        nlp_cat, nlp_conf = nlp_prediction
        
        print(f"\n🔮 Fusion Multimodale")
        print(f"   CV:       {cv_cat} ({cv_conf:.2%})")
        print(f"   Gabarits: {max(gabarit_scores, key=gabarit_scores.get)} ({max(gabarit_scores.values()):.2%})")
        print(f"   NLP:      {nlp_cat} ({nlp_conf:.2%})")
        
        if cv_cat == nlp_cat and cv_conf > 0.7 and nlp_conf > 0.7:
            final_conf = (cv_conf + nlp_conf) / 2
            return {
                'category': cv_cat,
                'confidence': final_conf,
                'method': 'perfect_agreement',
                'details': f"CV et NLP d'accord avec haute confiance"
            }
        
        if cv_conf > self.high_confidence_threshold:
            gabarit_score = gabarit_scores.get(cv_cat, 0.0)
            if gabarit_score > 0.6:
                return {
                    'category': cv_cat,
                    'confidence': cv_conf * 0.7 + gabarit_score * 0.3,
                    'method': 'cv_strong_validated',
                    'details': f"CV très confiant, validé par gabarits"
                }
        
        if nlp_conf > self.high_confidence_threshold:
            gabarit_score = gabarit_scores.get(nlp_cat, 0.0)
            if gabarit_score > 0.6:
                return {
                    'category': nlp_cat,
                    'confidence': nlp_conf * 0.7 + gabarit_score * 0.3,
                    'method': 'nlp_strong_validated',
                    'details': f"NLP très confiant, validé par gabarits"
                }
        
        all_categories = set(gabarit_scores.keys())
        combined_scores = {}
        
        for category in all_categories:
            cv_score = 1.0 if category == cv_cat else 0.0
            nlp_score = 1.0 if category == nlp_cat else 0.0
            gabarit_score = gabarit_scores.get(category, 0.0)
            
            combined_scores[category] = (
                cv_score * cv_conf * self.weights['cv'] +
                gabarit_score * self.weights['gabarits'] +
                nlp_score * nlp_conf * self.weights['nlp']
            )
        
        best_category = max(combined_scores, key=combined_scores.get)
        best_score = combined_scores[best_category]
        
        return {
            'category': best_category,
            'confidence': best_score,
            'method': 'weighted_vote',
            'details': f"Vote pondéré: CV={self.weights['cv']}, Gabarits={self.weights['gabarits']}, NLP={self.weights['nlp']}"
        }
    
    def should_reject(self, result: Dict) -> bool:
        return result['confidence'] < 0.50

if __name__ == "__main__":
    fusion = MultimodalFusion()
    
    result = fusion.fuse(
        cv_prediction=('identite', 0.92),
        gabarit_scores={
            'identite': 0.85,
            'releve_bancaire': 0.10,
            'facture_electricite': 0.05,
            'facture_eau': 0.00,
            'document_employeur': 0.00
        },
        nlp_prediction=('identite', 0.88)
    )
    print(f"\n✅ Résultat: {result}")