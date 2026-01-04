from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

class CamemBERTClassifier:
    def __init__(self, model_path="models/nlp/camembert_finetuned"):
        self.tokenizer = CamembertTokenizer.from_pretrained(model_path)
        self.model = CamembertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.classes = [
            'identite',
            'releve_bancaire',
            'facture_electricite',
            'facture_eau',
            'document_employeur'
        ]
    
    def classify(self, text: str) -> dict:
        """
        Retourne les scores pour toutes les classes
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
        
        scores = {
            self.classes[i]: probs[i].item() 
            for i in range(len(self.classes))
        }
        
        return scores
    
    def predict(self, text: str) -> tuple:
        """
        Retourne (catégorie, confiance)
        """
        scores = self.classify(text)
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        return best_category, confidence

if __name__ == "__main__":
    classifier = CamemBERTClassifier()
    
    text_test = "ROYAUME DU MAROC CARTE NATIONALE D'IDENTITÉ"
    category, conf = classifier.predict(text_test)
    print(f"Catégorie: {category}, Confiance: {conf:.2%}")