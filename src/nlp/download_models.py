from transformers import CamembertTokenizer, CamembertModel
import os

model_dir = "models/nlp/camembert-base"
os.makedirs(model_dir, exist_ok=True)

print("📥 Téléchargement de CamemBERT...")
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained('camembert-base')

print("💾 Sauvegarde locale...")
tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)

print("✅ CamemBERT sauvegardé!")