# 📦 Modèles Entraînés - Téléchargement

⚠️ **Les modèles ne sont pas inclus dans le dépôt GitHub en raison de leur taille (550 MB total).**

## 🔗 Lien de téléchargement

**Google Drive :** https://drive.google.com/drive/folders/1o9BVqmkTJUmcEABZhpHl4b5PYHCTaQSH?usp=sharing

## 📂 Contenu du dossier
```
models/
├── cv/
│   └── best_model.pth (100 MB)
│       - Modèle hybride ResNet50 + gabarits
│       - Accuracy : 98,24%
│
└── nlp/
    ├── camembert-base/
    │   └── (Modèle pré-entraîné)
    │
    └── camembert_finetuned/
        ├── pytorch_model.bin (440 MB)
        ├── config.json
        └── tokenizer_config.json
        - Modèle CamemBERT fine-tuné
        - Accuracy : 94,12%
```

## 🚀 Installation

1. **Télécharger** les modèles depuis le lien Google Drive ci-dessus
2. **Extraire** le contenu dans le dossier `models/` de votre projet local
3. **Vérifier** la structure :
```
   projet_classification/
   └── models/
       ├── cv/
       │   └── best_model.pth
       └── nlp/
           ├── camembert-base/
           └── camembert_finetuned/
```
4. **Lancer** le système : `python main.py`

## ✅ Vérification

Pour vérifier que les modèles sont correctement installés :
```bash
python -c "from pathlib import Path; print('CV:', Path('models/cv/best_model.pth').exists()); print('NLP:', Path('models/nlp/camembert_finetuned/pytorch_model.bin').exists())"
```

Résultat attendu : `CV: True` et `NLP: True`

---

**Note :** Les modèles ont été entraînés sur 92 documents administratifs marocains (CNIE, relevés bancaires, factures eau/électricité, documents employeur).
