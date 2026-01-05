# Automated Classification of Moroccan Administrative Documents

> **Multimodal AI system combining Computer Vision and Natural Language Processing**  
> Final Project - Computer Vision & NLP Course | ENSAM Rabat

[![Accuracy](https://img.shields.io/badge/Accuracy-98.5%25-success)](https://github.com/nada-alaoui-as/classification-documents-admin)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Project Overview

An intelligent document classification system that automatically categorizes Moroccan administrative documents into 5 classes:
- **National ID Cards (CNIE)** - Electronic identity cards with biometric data
- **Bank Statements** - Account transaction records
- **Electricity Bills** - Utility invoices (ONE, LYDEC, REDAL)
- **Water Bills** - Water consumption invoices (ONEP, RADEEMA)
- **Employment Documents** - Pay slips, work certificates, CNSS documents

**Problem Statement:** Manual document sorting in Moroccan organizations is slow (1000s of docs/day), expensive, and error-prone. This system automates classification with 98.5% accuracy, reducing manual workload by ~80%.

## 🏗️ System Architecture
```
┌─────────────┐
│  PDF Input  │
└──────┬──────┘
       │
   ┌───▼────────────────────────┐
   │  Preprocessing Pipeline    │
   │  (PDF→Images @ 300 DPI)   │
   └───┬───────────────────┬───┘
       │                   │
┌──────▼────────┐   ┌─────▼──────────┐
│ Computer      │   │ Natural        │
│ Vision Module │   │ Language       │
│               │   │ Processing     │
│ ResNet50 +    │   │ Module         │
│ Structural    │   │                │
│ Features      │   │ Tesseract OCR +│
│               │   │ CamemBERT      │
│ 98.24% acc    │   │ 94.12% acc     │
└──────┬────────┘   └─────┬──────────┘
       │                  │
       └────────┬─────────┘
                │
         ┌──────▼──────────┐
         │ Multimodal      │
         │ Fusion Layer    │
         │ (Weighted Vote) │
         │ 98.5% acc       │
         └──────┬──────────┘
                │
         ┌──────▼──────────┐
         │ Classification  │
         │ Result + Conf.  │
         └─────────────────┘
```

## 🔬 Technical Approach

### 1️⃣ Computer Vision Pipeline

**Hybrid CNN Architecture:**
- **ResNet50 Backbone** (Pre-trained on ImageNet)
  - Fine-tuned on 74 training documents
  - Feature extraction: 2048 → 128 dimensions
  - Transfer learning enables high accuracy with limited data

- **Structural Feature Extraction** (9 hand-crafted features)
  - Aspect ratio detection (card vs A4 format)
  - Face detection via Haar Cascades (for ID cards)
  - Table structure detection using Hough Line Transform
  - Dominant color analysis (blue/green backgrounds)
  - Text density computation
  - Signature zone detection
  - Numeric content density

**Model Details:**
```python
Input: 224×224 RGB image
├─ ResNet50 → 128 visual features
├─ MLP(9 structural features) → 32 features
└─ Concat(128+32) → FC(160→64→5) → Softmax
```

**Performance:** 98.24% accuracy, 0.3s inference time

### 2️⃣ Natural Language Processing Pipeline

**Stage 1: Text Extraction**
- Tesseract OCR 5.x with French language model
- PDF → 300 DPI images (optimal OCR quality)
- Character Error Rate: 2-15% depending on scan quality

**Stage 2: Baseline Classifier (Keyword Matching)**
- Manual keyword dictionaries (13-15 terms/class)
- Weighted scoring system (discriminative terms: kWh ×3, m³ ×3, CNSS ×2)
- Result: 70% accuracy, 33% confidence
- Purpose: Establishes baseline for improvement measurement

**Stage 3: Fine-tuned CamemBERT**
- **Model:** `camembert-base` (110M parameters)
- **Why CamemBERT?** French-specific BERT trained on 138GB French text (vs English BERT)
- **Training:**
  - Platform: Google Colab (T4 GPU)
  - Epochs: 10
  - Batch size: 4
  - Learning rate: 5e-5
  - Training time: 45 minutes
- **Tokenization:** SentencePiece subword (32k vocab, max 512 tokens)
- **Result:** 94.12% accuracy (+24.12 points vs baseline), 85% confidence

**Why It Works With Only 92 Documents:**  
Transfer learning! CamemBERT already understands French. We only train the final classification layer to recognize our 5 specific document categories.

### 3️⃣ Multimodal Fusion Strategy

**Late Fusion with Confidence-Weighted Voting:**
```python
Weights: CV=40%, Structural=30%, NLP=30%

Decision Rules (priority order):
1. Perfect Agreement: CV & NLP agree + both conf > 70%
   → Average confidences

2. CV Strong: CV conf > 85% + structural validation > 60%
   → 0.7×CV_conf + 0.3×structural_score

3. NLP Strong: NLP conf > 85% + structural validation > 60%
   → 0.7×NLP_conf + 0.3×structural_score

4. Weighted Vote: Σ(weight × confidence) for each class
   → Argmax
```

**Why Late Fusion?**
- Modularity: Models train independently
- Interpretability: Can inspect each modality's contribution
- Error compensation: CV corrects NLP mistakes and vice-versa

**Example:** Electricity bill misclassified as water bill by NLP (shared vocabulary: "consumption", "meter") → CV 98% confident → Fusion corrects to electricity bill.

## 📊 Results & Performance

| Model | Accuracy | Avg Confidence | Inference Time |
|-------|----------|----------------|----------------|
| CV (ResNet50 + Structural) | 98.24% | 85% | 0.3s |
| NLP Baseline (Keywords) | 70.00% | 33% | 0.1s |
| NLP Fine-tuned (CamemBERT) | 94.12% | 85% | 0.5s |
| **Multimodal Fusion** | **98.50%** | **88%** | **0.8s** |

**Confusion Matrix Analysis:**
- Main errors: Water bills ↔ Electricity bills (similar layouts)
- Zero confusion: CNIE vs other classes (distinct visual features)

**Robustness:**
- Clean documents: 100% accuracy
- Slightly blurred: 94% accuracy
- Heavily degraded (>20% OCR error): 70% accuracy

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch 2.0, torchvision
- **NLP:** Hugging Face Transformers, CamemBERT
- **Computer Vision:** OpenCV, PIL
- **OCR:** Tesseract 5.x, pytesseract
- **PDF Processing:** pdf2image, Poppler
- **Training:** Google Colab (T4 GPU)
- **Deployment:** CPU inference (no GPU required)

## 📁 Project Structure
```
classification-documents-admin/
├── data/
│   ├── raw/                    # Original PDFs (92 documents)
│   └── processed/              # Converted images (300 DPI)
│
├── src/
│   ├── preprocessing/
│   │   ├── pdf_to_images.py   # PDF conversion pipeline
│   │   └── pdf_to_text.py     # OCR extraction
│   │
│   ├── cv_module/
│   │   ├── hybrid_model.py    # ResNet50 + structural features
│   │   ├── gabarit_detector.py # Structural feature extraction
│   │   ├── train.py            # Training loop
│   │   └── cv_pipeline.py      # Inference pipeline
│   │
│   ├── nlp/
│   │   ├── keyword_classifier.py      # Baseline
│   │   ├── camembert_classifier.py    # Fine-tuned model
│   │   └── training/
│   │       └── camembert_finetuning.ipynb
│   │
│   └── fusion/
│       └── multimodal_fusion.py # Late fusion logic
│
├── models/
│   ├── cv/
│   │   └── best_model.pth      # Trained CV model (100 MB)
│   └── nlp/
│       └── camembert_finetuned/ # Fine-tuned CamemBERT (440 MB)
│
├── main.py                      # Main inference script
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/nada-alaoui-as/classification-documents-admin.git
cd classification-documents-admin

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (550 MB)
# Available at: https://drive.google.com/drive/folders/1o9BVqmkTJUmcEABZhpHl4b5PYHCTaQSH
# Extract to models/ directory
```

### Usage
```python
from src.cv_module.cv_pipeline import CVPipeline
from src.nlp.nlp_pipeline import NLPPipeline
from src.fusion.multimodal_fusion import MultimodalFusion

# Initialize pipelines
cv_pipeline = CVPipeline(model_path="models/cv/best_model.pth")
nlp_pipeline = NLPPipeline(tesseract_path="path/to/tesseract")
fusion = MultimodalFusion()

# Classify document
cv_result = cv_pipeline.process_pdf("document.pdf")
nlp_result = nlp_pipeline.process_pdf("document.pdf")

final_result = fusion.fuse(
    cv_prediction=(cv_result['category'], cv_result['confidence']),
    gabarit_scores=cv_result['gabarit_scores'],
    nlp_prediction=(nlp_result['category'], nlp_result['confidence'])
)

print(f"Category: {final_result['category']}")
print(f"Confidence: {final_result['confidence']:.2%}")
```

## 🎓 Key Learnings & Research Insights

1. **Transfer Learning Efficacy:** Pre-trained models (ResNet50, CamemBERT) achieve 94-98% accuracy with only 92 training examples—impossible from scratch.

2. **Multimodal Complementarity:** Visual and textual modalities capture orthogonal information. Fusion provides robustness on ambiguous cases (+20% on edge cases).

3. **Structural Priors Matter:** Hand-crafted features (aspect ratios, colors) remain valuable alongside deep learning. They provide interpretability and handle distribution shifts better.

4. **OCR Bottleneck:** Text extraction quality (2-15% CER) is the main limiting factor. Future work: Document denoising, correction models.

5. **Data Efficiency:** With smart architecture choices, high performance is achievable with limited labeled data—critical for practical deployment.

## 🔮 Future Work

**Short-term:**
- [ ] Dataset expansion to 200+ documents per class
- [ ] Image preprocessing (denoising, deskewing, binarization)
- [ ] Multi-page document handling
- [ ] REST API deployment (Flask/FastAPI)

**Medium-term:**
- [ ] Named Entity Recognition (extract dates, amounts, names)
- [ ] Fraud detection module
- [ ] Multilingual support (Arabic, English via mBERT/XLM-R)
- [ ] Active learning pipeline for continuous improvement

**Long-term:**
- [ ] End-to-end document understanding (LayoutLM, Donut)
- [ ] Question-answering on documents
- [ ] Production deployment as SaaS platform

## 📈 Impact & Applications

**Potential Deployment Scenarios:**
- **Government agencies:** Automated mail sorting
- **Banks:** Customer document processing (KYC)
- **Insurance:** Claims file organization
- **Accounting firms:** Tax document categorization

**Estimated Impact:** 80% reduction in manual processing time, handling 4500+ documents/hour in batch mode.

## 👥 Team

- **Nada ALAOUI** - NLP Module & Multimodal Fusion
- **Salma AMAL** - Computer Vision Module & Multimodal Fusion

**Course:** Computer Vision & Natural Language Processing  
**Institution:** ENSAM Rabat
**Instructor:** Prof. CHEFIRA  
**Date:** January 2026

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Pre-trained models: ResNet50 (He et al., 2015), CamemBERT (Martin et al., 2019)
- OCR: Tesseract (Google)
- Frameworks: PyTorch, Hugging Face Transformers

---

**⭐ If you find this project useful, please consider starring the repository!**
