# AI-Legal: Legal Document Analysis & Violation Prediction

An AI-powered system for analyzing legal documents and predicting ECHR violations using LegalBERT with web interface.

## 🎯 Overview

- **📄 PDF & Text Processing**: Extract and analyze legal documents
- **🔍 Violation Prediction**: LegalBERT-based ECHR violation detection
- **📝 Intelligent Summarization**: T5-based case summarization
- **🔗 Similar Case Retrieval**: Find relevant precedents via Google Search
- **🌐 Web Interface**: Flask web application

## 🚀 Quick Start

### Installation

```bash

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
# Visit http://localhost:8000
```

## 📊 Model Info

- **Architecture**: LegalBERT (nlpaueb/legal-bert-base-uncased)
- **Test Accuracy**: ~91%
- **F1 Score**: ~0.91
- **Training Data**: 10,000 balanced ECHR cases
- **Max Sequence Length**: 512 tokens

## 📁 Structure

```
AI-Legal/
├── app.py                          # Flask application
├── notebook9ff22b1d39.ipynb        # Training notebook
├── requirements.txt                # Dependencies
├── models/
│   ├── legalbert_echr_model/       # Trained model
│   ├── legalbert_echr_model_best/  # Best checkpoint
│   └── legalbert_echr_model_last/  # Last checkpoint
├── templates/
│   └── index.html                  # Web UI
├── static/
│   └── styles.css                  # Styling
└── New folder/all-data/            # Training dataset
```

## 🔧 Configuration

Edit `app.py` to set:
```python
GOOGLE_API_KEY = "your-api-key"
CSE_ID = "your-cse-id"
MODEL_PATH = "models/legalbert_echr_model"
```

## 📚 API Endpoints

### POST `/analyze`
Analyze text or PDF document
```json
{
  "text": "Case facts...",
  "pdf": "<file>"
}
```

### POST `/analyze_url`
Analyze content from URL
```json
{
  "url": "https://example.com"
}
```

## 📈 Performance

| Metric | Score |
|--------|-------|
| Accuracy | 91.2% |
| Precision | 0.890 |
| Recall | 0.923 |
| F1 Score | 0.906 |

## 🛠️ Tech Stack

- **Backend**: Flask, PyTorch, Transformers
- **NLP Models**: LegalBERT, T5
- **Data**: Pandas, NumPy, scikit-learn
- **PDF**: PyMuPDF
- **Frontend**: HTML5, CSS3, JavaScript

## 📝 Training Pipeline

1. Data loading from ECHR JSON dataset
2. Text extraction (facts, arguments, law sections)
3. Data cleaning & leakage removal
4. Dataset balancing (5K violation + 5K no-violation)
5. Tokenization with LegalBERT
6. Training (2 epochs, lr=2e-5)
7. Fine-tuning (2 epochs, lr=1e-5)

## ⚙️ Key Hyperparameters

- Batch Size: 4
- Learning Rate: 2e-5 → 1e-5
- Max Length: 512 tokens
- Optimizer: AdamW
- Warmup: 10% of total steps

## 🚨 Limitations

- ECHR-specific training
- Max document length: 512 tokens
- Requires fine-tuning for other jurisdictions
- Google Search API rate limits

## 📖 Notebook Sections

The training notebook includes:
1. Data Loading & Exploration
2. Label Extraction & Distribution
3. Text Field Extraction
4. Data Cleaning & Deduplication
5. Leakage Phrase Removal
6. Dataset Balancing
7. Text Normalization
8. Train/Val/Test Split (80/10/10)
9. Tokenization
10. PyTorch Dataset & DataLoader
11. Model Loading & Training
12. Evaluation & Metrics
13. Model Saving
14. Inference Functions
15. Fine-tuning
16. Model Reloading


## 📄 License

Educational and research purposes.

---

**Last Updated**: November 2025  
**Status**: Active
