# Bangla News Headline Generation

An end-to-end NLP application that automatically generates Bangla news headlines from full Bangla news articles using a fine-tuned **mT5-small** model.

---

## Project Overview

This project fine-tunes Google's multilingual T5 (mT5-small) model on Bangla newspaper data to generate concise, accurate headlines from full article text. It includes a complete training pipeline, evaluation metrics, REST API, and a modern web dashboard.

---

## Dataset Description

- **Name:** Bangla Newspaper Dataset
- **Format:** JSON (array of objects with `title` and `content` fields)
- **Total Samples Used:** 50,000
- **Train Split:** 45,000
- **Validation Split:** 5,000
- **Filter:** Articles with content < 100 characters are excluded

### Dataset Structure
```json
{
  "title": "News Headline (Target)",
  "content": "Full Bangla Article (Input)"
}
```

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | google/mt5-small |
| Task | Text-to-Text Generation |
| Input Format | `headline: <full_article>` |
| Output | `<generated_headline>` |
| Framework | PyTorch + HuggingFace Transformers |
| Language | Bangla (bn) |

---

## Training Procedure

| Setting | Value |
|---------|-------|
| Epochs | 1 |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| Optimizer | AdamW |
| Evaluation Strategy | Every 1000 steps |
| Max Input Length | 512 tokens |
| Max Target Length | 64 tokens |
| FP16 | Enabled (if GPU available) |

---

## Installation Guide

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- pip

### Setup
```bash
# Clone the project
git clone <repository-url>
cd Bangla_dataset

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for BLEU evaluation)
python -c "import nltk; nltk.download('punkt')"
```

---

## Usage Guide

### 1. Prepare Dataset
Place your `data.json` file in the `dataset/` directory.

### 2. Preprocess Data
```bash
python training/preprocess.py
```

### 3. Train Model
```bash
python training/train.py
```
Training will save the model to `model/saved_model/` and metrics to `reports/`.

### 4. Evaluate Model
```bash
python training/evaluate.py
```
Generates `evaluation_report.json` and `evaluation_report.pdf` in `reports/`.

### 5. Run the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6. Access the Web UI
Open `http://localhost:8000` in your browser.

---

## API Documentation

### POST /generate
Generate a headline from an article.

**Request:**
```json
{
  "article": "Full Bangla article text...",
  "actual_headline": "Optional actual headline for comparison"
}
```

**Response:**
```json
{
  "headline": "Generated Bangla headline",
  "actual_headline": "Actual headline (if provided)",
  "input_length": 1234,
  "device": "cuda",
  "timestamp": "2024-01-01 12:00:00"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2024-01-01 12:00:00"
}
```

### GET /model-info
Returns model metadata and configuration.

### GET /predictions
Returns prediction history.

### DELETE /predictions
Clears prediction history.

### GET /predictions/export
Downloads prediction history as CSV.

### GET /training-metrics
Returns training metrics.

**Swagger Docs:** Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Project Structure

```
project/
├── app/
│   ├── main.py          # FastAPI application
│   ├── inference.py     # Inference engine
│   ├── utils.py         # Utility functions
│   └── config.py        # Configuration
├── training/
│   ├── train.py         # Training script
│   ├── preprocess.py    # Data preprocessing
│   └── evaluate.py      # Evaluation pipeline
├── model/
│   └── saved_model/     # Trained model files
├── dataset/
│   └── data.json        # Raw dataset
├── frontend/
│   ├── index.html       # Web dashboard
│   ├── style.css        # Styles (dark theme)
│   └── script.js        # Frontend logic
├── reports/
│   ├── training_metrics.json
│   ├── evaluation_report.json
│   ├── evaluation_report.pdf
│   └── loss_curve.png
├── requirements.txt
└── README.md
```

---

## Google Colab / Kaggle Usage

```python
# Install dependencies
!pip install -r requirements.txt

# Upload dataset to dataset/data.json or use Kaggle dataset API

# Train
!python training/train.py

# Evaluate
!python training/evaluate.py

# Run API (use ngrok for public URL)
!pip install pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
public_url = ngrok.connect(8000)
print(public_url)

!uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Evaluation Metrics

The model is evaluated using:
- **ROUGE-1:** Unigram overlap
- **ROUGE-2:** Bigram overlap
- **ROUGE-L:** Longest common subsequence
- **BLEU:** Bilingual evaluation understudy

---

## Screenshots

The web interface features:
- 🏠 Home page with project overview and statistics
- ✨ Headline Generator with real-time generation
- 📊 Model Information with training metrics
- 📋 Prediction History with export capabilities
- ℹ️ About page with API documentation

---

## Future Scope

- Fine-tune larger models (mT5-base, mT5-large)
- Add multi-document summarization
- Implement abstractive + extractive hybrid approach
- Add user authentication
- Deploy on cloud (AWS/GCP/Azure)
- Add model A/B testing
- Implement streaming generation
- Add support for other Bangla NLP tasks (NER, sentiment)
- Build mobile app with React Native

---

## License

This project is for educational and research purposes.
