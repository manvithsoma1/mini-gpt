# 🤖 End-to-End Intelligent Question Answering System
### NLP Final Project | Academic + Industry Grade

---

## 📌 Project Overview

This project implements a **Hybrid NLP + LLM Question Answering System** that demonstrates the full NLP pipeline — from raw text preprocessing to deep learning-based QA — deployed as an interactive Streamlit web application.

**Dataset**: Stanford SQuAD v1.1 (Wikipedia-based QA pairs)

---

## 📁 Project Structure

```
mini chatgpt/
├── nlp_qa_project.ipynb     ← Full notebook (Google Colab / Jupyter)
├── app.py                   ← Streamlit interactive UI
├── requirements.txt         ← All dependencies
└── README.md                ← This file
```

---

## 🔧 Setup Instructions

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download Required NLP Models

Run this once (already included in notebook Cell 1):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import spacy
# Run this in terminal:
# python -m spacy download en_core_web_sm
```

### Step 3 — Run the Notebook

Open `nlp_qa_project.ipynb` in:
- **Google Colab** (recommended for GPU access)
- **Jupyter Lab / Notebook**

Run all cells top-to-bottom. The notebook will:
1. Load and explore the SQuAD dataset
2. Run the full NLP preprocessing pipeline
3. Build TF-IDF and Word2Vec features
4. Build the Information Retrieval system
5. Run both QA approaches
6. Evaluate and compare methods

---

## 🚀 Running the Streamlit UI

### Step 1 — Install Streamlit
```bash
pip install streamlit
```

### Step 2 — Launch the App
```bash
streamlit run app.py
```

The app will open at: `http://localhost:8501`

---

## 🧪 Test Questions

Try these in the UI:
- `"What is machine learning?"`
- `"Explain overfitting"`
- `"What is a neural network?"`
- `"Who invented the telephone?"`
- `"What is the capital of France?"`

---

## 🏗️ NLP Pipeline Covered

1. Text Cleaning
2. Tokenization (Sentence + Word)
3. Normalization
4. Stopword Removal
5. Stemming vs Lemmatization
6. POS Tagging
7. Dependency Parsing

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| NLP Preprocessing | NLTK, spaCy |
| Feature Engineering | scikit-learn, gensim |
| Information Retrieval | TF-IDF + cosine similarity |
| Extractive QA | sentence scoring |
| Deep Learning QA | HuggingFace Transformers (RoBERTa) |
| UI | Streamlit |
| Data | HuggingFace Datasets (SQuAD) |

---

## 📊 Methods Compared

| Method | Speed | Accuracy | Complexity |
|---|---|---|---|
| TF-IDF Extractive | ⚡ Fast | Moderate | Low |
| BERT/RoBERTa QA | 🐢 Slower | High | High |

---

## 👤 Author

NLP Final Project — Academic Submission  
*Demonstrates end-to-end NLP pipeline + system design*
