# 🎬 IMDB Review Sentiment Classification  
Negative Review Detection with TF-IDF, Logistic Regression & LightGBM

---

## 📖 Project Overview

This project builds a machine learning pipeline to automatically detect negative movie reviews using an IMDB dataset labeled by sentiment polarity. The goal is to support content filtering and categorization by classifying reviews as positive or negative.

The target metric is **F1-score ≥ 0.85** on the test set.

---

## 🎯 Business Objective

- Classify movie reviews as **positive** or **negative**
- Achieve **F1 ≥ 0.85** on the test set
- Compare multiple models and explain performance differences
- Validate models using both test data and manually written reviews

---

## 🗂 Data

- IMDB movie reviews dataset with sentiment polarity labels  
- Target: review sentiment (positive vs negative)
- Class balance: approximately balanced across train/test splits

---

## 🛠️ Methodology

### 1️⃣ Text Preprocessing
- Text normalization
- Tokenization and cleanup
- Stopword handling
- Optional linguistic preprocessing using spaCy (normalized text variant)

### 2️⃣ Feature Engineering
- **TF-IDF vectorization** (max_features=10,000)
- Features built from:
  - normalized text (`review_norm`)
  - spaCy-processed text (`review_spacy`)

### 3️⃣ Models Trained
- **Dummy Classifier** (baseline)
- **Logistic Regression** (TF-IDF)
- **LightGBM Classifier** (TF-IDF)

### 4️⃣ Evaluation
Models were evaluated with:
- F1-score (threshold-optimized)
- Accuracy
- ROC-AUC
- Average Precision (APS)

---

## 📊 Results

| Model | Vectorization / Text Version | Test F1 | Test ROC-AUC |
|------|-------------------------------|--------:|-------------:|
| Dummy Classifier | Baseline | 0.67 | 0.50 |
| Logistic Regression | TF-IDF + stopwords (`review_norm`) | **0.88** | **0.95** |
| Logistic Regression | TF-IDF (`review_spacy`) | **0.88** | **0.95** |
| LightGBM | TF-IDF (`review_spacy`) | **0.87** | **0.95** |

All trained models exceeded the project requirement (**F1 ≥ 0.85**). Logistic Regression achieved the best overall balance of performance and simplicity.

---

## 🧪 Manual Review Testing

A small set of manually written reviews was classified using all trained models. Differences between model outputs highlight how preprocessing and model architecture influence sensitivity to wording, tone, and ambiguity.

A BERT embedding routine was also explored for small-scale experimentation (not used for full dataset training due to computational constraints).

---

## ✅ Key Takeaways

- TF-IDF + Logistic Regression provides strong and stable performance for sentiment classification.
- LightGBM achieves comparable results, with slight differences in generalization behavior.
- Baseline comparison confirms that the models provide clear predictive value beyond a dummy approach.

---

## 🧰 Tools & Technologies

- Python  
- Pandas, NumPy  
- Scikit-learn  
- LightGBM  
- NLTK / spaCy  
- Matplotlib  
