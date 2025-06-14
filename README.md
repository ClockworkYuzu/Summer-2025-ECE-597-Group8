# Summer-2025-ECE-597-Group8

# 🎣 Phishing Email Detection – Capstone Project

This is a full-stack machine learning project for detecting phishing emails.  
We use traditional ML models (e.g., Naive Bayes, Decision Trees) trained on manually engineered and TF-IDF features extracted from real phishing email data.

---

## ✅ Project Stages

## 🧭 Project Stages

| Stage                         | Status       | Description                                                                 |
|------------------------------|--------------|-----------------------------------------------------------------------------|
| 1. Literature Review         | ✅ Completed | Studied prior research on phishing detection and email analysis techniques |
| 2. Data Understanding        | ✅ Completed | Explored email structure: subject, body, headers, common patterns          |
| 3. Preprocessing & Features  | ✅ Completed | Cleaned emails, extracted structural + TF-IDF features                     |
| 4. Traditional ML Modeling   | 🔜 Upcoming  | Train Naive Bayes, Logistic Regression, Random Forest                     |
| 5. Model Evaluation          | 🔜 Upcoming  | Evaluate using AUC, confusion matrix, balanced accuracy                    |
| 6. Optimization & Validation | 🔜 Upcoming  | Address class imbalance, apply cross-validation, hold-out test set         |
| 7. Advanced: Embedding Model | 🔜 Planned   | Replace TF-IDF with dense semantic embeddings (e.g., Word2Vec, BERT)       |
| 8. Final Reporting / CLI     | 🔜 Optional  | Summarize findings, optionally deploy as script or CLI tool                |

## 🗂️ Project Structure

<pre><code>phishing-email-detector/
├── data/                             # Raw input data
│   └── CaptstoneProjectData_2025.csv   # Raw phishing emails
├── features/                         # Engineered features
│   ├── features.csv                     # Extracted feature matrix
│   └── labels.csv                       # Labels (currently all 1s, put in later when there are 0s)
├── models/                          # Future model storage
├── src/                             # Core Python scripts
│   ├── phishing_email_preprocessing.py  # Feature extraction
│   └── train_model.py                  # (To be added) ML model training
├── notebooks/                       # (Optional) Exploratory analysis
├── requirements.txt
└── README.md
</code></pre>

---

## 🧠 Features Extracted

### Structural (manual):
- `char_count`, `word_count`, `uppercase_ratio`, `num_urls`, `num_exclamations`

### Textual (semantic):
- Top 300 most informative words via **TF-IDF** on cleaned content

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt

```
### 2. Install dependencies

```bash
python src/phishing_email_preprocessing.py
```

Outputs:
	•	features/features.csv
	•	features/labels.csv (labels = 1 for now)