# Summer-2025-ECE-597-Group8

# 🎣 Phishing Email Detection – Capstone Project

This is a full-stack machine learning project for detecting phishing emails.
We use traditional ML models (e.g., Naive Bayes, Decision Trees) trained on manually engineered and TF-IDF features extracted from real phishing email data.

---

## ✅ Project Stages

## 🧭 Project Stages

| Stage                        | Status       | Description                                                                |
| ---------------------------- | ------------ | -------------------------------------------------------------------------- |
| 1. Literature Review         | ✅ Completed | Studied prior research on phishing detection and email analysis techniques |
| 2. Data Understanding        | ✅ Completed | Explored email structure: subject, body, headers, common patterns          |
| 3. Preprocessing & Features  | ✅ Completed | Cleaned emails, extracted structural + TF-IDF features                     |
| 4. Traditional ML Modeling   | 🚀 Ongoing   | Train Naive Bayes, Logistic Regression, Random Forest                      |
| 5. Model Evaluation          | 🚀 Ongoing   | Evaluate using AUC, confusion matrix, balanced accuracy                    |
| 6. Optimization & Validation | 🔜 Upcoming  | Address class imbalance, apply cross-validation, hold-out test set         |
| 7. Advanced: Embedding Model | ✅ Completed | Replace TF-IDF with dense semantic embeddings (e.g., Word2Vec, BERT)       |
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

## 📊 Model Progress

### ✅ Logistic Regression

**Accuracy**: `95%`

**Confusion Matrix**:
```
[[704  31]
 [ 24 276]]
```

**Classification Report**:

| Metric         | Ham (0) | Spam (1) | Avg / Total |
|----------------|---------|----------|-------------|
| Precision      | 0.97    | 0.90     | 0.93        |
| Recall         | 0.96    | 0.92     | 0.94        |
| F1-Score       | 0.96    | 0.91     | 0.94        |
| Support        | 735     | 300      | 1035        |

---

### ✅ Decision Tree

**Accuracy**: `92%`

**Confusion Matrix**:
```
[[684  51]
 [ 29 271]]
```

**Classification Report**:

| Metric         | Ham (0) | Spam (1) | Avg / Total |
|----------------|---------|----------|-------------|
| Precision      | 0.96    | 0.84     | 0.90        |
| Recall         | 0.93    | 0.90     | 0.92        |
| F1-Score       | 0.94    | 0.87     | 0.91        |
| Support        | 735     | 300      | 1035        |

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt

```

### 2. Run features extractor

```bash
python src/phishing_email_preprocessing.py
```

Outputs:
	•	features/features.csv
	•	features/labels.csv (labels = 1 for now)

## 🌿 Git Branch Naming Guidelines

To keep our collaboration organized and readable, we follow a simple and flexible branch naming strategy:

### ✅ Current Naming Style

We use `yourname/task` or `yourname/purpose` format:

- Examples:
  - `lily/extractor`
  - `alex/train_model`
  - `bob/literature_review`

### 🧪 Tips

- Avoid vague names like `changes`, `update`, `fix`
- Keep it short and meaningful
- Use lowercase and hyphens for readability

### ✅ Workflow

1. Pull latest changes:`git pull origin main`
2. Create a branch:`git checkout -b yourname/task`
3. Make changes and commit:`git add .``git commit -m "Meaningful message"`
4. Push branch:`git push origin yourname/task`
5. Open Pull Request on GitHub

- **Base**: `main`
- **Compare**: `yourname/task` (your feature branch)
- Add a clear and concise description of what you did

6. Share the PR link in the team chat and wait for at least one teammate to approve before merging

---
