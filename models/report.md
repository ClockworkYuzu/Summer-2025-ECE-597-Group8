# 📊 Phishing Email Detection Report

## 🔍 Models Introduction

### 1. Logistic Regression

Logistic Regression is a fundamental statistical model used for binary classification tasks, which means it's used to predict an outcome that can only have two possible values (e.g., yes/no, true/false, 0/1). Despite its name, it's a classification algorithm, not a regression one.

### 2. Decision Tree

A Decision Tree is a versatile and intuitive machine learning algorithm used for both classification and regression tasks. It creates a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes is a classification technique based on Bayes' Theorem. It belongs to the family of Naive Bayes classifiers, which are known for their simplicity and effectiveness, particularly with high-dimensional data.

---

## 📈 Performance Comparison

| Model                         | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
| ----------------------------- | -------- | ---------------- | ------------- | --------------- |
| **Logistic Regression** | 95%      | 0.90             | 0.92          | 0.91            |
| **Decision Tree**       | 92%      | 0.84             | 0.90          | 0.87            |
| **Naive Bayes**         | 93%      | 0.84             | 0.96          | 0.89            |

---

## 🧠 Key Insights

- **Logistic Regression** achieved the highest overall accuracy and balanced performance across all metrics. It is likely the most robust baseline.
- **Naive Bayes** had the highest **recall** for spam detection (0.96), making it effective when minimizing false negatives is critical (e.g., in security settings).
- **Decision Tree** offered interpretable decisions but underperformed slightly in precision compared to the others.
