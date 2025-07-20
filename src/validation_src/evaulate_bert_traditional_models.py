import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)
import joblib

# ================================
# 1. Load BERT embeddings & labels
# ================================
X = pd.read_csv("../../features/bert_embeddings.csv")
y = pd.read_csv("../../features/labels.csv")["label"]

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 2. Load models
# ================================
models = {
    "Logistic Regression (BERT)": "../../models/bert_lr_model.joblib",
    "SVM (BERT)": "../../models/bert_svm_model.joblib",
    "Random Forest (BERT)": "../../models/bert_rf_model.joblib",
    "Naive Bayes (BERT)": "../../models/bert_nb_model.joblib"
}

# ================================
# 3. Evaluate each model
# ================================
target_names = ["Ham (0)", "Spam (1)"]

for name, path in models.items():
    print(f"\n=== {name} ===")
    model = joblib.load(path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    if y_prob is not None:
        print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))