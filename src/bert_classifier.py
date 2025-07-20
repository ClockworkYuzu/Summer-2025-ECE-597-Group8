import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 1. Load BERT embeddings and labels
X = pd.read_csv("../features/bert_embeddings.csv")
y = pd.read_csv("../features/labels.csv")["label"]

print(f"BERT embeddings shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# 2. Split into training and test sets (same split as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the labels for the classes
target_names = ['Ham (0)', 'Spam (1)']

# 3. Test multiple classifiers with BERT embeddings

# Logistic Regression
print("\n=== Logistic Regression with BERT ===")
lr_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

lr_param_grid = {
    "lr__C": [0.1, 1, 10, 100]
}

lr_grid = GridSearchCV(lr_pipe, lr_param_grid, cv=5, scoring="accuracy", n_jobs=-1)
lr_grid.fit(X_train, y_train)

print("Best parameters:", lr_grid.best_params_)
print("CV accuracy:", lr_grid.best_score_)

lr_pred = lr_grid.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, lr_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# Save the model
joblib.dump(lr_grid.best_estimator_, '../models/bert_lr_model.joblib')
print("Model saved to ../models/bert_lr_model.joblib")

# SVM
print("\n=== SVM with BERT ===")
svm_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svm", SVC())
])

svm_param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["linear", "rbf"]
}

svm_grid = GridSearchCV(svm_pipe, svm_param_grid, cv=5, scoring="accuracy", n_jobs=-1)
svm_grid.fit(X_train, y_train)

print("Best parameters:", svm_grid.best_params_)
print("CV accuracy:", svm_grid.best_score_)

svm_pred = svm_grid.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, svm_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))

# Save the model
joblib.dump(svm_grid.best_estimator_, '../models/bert_svm_model.joblib')
print("Model saved to ../models/bert_svm_model.joblib")

# Random Forest
print("\n=== Random Forest with BERT ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, rf_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Save the model
joblib.dump(rf_model, '../models/bert_rf_model.joblib')
print("Model saved to ../models/bert_rf_model.joblib")

# Naive Bayes
print("\n=== Naive Bayes with BERT ===")
nb_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("nb", GaussianNB())
])

nb_param_grid = {
    "nb__var_smoothing": np.logspace(-12, -1, 12)
}

nb_grid = GridSearchCV(nb_pipe, nb_param_grid, cv=5, scoring="accuracy", n_jobs=-1)
nb_grid.fit(X_train, y_train)

print("Best parameters:", nb_grid.best_params_)
print("CV accuracy:", nb_grid.best_score_)

nb_pred = nb_grid.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, nb_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, nb_pred))

# Save the model
joblib.dump(nb_grid.best_estimator_, '../models/bert_nb_model.joblib')
print("Model saved to ../models/bert_nb_model.joblib")

print("\n=== Summary ===")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_pred):.4f}")