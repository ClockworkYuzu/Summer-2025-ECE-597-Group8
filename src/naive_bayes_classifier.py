import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load features and labels
X = pd.read_csv("../features/features.csv")
y = pd.read_csv("../features/labels.csv")["label"]

# 2. Split into training and test sets (same split as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)


pipe = Pipeline([
    ("scale", StandardScaler()),
    ("nb", GaussianNB())
])

param_grid = {
    "nb__var_smoothing": np.logspace(-12, -1, 12)
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("Best smoothing:", grid.best_params_)
print("CV accuracy:", grid.best_score_)

# Final model
best_model = grid.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
# Define the labels for the classes
target_names = ['Ham (0)', 'Spam (1)']

# Evaluate the model on the test set
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Print the classification report with custom target names
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model to disk
model_path = '../models/naive_bayes_model.joblib'
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

