import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

# 1. Load features and labels
X = pd.read_csv("../features/features.csv")
y = pd.read_csv("../features/labels.csv")["label"]

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print("✅ Logistic Regression Model Evaluation:\n")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Save the model
dump(model, "../models/logistic_regression_model.joblib")
print("✅ Model saved to '../models/logistic_regression_model.joblib'")