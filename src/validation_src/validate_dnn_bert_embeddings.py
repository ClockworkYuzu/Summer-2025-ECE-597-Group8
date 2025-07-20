import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ================================
# 1. Load BERT embeddings & labels
# ================================
X = pd.read_csv("../../features/bert_embeddings.csv")
y = pd.read_csv("../../features/labels.csv")["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 2. Fit scaler on training set only
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 3. Load trained DNN model
# ================================
MODEL_PATH = "../../models/bert_dnn_model_v2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ================================
# 4. Predict on test set
# ================================
y_pred_probs = model.predict(X_test_scaled)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# ================================
# 5. Show Confusion Matrix & Report
# ================================
print("\n✅ Confusion Matrix (DNN with BERT Embeddings):")
print(confusion_matrix(y_test, y_pred))

print("\n✅ Classification Report (DNN with BERT Embeddings):")
print(classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"]))