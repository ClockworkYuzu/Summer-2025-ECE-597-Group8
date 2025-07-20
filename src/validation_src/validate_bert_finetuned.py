import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# ================================
# 1. Load dataset
# ================================
df = pd.read_csv("../../data/spam_ham_dataset_cleaned.csv")
label_col = "Label"
text_col = "clean_text"

if df[label_col].dtype == object:
    df[label_col] = df[label_col].replace({"ham": 0, "spam": 1}).astype(int)

X = df[text_col].astype(str).tolist()
y = df[label_col].astype(int).tolist()

from sklearn.model_selection import train_test_split
_, X_test_texts, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 2. Load saved model & tokenizer
# ================================
MODEL_DIR = "../../models/bert_finetuned_spam_best"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)

# ================================
# 3. Tokenize test set
# ================================
encodings = tokenizer(
    X_test_texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="tf"
)
X_test_dataset = tf.data.Dataset.from_tensor_slices(dict(encodings)).batch(32)

# ================================
# 4. Predict on test set
# ================================
y_pred_logits = model.predict(X_test_dataset)
y_pred = np.argmax(y_pred_logits.logits, axis=1)
y_test = np.array(y_test)

# ================================
# 5. Show Confusion Matrix & Report
# ================================
print("\n✅ Confusion Matrix (BERT Fine-tuning):")
print(confusion_matrix(y_test, y_pred))

print("\n✅ Classification Report (BERT Fine-tuning):")
print(classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"]))