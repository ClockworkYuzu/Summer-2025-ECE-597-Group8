import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================
# 1. Load dataset and labels
# ================================
df = pd.read_csv("../data/spam_ham_dataset_cleaned.csv")
print("Columns in CSV:", df.columns)

label_col = "Label"
text_col = "clean_text"

if df[label_col].dtype == object:
    df[label_col] = df[label_col].replace({"ham": 0, "spam": 1}).astype(int)

X = df[text_col].astype(str).tolist()
y = df[label_col].astype(int).tolist()

print(f"Texts: {len(X)}, Labels: {len(y)} (using '{label_col}')")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 2. Tokenization for BERT
# ================================
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts, labels):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )
    return tf.data.Dataset.from_tensor_slices((dict(encodings), tf.convert_to_tensor(labels)))

X_train = tokenize(X_train, y_train).shuffle(1000).batch(32)
X_test = tokenize(X_test, y_test).batch(32)

# ================================
# 3. Define a fine-tuning BERT model
# ================================
model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

num_train_steps = len(X_train) * 8
optimizer, lr_schedule = create_optimizer(
    init_lr=2e-5, num_warmup_steps=0, num_train_steps=num_train_steps
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ================================
# 4. Train with improved EarlyStopping
# ================================
best_val_acc = 0
patience = 4
wait = 0
max_epochs = 8

best_model_dir = "../models/bert_finetuned_spam_best"
os.makedirs(best_model_dir, exist_ok=True)

history_data = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

for epoch in range(max_epochs):
    print(f"\nEpoch {epoch + 1}/{max_epochs}")
    history_epoch = model.fit(
        X_train,
        validation_data=X_test,
        epochs=1,
        verbose=1
    )

    for k in history_epoch.history:
        history_data[k].extend(history_epoch.history[k])

    val_acc = history_epoch.history['val_accuracy'][-1]
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        wait = 0
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        print(f"✅ Best model updated (val_accuracy: {best_val_acc:.4f})")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

class FakeHistory:
    def __init__(self, h): self.history = h
history = FakeHistory(history_data)

# ================================
# 5. Evaluate on test set
# ================================
loss, acc = model.evaluate(X_test)

print(f"\n✅ Test Accuracy (BERT Fine-tuned Optimized): {acc:.4f}")
print(f"Best Validation Accuracy during training: {best_val_acc:.4f}")

# ================================
# 6. Visualization: Save Training Curve
# ================================
os.makedirs("../imgs", exist_ok=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('BERT Fine-tuning Accuracy (Final Optimized)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('BERT Fine-tuning Loss (Final Optimized)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("../imgs/bert_finetuning_training_curve_final_optimized.png", dpi=300)
print("✅ Training curve saved successfully to ../imgs/bert_finetuning_training_curve_final_optimized.png")
