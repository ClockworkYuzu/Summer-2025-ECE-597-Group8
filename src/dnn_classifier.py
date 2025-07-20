import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ================================
# 1. Load BERT embeddings and labels
# ================================
X = pd.read_csv("../features/bert_embeddings.csv")       # BERT sentence embeddings (e.g., shape: [n_samples, 768])
y = pd.read_csv("../features/labels.csv")["label"]       # Labels: 0 = Ham, 1 = Spam

print(f"BERT embeddings shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split into training and testing sets (80/20 split, stratified by class)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (important for DNN to converge faster)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 2. Define DNN model
# ================================
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Input layer (BERT embedding dimension, e.g., 768)
    layers.Dense(256, activation='relu'),     # Fully connected layer with 256 neurons and ReLU activation
    layers.Dropout(0.3),                      # Dropout to prevent overfitting (30% of neurons dropped)
    layers.Dense(128, activation='relu'),     # Hidden layer with 128 neurons
    layers.Dropout(0.3),                      # Dropout layer again
    layers.Dense(64, activation='relu'),      # Hidden layer with 64 neurons
    layers.Dense(1, activation='sigmoid')     # Output layer: 1 neuron with Sigmoid for binary classification
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Adam optimizer with learning rate 0.0001
    loss='binary_crossentropy',                               # Binary cross-entropy for 0/1 classification
    metrics=['accuracy']                                      # Monitor accuracy during training
)

model.summary()  # Print model architecture

# ================================
# 3. Train the model
# ================================
early_stop = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)  # Stop early if validation loss doesn't improve for 3 epochs

history = model.fit(
    X_train, y_train,
    validation_split=0.2,   # 20% of training set used as validation set
    epochs=20,              # Maximum 20 training epochs
    batch_size=32,          # Mini-batch size
    callbacks=[early_stop], # Apply early stopping
    verbose=1               # Show training progress
)

# ================================
# 4. Evaluate on test set
# ================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")

# ================================
# 5. Save the trained model
# ================================
model.save('../models/bert_dnn_model.h5')
print("Model saved to ../models/bert_dnn_model.h5")