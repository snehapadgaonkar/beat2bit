# %% [markdown]
# # Milestone 2: Baseline 1D CNN
# This notebook builds and trains the baseline Convolutional Neural Network 
# using the extracted MIT-BIH dataset.

# %%
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import os
import time

# %% [markdown]
# ### 1. Load the Preprocessed Dataset
# %%
print("Loading preprocessed dataset...")
X_train = np.load('../data/processed/X_train.npy')
y_train = np.load('../data/processed/y_train.npy')
X_test = np.load('../data/processed/X_test.npy')
y_test = np.load('../data/processed/y_test.npy')

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# %% [markdown]
# ### 2. Build the 1D CNN Architecture
# A lightweight architecture strictly designed for Edge device constraints.
# %%
model = tf.keras.Sequential([
    tf.keras.Input(shape=(180, 1)),
    tf.keras.layers.Conv1D(16, kernel_size=7, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# %% [markdown]
# ### 3. Handle Class Imbalance & Train
# We compute class weights since Normal beats heavily outnumber Abnormal beats.
# %%
neg, pos = np.bincount(y_train)
total = neg + pos
class_weight = {0: (1 / neg)*(total/2.0), 1: (1 / pos)*(total/2.0)}
print(f"Class Weights: {class_weight}")

print("Starting Training...")
start_time = time.time()
history = model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=128, 
    validation_split=0.1, 
    class_weight=class_weight, 
    verbose=1
)
print(f"Training finished in {time.time() - start_time:.2f} seconds.")

# %% [markdown]
# ### 4. Evaluate and Save
# %%
print("\nEvaluating on Test Set (DS2)...")
y_pred_prob = model.predict(X_test, batch_size=128)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Abnormal (1)']))

os.makedirs('../models', exist_ok=True)
model_path = '../models/baseline_cnn.keras'
model.save(model_path)
print(f"\nBaseline Model Size: {os.path.getsize(model_path) / 1024:.2f} KB")
