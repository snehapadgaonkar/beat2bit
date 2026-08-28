# %% [markdown]
# # Milestone 3 & 4: Quantization and Pruning
# This notebook takes the trained baseline FP32 model and applies 
# Magnitude Pruning and INT8 Post-Training Quantization.

# %%
import tensorflow as tf
import numpy as np
import os
import tensorflow_model_optimization as tfmot

# Project root (parent of this script's directory) so paths resolve from any cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# %% [markdown]
# ### 1. Load the Dataset and Baseline Model
# %%
X_train = np.load(os.path.join(PROJECT_ROOT, 'data', 'processed', 'X_train.npy'))
X_test = np.load(os.path.join(PROJECT_ROOT, 'data', 'processed', 'X_test.npy'))
y_test = np.load(os.path.join(PROJECT_ROOT, 'data', 'processed', 'y_test.npy'))

model_path = os.path.join(PROJECT_ROOT, 'models', 'baseline_cnn.keras')
baseline_model = tf.keras.models.load_model(model_path)
fp32_size = os.path.getsize(model_path)
print(f"Loaded Baseline FP32 Model. Size: {fp32_size / 1024:.2f} KB")

# %% [markdown]
# ### 2. Magnitude Pruning
# We define a pruning schedule to achieve 70% sparsity (setting 70% of weights to zero).
# %%
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Calculate end step based on batch size and epochs
batch_size = 128
epochs = 4
validation_split = 0.1
num_images = X_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
          initial_sparsity=0.20,
          final_sparsity=0.70,
          begin_step=0,
          end_step=end_step)
}

pruned_model = prune_low_magnitude(baseline_model, **pruning_params)
pruned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the pruned model
import tempfile
logdir = tempfile.mkdtemp()
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

print("Fine-tuning pruned model...")
pruned_model.fit(
    X_train, y_train, # Note: y_train must be loaded in the actual run
    batch_size=batch_size, 
    epochs=epochs, 
    validation_split=validation_split,
    callbacks=callbacks,
    verbose=1
)

# Strip pruning wrappers to get a clean model
model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)

# %% [markdown]
# ### 3. INT8 Quantization + TFLite Conversion
# We quantize the stripped pruned model down to 8-bit integers.
# %%
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(500):
        yield [tf.cast(input_value, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()

quantized_model_path = os.path.join(PROJECT_ROOT, 'models', 'model_pruned_quantized.tflite')
with open(quantized_model_path, 'wb') as f:
    f.write(tflite_model_quant)

# %% [markdown]
# ### 4. Compression Results
# %%
int8_size = os.path.getsize(quantized_model_path)
print("\n=== Size Comparison ===")
print(f"Original FP32 Model Size : {fp32_size / 1024:.2f} KB")
print(f"Pruned+INT8 Model Size   : {int8_size / 1024:.2f} KB")
print(f"Compression Ratio        : {fp32_size / int8_size:.2f}x smaller!")
