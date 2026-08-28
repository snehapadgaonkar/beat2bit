# Simple test to verify imports work
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Testing imports...")

try:
    # Test benchmarking modules
    from src.benchmarking import model_evaluator, complexity_analyzer, latency_benchmarker, comparison_engine, report_generator
    print("SUCCESS: Benchmarking modules imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import benchmarking modules: {e}")

try:
    # Test experiment tracking modules
    from src.utils import config, logging
    from src.experiments import tracker
    print("SUCCESS: Experiment tracking modules imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import experiment tracking modules: {e}")

try:
    # Test that we can create a simple model
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(180, 1)),
        tf.keras.layers.Conv1D(16, kernel_size=7, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print("SUCCESS: TensorFlow model creation successful")
except Exception as e:
    print(f"ERROR: Failed to create TensorFlow model: {e}")

print("Import testing complete.")