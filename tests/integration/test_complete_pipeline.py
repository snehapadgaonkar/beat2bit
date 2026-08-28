"""
Integration test for Beat2Bit complete pipeline.
Tests data loading, preprocessing, model creation, training, and evaluation working together.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_complete_pipeline():
    """Test the complete pipeline working together."""
    print("Testing complete Beat2Bit pipeline...")

    try:
        # Import all necessary modules
        from src.data.loaders import load_synthetic_data, get_input_shape, get_num_classes
        from src.data.preprocessing import preprocess_ecg_signal, extract_heartbeat_windows
        from src.models.architectures import model_factory, compile_model
        from src.training.loops import train_model
        from src.evaluation.metrics import evaluate_model_predictions, calculate_ami_metrics
        from src.benchmarking.model_evaluator import evaluate_model_predictions as eval_model_pred
        from src.benchmarking.complexity_analyzer import analyze_model_complexity
        from src.benchmarking.latency_benchmarker import benchmark_single_inference_latency

        print("PASS: All modules imported successfully")

        # Create temporary directory for test data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Create or load data
            print("\n1. Setting up test data...")

            # Create synthetic ECG-like data for testing
            np.random.seed(42)  # For reproducibility
            X_train_raw = np.random.randn(50, 180, 1).astype(np.float32)
            y_train = np.random.randint(0, 2, 50)
            X_val_raw = np.random.randn(10, 180, 1).astype(np.float32)
            y_val = np.random.randint(0, 2, 10)

            # Simulate preprocessing (in real scenario, this would come from data loader)
            X_train = np.array([preprocess_ecg_signal(signal.flatten(), fs=360).reshape(180, 1)
                               for signal in X_train_raw])
            X_val = np.array([preprocess_ecg_signal(signal.flatten(), fs=360).reshape(180, 1)
                             for signal in X_val_raw])

            print(f"   Training data shape: {X_train.shape}")
            print(f"   Validation data shape: {X_val.shape}")

            # Step 2: Create model
            print("\n2. Creating model...")
            input_shape = (180, 1)
            num_classes = 2

            model = model_factory(
                architecture='baseline',
                input_shape=input_shape,
                num_classes=num_classes,
                filters=[16, 32],
                kernel_sizes=[7, 5],
                pool_sizes=[2, 2],
                dense_units=[16],
                dropout_rate=0.3
            )

            model = compile_model(model, learning_rate=0.001)
            print(f"   Model created with {model.count_params():,} parameters")

            # Step 3: Train model (briefly for testing)
            print("\n3. Training model...")
            # Use a very small number of epochs for quick testing
            results = train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=2, batch_size=16,
                training_mode="standard",
                save_tflite=False,  # Skip TFLite for speed in testing
                verbose=0  # Silent training for cleaner output
            )

            trained_model = results['model']
            history = results['history']
            print(f"   Training completed. Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")

            # Step 4: Evaluate model
            print("\n4. Evaluating model...")
            # Get predictions
            y_val_pred_prob = trained_model.predict(X_val, verbose=0)
            if len(y_val_pred_prob.shape) == 2 and y_val_pred_prob.shape[1] == 2:
                y_val_pred_prob = y_val_pred_prob[:, 1]  # Get positive class probabilities

            # Evaluate using our metrics
            metrics = evaluate_model_predictions(y_val, y_val_pred_prob, threshold=0.5)
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   AMI Sensitivity: {metrics['ami_sensitivity']:.4f}")
            print(f"   AMI Positive Predictivity: {metrics['ami_positive_predictivity']:.4f}")
            print(f"   AMI Effectiveness: {metrics['ami_effectiveness']:.4f}")

            # Step 5: Analyze complexity
            print("\n5. Analyzing model complexity...")
            complexity = analyze_model_complexity(trained_model, batch_size=1)
            print(f"   Parameters: {complexity['parameters']['total_parameters']:,}")
            print(f"   Model size (FP32): {complexity['memory_size']['fp32_mb']:.4f} MB")
            print(f"   FLOPs: {complexity['computational_complexity']['total_flops']:,}")

            # Step 6: Benchmark latency
            print("\n6. Benchmarking latency...")
            # Use a small subset for latency testing
            test_sample = X_val[:5]  # Just 5 samples for quick test
            latency_results = benchmark_single_inference_latency(
                trained_model, test_sample, n_warmup=2, n_measurements=5
            )
            print(f"   Mean latency: {latency_results['mean_latency_ms']:.2f} ms")
            print(f"   95th percentile latency: {latency_results['p95_latency_ms']:.2f} ms")

            # Step 7: Verify everything worked
            print("\n7. Verifying results...")
            assert history is not None, "Training history should exist"
            assert len(history.history['loss']) == 2, "Should have 2 epochs of training data"
            assert trained_model is not None, "Trained model should exist"
            assert metrics['accuracy'] >= 0.0 and metrics['accuracy'] <= 1.0, "Accuracy should be valid"
            assert complexity['parameters']['total_parameters'] > 0, "Should have parameters"
            assert latency_results['mean_latency_ms'] > 0, "Latency should be positive"

            print("   All verifications passed!")

        print("\n" + "="*50)
        print("PASS: Complete pipeline test successful!")
        print("="*50)
        return True

    except Exception as e:
        print(f"FAIL: Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete pipeline integration test."""
    print("=" * 50)
    print("Beat2Bit Complete Pipeline Integration Test")
    print("=" * 50)

    success = test_complete_pipeline()

    if success:
        print("\n✓ All integration tests passed! The pipeline is working correctly.")
        return 0
    else:
        print("\n✗ Integration test failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())