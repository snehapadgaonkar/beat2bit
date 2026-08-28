"""
Unit tests for training loops module.
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

def test_create_callbacks():
    """Test callback creation."""
    print("Testing callback creation...")

    try:
        from src.training.loops import create_callbacks

        with tempfile.TemporaryDirectory() as temp_dir:
            callbacks = create_callbacks(
                checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
                log_dir=os.path.join(temp_dir, "logs")
            )

            # Check that we have the expected callbacks
            assert len(callbacks) == 4  # ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

            # Check callback types
            callback_types = [type(cb).__name__ for cb in callbacks]
            assert 'ModelCheckpoint' in callback_types
            assert 'TensorBoard' in callback_types
            assert 'ReduceLROnPlateau' in callback_types
            assert 'EarlyStopping' in callback_types

            print("PASS: Callbacks created correctly")
            return True

    except Exception as e:
        print(f"FAIL: Callback creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_standard_training():
    """Test standard training loop."""
    print("Testing standard training...")

    try:
        from src.training.loops import train_standard
        from src.models.architectures import create_baseline_1dcnn, compile_model

        # Create and compile model
        model = create_baseline_1dcnn(input_shape=(180, 1), num_classes=2)
        model = compile_model(model, learning_rate=0.001)

        # Create small dataset for testing
        X_train = np.random.randn(20, 180, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.randn(10, 180, 1).astype(np.float32)
        y_val = np.random.randint(0, 2, 10)

        # Train for just 2 epochs for quick test
        history = train_standard(
            model, X_train, y_train, X_val, y_val,
            epochs=2, batch_size=10, verbose=0
        )

        # Check that training produced history
        assert hasattr(history, 'history')
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert 'val_loss' in history.history
        assert 'val_accuracy' in history.history

        # Check that we have values for each epoch
        assert len(history.history['loss']) == 2
        assert len(history.history['val_loss']) == 2

        print(f"PASS: Standard training completed. Loss: {history.history['loss'][-1]:.4f}")
        return True

    except Exception as e:
        print(f"FAIL: Standard training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pruning_training():
    """Test pruning-aware training loop."""
    print("Testing pruning-aware training...")

    try:
        from src.training.loops import train_with_pruning
        from src.models.architectures import create_baseline_1dcnn, compile_model

        # Create and compile model
        model = create_baseline_1dcnn(input_shape=(180, 1), num_classes=2)
        model = compile_model(model, learning_rate=0.001)

        # Create small dataset for testing
        X_train = np.random.randn(20, 180, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.randn(10, 180, 1).astype(np.float32)
        y_val = np.random.randint(0, 2, 10)

        # Train with pruning for just 2 epochs for quick test
        history, pruned_model = train_with_pruning(
            model, X_train, y_train, X_val, y_val,
            epochs=2, batch_size=10,
            initial_sparsity=0.3, final_sparsity=0.6,  # Lower sparsity for quick test
            verbose=0
        )

        # Check that training produced history
        assert hasattr(history, 'history')
        assert len(history.history['loss']) == 2

        # Check that we got a model back
        assert pruned_model is not None
        assert isinstance(pruned_model, tf.keras.Model)

        # Check that we can make predictions
        test_input = np.random.randn(5, 180, 1).astype(np.float32)
        predictions = pruned_model.predict(test_input, verbose=0)
        assert predictions.shape == (5, 2)

        print(f"PASS: Pruning training completed. Loss: {history.history['loss'][-1]:.4f}")
        return True

    except Exception as e:
        print(f"FAIL: Pruning training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qat_training():
    """Test quantization-aware training loop."""
    print("Testing quantization-aware training...")

    try:
        from src.training.loops import train_with_quantization_aware
        from src.models.architectures import create_baseline_1dcnn, compile_model

        # Create and compile model
        model = create_baseline_1dcnn(input_shape=(180, 1), num_classes=2)
        model = compile_model(model, learning_rate=0.001)

        # Create small dataset for testing
        X_train = np.random.randn(20, 180, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.randn(10, 180, 1).astype(np.float32)
        y_val = np.random.randint(0, 2, 10)

        # Train with QAT for just 2 epochs for quick test
        history, qat_model = train_with_quantization_aware(
            model, X_train, y_train, X_val, y_val,
            epochs=2, batch_size=10, verbose=0
        )

        # Check that training produced history
        assert hasattr(history, 'history')
        assert len(history.history['loss']) == 2

        # Check that we got a model back
        assert qat_model is not None
        assert isinstance(qat_model, tf.keras.Model)

        # Check that we can make predictions
        test_input = np.random.randn(5, 180, 1).astype(np.float32)
        predictions = qat_model.predict(test_input, verbose=0)
        assert predictions.shape == (5, 2)

        print(f"PASS: QAT training completed. Loss: {history.history['loss'][-1]:.4f}")
        return True

    except Exception as e:
        print(f"FAIL: QAT training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tflite_conversion():
    """Test TFLite conversion."""
    print("Testing TFLite conversion...")

    try:
        from src.training.loops import convert_to_tflite
        from src.models.architectures import create_baseline_1dcnn, compile_model

        # Create and compile model
        model = create_baseline_1dcnn(input_shape=(180, 1), num_classes=2)
        model = compile_model(model, learning_rate=0.001)

        # Create small representative dataset
        rep_data = np.random.randn(10, 180, 1).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            tflite_path = os.path.join(temp_dir, "test_model.tflite")

            # Convert to TFLite
            success = convert_to_tflite(
                model,
                output_path=tflite_path,
                representative_data=rep_data,
                inference_type="float"
            )

            assert success, "TFLite conversion should succeed"
            assert os.path.exists(tflite_path), "TFLite file should exist"

            # Check file size
            file_size = os.path.getsize(tflite_path)
            assert file_size > 0, "TFLite file should not be empty"

            print(f"PASS: TFLite conversion successful. File size: {file_size} bytes")
            return True

    except Exception as e:
        print(f"FAIL: TFLite conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_training():
    """Test unified training function."""
    print("Testing unified training function...")

    try:
        from src.training.loops import train_model
        from src.models.architectures import create_baseline_1dcnn, compile_model

        # Create and compile model
        model = create_baseline_1dcnn(input_shape=(180, 1), num_classes=2)
        model = compile_model(model, learning_rate=0.001)

        # Create small dataset for testing
        X_train = np.random.randn(20, 180, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.randn(10, 180, 1).astype(np.float32)
        y_val = np.random.randint(0, 2, 10)

        # Test standard training mode
        with tempfile.TemporaryDirectory() as temp_dir:
            results = train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=2, batch_size=10,
                training_mode="standard",
                tflite_output_path=os.path.join(temp_dir, "model.tflite"),
                save_tflite=False,  # Skip TFLite for speed
                verbose=0
            )

            assert 'model' in results
            assert 'history' in results
            assert results['training_mode'] == "standard"
            assert hasattr(results['history'], 'history')
            assert len(results['history'].history['loss']) == 2

            print("PASS: Unified training (standard mode) works")

        # Test that invalid mode raises error
        try:
            train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=1, batch_size=10,
                training_mode="invalid_mode",
                verbose=0
            )
            print("FAIL: Should have raised ValueError for invalid training mode")
            return False
        except ValueError:
            print("PASS: Correctly rejected invalid training mode")

        return True

    except Exception as e:
        print(f"FAIL: Unified training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all training loop tests."""
    print("=" * 50)
    print("Beat2Bit Training Loops Tests")
    print("=" * 50)

    tests = [
        test_create_callbacks,
        test_standard_training,
        test_pruning_training,
        test_qat_training,
        test_tflite_conversion,
        test_unified_training
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests

    print("=" * 50)
    print(f"Training Loops Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("PASS: All training loops tests passed!")
        return 0
    else:
        print("FAIL: Some training loops tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())