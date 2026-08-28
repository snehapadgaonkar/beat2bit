"""
Unit tests for model architecture modules.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_baseline_1dcnn():
    """Test baseline 1D CNN creation."""
    print("Testing baseline 1D CNN...")

    try:
        from src.models.architectures import create_baseline_1dcnn, compile_model

        # Create model
        model = create_baseline_1dcnn(
            input_shape=(180, 1),
            num_classes=2,
            filters=[16, 32, 64],
            kernel_sizes=[7, 5, 3],
            pool_sizes=[2, 2, 2]
        )

        # Check that model was created
        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert model.name == 'baseline_1dcnn_ecg'

        # Check parameter count
        params = model.count_params()
        assert params > 0
        print(f"PASS: Baseline model created with {params:,} parameters")

        # Compile model
        model = compile_model(model, learning_rate=0.001)
        assert model.optimizer is not None
        print("PASS: Model compiled successfully")

        # Test forward pass
        test_input = np.random.randn(10, 180, 1).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        assert output.shape == (10, 2)  # Batch size x num_classes
        print("PASS: Forward pass successful")

        return True

    except Exception as e:
        print(f"FAIL: Baseline 1D CNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deeper_1dcnn():
    """Test deeper 1D CNN creation."""
    print("Testing deeper 1D CNN...")

    try:
        from src.models.architectures import create_deeper_1dcnn, compile_model

        # Create model
        model = create_deeper_1dcnn(
            input_shape=(180, 1),
            num_classes=2,
            filters=[32, 64, 128, 256],
            kernel_sizes=[9, 7, 5, 3],
            pool_sizes=[2, 2, 2, 2]
        )

        # Check that model was created
        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert model.name == 'deeper_1dcnn_ecg'

        # Check parameter count (should be more than baseline)
        params = model.count_params()
        assert params > 0
        print(f"PASS: Deeper model created with {params:,} parameters")

        # Compile model
        model = compile_model(model, learning_rate=0.001)
        assert model.optimizer is not None
        print("PASS: Model compiled successfully")

        # Test forward pass
        test_input = np.random.randn(5, 180, 1).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        assert output.shape == (5, 2)
        print("PASS: Forward pass successful")

        return True

    except Exception as e:
        print(f"FAIL: Deeper 1D CNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_residual_1dcnn():
    """Test residual 1D CNN creation."""
    print("Testing residual 1D CNN...")

    try:
        from src.models.architectures import create_residual_1dcnn, compile_model

        # Create model
        model = create_residual_1dcnn(
            input_shape=(180, 1),
            num_classes=2,
            filters=[32, 64, 128],
            kernel_size=7,
            pool_size=2,
            dense_units=[64]
        )

        # Check that model was created
        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert model.name == 'residual_1dcnn_ecg'

        # Check parameter count
        params = model.count_params()
        assert params > 0
        print(f"PASS: Residual model created with {params:,} parameters")

        # Compile model
        model = compile_model(model, learning_rate=0.001)
        assert model.optimizer is not None
        print("PASS: Model compiled successfully")

        # Test forward pass
        test_input = np.random.randn(5, 180, 1).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        assert output.shape == (5, 2)
        print("PASS: Forward pass successful")

        return True

    except Exception as e:
        print(f"FAIL: Residual 1D CNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lightweight_1dcnn():
    """Test lightweight 1D CNN creation."""
    print("Testing lightweight 1D CNN...")

    try:
        from src.models.architectures import create_lightweight_1dcnn, compile_model

        # Create model
        model = create_lightweight_1dcnn(
            input_shape=(180, 1),
            num_classes=2,
            filters=[8, 16],
            kernel_sizes=[5, 3],
            pool_sizes=[2, 2],
            dense_units=[16],
            dropout_rate=0.3,
            use_batch_norm=False
        )

        # Check that model was created
        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert model.name == 'lightweight_1dcnn_ecg'

        # Check parameter count (should be less than baseline)
        params = model.count_params()
        assert params > 0
        print(f"PASS: Lightweight model created with {params:,} parameters")

        # Compile model
        model = compile_model(model, learning_rate=0.001)
        assert model.optimizer is not None
        print("PASS: Model compiled successfully")

        # Test forward pass
        test_input = np.random.randn(5, 180, 1).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        assert output.shape == (5, 2)
        print("PASS: Forward pass successful")

        return True

    except Exception as e:
        print(f"FAIL: Lightweight 1D CNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_factory():
    """Test model factory function."""
    print("Testing model factory...")

    try:
        from src.models.architectures import model_factory, compile_model

        # Test creating each architecture type
        architectures = ['baseline', 'deeper', 'residual', 'lightweight']

        for arch in architectures:
            model = model_factory(
                architecture=arch,
                input_shape=(180, 1),
                num_classes=2,
                filters=[16, 32] if arch in ['baseline', 'lightweight'] else [32, 64],
                dropout_rate=0.3
            )

            assert model is not None
            assert isinstance(model, tf.keras.Model)
            print(f"PASS: Factory created {arch} architecture")

            # Compile and test
            model = compile_model(model)
            test_input = np.random.randn(2, 180, 1).astype(np.float32)
            output = model.predict(test_input, verbose=0)
            assert output.shape == (2, 2)

        # Test invalid architecture
        try:
            model_factory(architecture='invalid', input_shape=(180, 1), num_classes=2)
            print("FAIL: Should have raised ValueError for invalid architecture")
            return False
        except ValueError:
            print("PASS: Correctly rejected invalid architecture")

        return True

    except Exception as e:
        print(f"FAIL: Model factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_differences():
    """Test that different architectures have different characteristics."""
    print("Testing model differences...")

    try:
        from src.models.architectures import (
            create_baseline_1dcnn,
            create_deeper_1dcnn,
            create_residual_1dcnn,
            create_lightweight_1dcnn
        )

        # Create models
        baseline = create_baseline_1dcnn(input_shape=(180, 1), num_classes=2)
        deeper = create_deeper_1dcnn(input_shape=(180, 1), num_classes=2)
        residual = create_residual_1dcnn(input_shape=(180, 1), num_classes=2)
        lightweight = create_lightweight_1dcnn(input_shape=(180, 1), num_classes=2)

        # Get parameter counts
        baseline_params = baseline.count_params()
        deeper_params = deeper.count_params()
        residual_params = residual.count_params()
        lightweight_params = lightweight.count_params()

        print(f"Baseline params: {baseline_params:,}")
        print(f"Deeper params: {deeper_params:,}")
        print(f"Residual params: {residual_params:,}")
        print(f"Lightweight params: {lightweight_params:,}")

        # Verify relationships (these are approximate based on our architectures)
        # Deeper should have more params than baseline
        assert deeper_params > baseline_params, "Deeper model should have more parameters than baseline"

        # Lightweight should have fewer params than baseline
        assert lightweight_params < baseline_params, "Lightweight model should have fewer parameters than baseline"

        print("PASS: Model parameter relationships are correct")
        return True

    except Exception as e:
        print(f"FAIL: Model differences test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all model architecture tests."""
    print("=" * 50)
    print("Beat2Bit Model Architecture Tests")
    print("=" * 50)

    tests = [
        test_baseline_1dcnn,
        test_deeper_1dcnn,
        test_residual_1dcnn,
        test_lightweight_1dcnn,
        test_model_factory,
        test_model_differences
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests

    print("=" * 50)
    print(f"Model Architecture Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("PASS: All model architecture tests passed!")
        return 0
    else:
        print("FAIL: Some model architecture tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())