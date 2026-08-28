"""
Verification script for Beat2Bit implementation.
Tests the core components of the benchmarking suite and experiment tracking system.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")

    try:
        # Test benchmarking modules
        from src.benchmarking import model_evaluator, complexity_analyzer, latency_benchmarker, comparison_engine, report_generator
        print("PASS: Benchmarking modules imported successfully")
    except Exception as e:
        print(f"FAIL Failed to import benchmarking modules: {e}")
        return False

    try:
        # Test experiment tracking modules
        from src.utils import config, logging
        from src.experiments import tracker
        print("PASS: Experiment tracking modules imported successfully")
    except Exception as e:
        print(f"FAIL: Failed to import experiment tracking modules: {e}")
        return False

    try:
        # Test that we can create a simple model
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
        print("PASS: TensorFlow model creation successful")
    except Exception as e:
        print(f"FAIL Failed to create TensorFlow model: {e}")
        return False

    return True

def test_model_evaluator():
    """Test the model evaluator functionality."""
    print("\nTesting model evaluator...")

    try:
        from src.benchmarking.model_evaluator import (
            calculate_binary_classification_metrics,
            calculate_ami_metrics,
            evaluate_model_predictions,
            calculate_optimal_threshold
        )

        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.7, 0.3, 0.6, 0.8, 0.1])

        # Test binary classification metrics
        binary_metrics = calculate_binary_classification_metrics(y_true, y_pred)
        assert 'accuracy' in binary_metrics
        assert 'f1_score' in binary_metrics
        print("PASS: Binary classification metrics calculated")

        # Test AMI metrics
        ami_metrics = calculate_ami_metrics(y_true, y_pred)
        assert 'ami_sensitivity' in ami_metrics
        assert 'ami_positive_predictivity' in ami_metrics
        print("PASS: AMI metrics calculated")

        # Test evaluation with probabilities
        eval_metrics = evaluate_model_predictions(y_true, y_pred_prob)
        assert 'accuracy' in eval_metrics
        assert 'ami_sensitivity' in eval_metrics
        print("PASS: Probability-based evaluation completed")

        # Test optimal threshold calculation
        opt_thresh, opt_metrics = calculate_optimal_threshold(y_true, y_pred_prob)
        assert isinstance(opt_thresh, float)
        assert 0 <= opt_thresh <= 1
        print("PASS: Optimal threshold calculated")

        return True
    except Exception as e:
        print(f"FAIL Model evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complexity_analyzer():
    """Test the complexity analyzer functionality."""
    print("\nTesting complexity analyzer...")

    try:
        from src.benchmarking.complexity_analyzer import (
            count_model_parameters,
            estimate_model_size_mb,
            calculate_flops,
            analyze_model_complexity
        )

        # Create test model
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

        # Test parameter counting
        params = count_model_parameters(model)
        assert 'total_parameters' in params
        assert params['total_parameters'] > 0
        print(f"PASS: Parameter counting: {params['total_parameters']:,} parameters")

        # Test size estimation
        size_fp32 = estimate_model_size_mb(model, 'float32')
        size_int8 = estimate_model_size_mb(model, 'int8')
        assert size_fp32['size_mb'] > 0
        assert size_int8['size_mb'] > 0
        print(f"PASS: Size estimation: FP32={size_fp32['size_mb']:.3f} MB, INT8={size_int8['size_mb']:.3f} MB")

        # Test FLOPs calculation
        flops = calculate_flops(model, batch_size=1)
        assert 'total_flops' in flops
        assert flops['total_flops'] > 0
        print(f"PASS: FLOPs calculation: {flops['total_flops']:,}")

        # Test comprehensive analysis
        complexity = analyze_model_complexity(model, batch_size=16)
        assert 'parameters' in complexity
        assert 'memory_size' in complexity
        assert 'computational_complexity' in complexity
        print("PASS: Comprehensive complexity analysis completed")

        return True
    except Exception as e:
        print(f"FAIL Complexity analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_latency_benchmarker():
    """Test the latency benchmarker functionality."""
    print("\nTesting latency benchmarker...")

    try:
        from src.benchmarking.latency_benchmarker import (
            benchmark_inference_latency,
            benchmark_single_inference_latency,
            simulate_edge_device_latency
        )

        # Create test model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(180, 1)),
            tf.keras.layers.Conv1D(8, kernel_size=5, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Create test data
        input_data = np.random.randn(20, 180, 1).astype(np.float32)

        # Test latency benchmarking (short run for testing)
        latency_results = benchmark_inference_latency(
            model, input_data,
            batch_sizes=[1, 4],
            n_warmup=2,
            n_measurements=5
        )

        assert 'latency_stats' in latency_results
        assert 'batch_size_1' in latency_results['latency_stats']
        assert 'mean_latency_ms' in latency_results['latency_stats']['batch_size_1']
        print("PASS Latency benchmarking completed")

        # Test single sample latency
        single_latency = benchmark_single_inference_latency(
            model, input_data, n_warmup=2, n_measurements=5
        )
        assert 'mean_latency_ms' in single_latency
        assert single_latency['mean_latency_ms'] > 0
        print(f"PASS Single sample latency: {single_latency['mean_latency_ms']:.2f} ms")

        # Test edge device simulation
        edge_metrics = simulate_edge_device_latency(
            model, input_data, target_fps=50.0, n_measurements=5
        )
        assert 'achieved_mean_latency_ms' in edge_metrics
        assert 'meets_realtime_requirement' in edge_metrics
        print(f"PASS Edge device simulation: {edge_metrics['achieved_mean_latency_ms']:.2f} ms, "
              f"meets realtime: {edge_metrics['meets_realtime_requirement']}")

        return True
    except Exception as e:
        print(f"FAIL Latency benchmarker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_manager():
    """Test the configuration manager functionality."""
    print("\nTesting configuration manager...")

    try:
        from src.utils.config import ConfigManager, ExperimentConfig

        # Create config manager
        config_manager = ConfigManager(config_dir="test_configs")

        # Create a test configuration
        config = config_manager.create_config(
            experiment_name="test_experiment",
            description="A test experiment for verification",
            epochs=5,
            batch_size=16,
            learning_rate=0.01
        )

        # Validate configuration
        is_valid, errors = config_manager.validate_config(config)
        assert is_valid, f"Configuration validation failed: {errors}"
        print("PASS Configuration validation passed")

        # Save configuration
        config_path = config_manager.save_config(config, "test_config")
        assert os.path.exists(config_path)
        print(f"PASS Configuration saved to: {config_path}")

        # Load configuration back
        loaded_config = config_manager.load_config(config_path)
        assert loaded_config.experiment_name == "test_experiment"
        assert loaded_config.epochs == 5
        print("PASS Configuration loaded successfully")

        # Clean up
        import shutil
        if os.path.exists("test_configs"):
            shutil.rmtree("test_configs")

        return True
    except Exception as e:
        print(f"FAIL Config manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_experiment_tracker():
    """Test the experiment tracker functionality."""
    print("\nTesting experiment tracker...")

    try:
        from src.experiments.tracker import ExperimentTracker

        # Create test tracker
        tracker = ExperimentTracker(base_dir="test_experiments")

        # Create experiment
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 5,
            "model_architecture": "test_cnn"
        }

        exp_id = tracker.create_experiment(
            name="test_experiment",
            config=config,
            description="A test experiment",
            tags=["test", "verification"]
        )

        assert exp_id is not None
        assert len(exp_id) > 0
        print(f"PASS Experiment created: {exp_id}")

        # Update status
        tracker.update_experiment_status(exp_id, "running")
        print("PASS Experiment status updated to running")

        # Log metrics
        tracker.log_metrics(exp_id, {"accuracy": 0.85, "loss": 0.4}, step=1)
        tracker.log_metrics(exp_id, {"accuracy": 0.90, "loss": 0.2}, step=2)
        print("PASS Metrics logged")

        # Complete experiment
        tracker.update_experiment_status(exp_id, "completed")
        print("PASS Experiment status updated to completed")

        # Retrieve experiment
        experiment = tracker.get_experiment(exp_id)
        assert experiment is not None
        assert experiment["experiment_id"] == exp_id
        assert experiment["status"] == "completed"
        assert len(experiment["metrics_history"]) == 2
        print("PASS Experiment retrieved successfully")

        # List experiments
        experiments = tracker.list_experiments()
        assert len(experiments) >= 1
        print(f"PASS Listed {len(experiments)} experiments")

        # Clean up
        tracker.delete_experiment(exp_id)
        import shutil
        if os.path.exists("test_experiments"):
            shutil.rmtree("test_experiments")

        return True
    except Exception as e:
        print(f"FAIL Experiment tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between components."""
    print("\nTesting component integration...")

    try:
        # Test that we can create a model, evaluate it, analyze complexity, and benchmark latency
        from src.benchmarking.model_evaluator import evaluate_model_predictions
        from src.benchmarking.complexity_analyzer import analyze_model_complexity
        from src.benchmarking.latency_benchmarker import benchmark_single_inference_latency

        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(180, 1)),
            tf.keras.layers.Conv1D(8, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Create test data
        X_test = np.random.randn(10, 180, 1).astype(np.float32)
        y_test = np.random.randint(0, 2, 10)

        # Get predictions
        y_pred_prob = model.predict(X_test, verbose=0)

        # Evaluate model
        metrics = evaluate_model_predictions(y_test, y_pred_prob)
        assert 'accuracy' in metrics
        print(f"PASS Model evaluation: accuracy = {metrics['accuracy']:.3f}")

        # Analyze complexity
        complexity = analyze_model_complexity(model, batch_size=1)
        assert complexity['parameters']['total_parameters'] > 0
        print(f"PASS Complexity analysis: {complexity['parameters']['total_parameters']:,} parameters")

        # Benchmark latency
        latency = benchmark_single_inference_latency(model, X_test, n_warmup=2, n_measurements=5)
        assert latency['mean_latency_ms'] > 0
        print(f"PASS Latency benchmarking: {latency['mean_latency_ms']:.2f} ms")

        return True
    except Exception as e:
        print(f"FAIL Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Beat2Bit Implementation Verification")
    print("=" * 60)

    tests = [
        test_imports,
        test_model_evaluator,
        test_complexity_analyzer,
        test_latency_benchmarker,
        test_config_manager,
        test_experiment_tracker,
        test_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests

    print("=" * 60)
    print(f"Verification Results: {passed}/{total} tests passed")

    if passed == total:
        print("PASS: All tests passed! Implementation is ready for use.")
        return 0
    else:
        print("FAIL: Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())