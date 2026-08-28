"""
Unit tests for evaluation metrics module.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_binary_classification_metrics():
    """Test binary classification metrics calculation."""
    print("Testing binary classification metrics...")

    try:
        from src.evaluation.metrics import calculate_binary_classification_metrics

        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0])

        # Calculate metrics
        metrics = calculate_binary_classification_metrics(y_true, y_pred)

        # Check that expected metrics are present
        expected_keys = ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity',
                        'f1_score', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

        # Check specific values
        # TP=4, FP=2, TN=4, FN=0
        assert metrics['true_positives'] == 4
        assert metrics['false_positives'] == 2
        assert metrics['true_negatives'] == 4
        assert metrics['false_negatives'] == 0
        assert abs(metrics['accuracy'] - 0.8) < 1e-10  # (4+4)/10
        assert abs(metrics['precision'] - 0.6666) < 1e-3  # 4/(4+2)
        assert abs(metrics['recall'] - 1.0) < 1e-10    # 4/(4+0)
        assert abs(metrics['f1_score'] - 0.8) < 1e-10  # 2*(0.6666*1.0)/(0.6666+1.0)

        print("PASS: Binary classification metrics calculated correctly")
        return True

    except Exception as e:
        print(f"FAIL: Binary classification metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ami_metrics():
    """Test AAMI EC57 metrics calculation."""
    print("Testing AAMI EC57 metrics...")

    try:
        from src.evaluation.metrics import calculate_ami_metrics

        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0])

        # Calculate metrics
        metrics = calculate_ami_metrics(y_true, y_pred)

        # Check that expected metrics are present
        expected_keys = ['ami_sensitivity', 'ami_positive_predictivity', 'ami_effectiveness',
                        'ami_false_positive_rate', 'ami_false_negative_rate',
                        'ami_true_positives', 'ami_false_positives', 'ami_true_negatives', 'ami_false_negatives']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

        # Check specific values (same as binary classification but with AAMI names)
        # TP=4, FP=2, TN=4, FN=0
        assert metrics['ami_true_positives'] == 4
        assert metrics['ami_false_positives'] == 2
        assert metrics['ami_true_negatives'] == 4
        assert metrics['ami_false_negatives'] == 0
        assert abs(metrics['ami_sensitivity'] - 1.0) < 1e-10    # 4/(4+0)
        assert abs(metrics['ami_positive_predictivity'] - 0.6666) < 1e-3  # 4/(4+2)
        assert abs(metrics['ami_effectiveness'] - 0.8165) < 1e-3  # sqrt(1.0*0.6666)
        assert abs(metrics['ami_false_positive_rate'] - 0.3333) < 1e-3  # 2/(2+4)
        assert abs(metrics['ami_false_negative_rate'] - 0.0) < 1e-10  # 0/(4+0)

        print("PASS: AAMI EC57 metrics calculated correctly")
        return True

    except Exception as e:
        print(f"FAIL: AAMI EC57 metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluate_model_predictions():
    """Test model prediction evaluation."""
    print("Testing model prediction evaluation...")

    try:
        from src.evaluation.metrics import evaluate_model_predictions

        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        # Probabilities for positive class
        y_pred_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.7, 0.3, 0.6, 0.8, 0.1])

        # Evaluate with default threshold (0.5)
        metrics = evaluate_model_predictions(y_true, y_pred_prob, threshold=0.5)

        # Should have both binary and AAMI metrics
        assert 'accuracy' in metrics
        assert 'ami_sensitivity' in metrics
        assert 'ami_positive_predictivity' in metrics
        assert 'ami_effectiveness' in metrics
        assert 'threshold' in metrics

        # With threshold 0.5, predictions should be: [0,1,1,1,0,1,0,1,1,0]
        # Same as our test data in binary classification test
        assert abs(metrics['accuracy'] - 0.8) < 1e-10
        assert abs(metrics['ami_sensitivity'] - 1.0) < 1e-10
        assert abs(metrics['ami_positive_predictivity'] - 0.6666) < 1e-3

        print("PASS: Model prediction evaluation works correctly")
        return True

    except Exception as e:
        print(f"FAIL: Model prediction evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimal_threshold():
    """Test optimal threshold calculation."""
    print("Testing optimal threshold calculation...")

    try:
        from src.evaluation.metrics import calculate_optimal_threshold

        # Create test data where we know the optimal threshold
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        # Probabilities that clearly separate classes except for one ambiguous case
        y_pred_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.4, 0.8, 0.15])

        # Test optimizing for F1 score
        opt_thresh, opt_metrics = calculate_optimal_threshold(
            y_true, y_pred_prob, metric_to_optimize='f1_score'
        )

        # Check that we got reasonable values
        assert 0 <= opt_thresh <= 1
        assert 'f1_score' in opt_metrics
        assert 'accuracy' in opt_metrics
        assert 'ami_sensitivity' in opt_metrics
        assert 'optimal_threshold' in opt_metrics
        assert abs(opt_metrics['optimal_threshold'] - opt_thresh) < 1e-10

        # Test optimizing for AMI effectiveness
        opt_thresh2, opt_metrics2 = calculate_optimal_threshold(
            y_true, y_pred_prob, metric_to_optimize='ami_effectiveness'
        )

        assert 0 <= opt_thresh2 <= 1
        assert 'ami_effectiveness' in opt_metrics2
        assert 'optimal_threshold' in opt_metrics2

        print("PASS: Optimal threshold calculation works correctly")
        return True

    except Exception as e:
        print(f"FAIL: Optimal threshold test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mcnemar_test():
    """Test McNemar's test for comparing classifiers."""
    print("Testing McNemar's test...")

    try:
        from src.evaluation.metrics import calculate_mcnemar_test

        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        # Model 1: mostly correct
        y_pred_model1 = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        # Model 2: makes different errors
        y_pred_model2 = np.array([0, 1, 1, 1, 0, 0, 0, 0, 1, 0])

        # Calculate McNemar's test
        results = calculate_mcnemar_test(y_true, y_pred_model1, y_pred_model2)

        # Check that expected results are present
        expected_keys = ['mcnemar_statistic', 'mcnemar_pvalue', 'contingency_table',
                        'both_correct', 'model1_only_correct', 'model2_only_correct', 'both_wrong']
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

        # Check contingency table values
        # Model 1 correct: all correct (10/10)
        # Model 2 correct: indices 0,2,3,4,6,7,8,9 = 8/10 correct (wrong at indices 1,5)
        # Both correct: indices 0,2,3,4,6,7,8,9 = 8 cases
        # Model 1 only correct: indices 1,5 = 2 cases (Model1 correct, Model2 wrong)
        # Model 2 only correct: none = 0 cases
        # Both wrong: none = 0 cases
        assert results['both_correct'] == 8
        assert results['model1_only_correct'] == 2
        assert results['model2_only_correct'] == 0
        assert results['both_wrong'] == 0

        print("PASS: McNemar's test works correctly")
        return True

    except Exception as e:
        print(f"FAIL: McNemar's test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_agreement():
    """Test model agreement calculation."""
    print("Testing model agreement...")

    try:
        from src.evaluation.metrics import calculate_model_agreement

        # Create test predictions
        y_pred_model1 = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred_model2 = np.array([0, 1, 1, 1, 0, 0, 0, 0, 1, 0])

        # Calculate agreement
        results = calculate_model_agreement(y_pred_model1, y_pred_model2)

        # Check that expected results are present
        expected_keys = ['overall_agreement', 'cohen_kappa']
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

        # Check specific values
        # Agreements at indices: 0,2,3,4,6,7,8,9 = 8 out of 10 = 0.8
        assert abs(results['overall_agreement'] - 0.8) < 1e-10

        # Kappa should be less than raw agreement due to chance agreement
        assert results['cohen_kappa'] <= results['overall_agreement']
        assert results['cohen_kappa'] >= -1  # Kappa ranges from -1 to 1

        print("PASS: Model agreement calculation works correctly")
        return True

    except Exception as e:
        print(f"FAIL: Model agreement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all evaluation metrics tests."""
    print("=" * 50)
    print("Beat2Bit Evaluation Metrics Tests")
    print("=" * 50)

    tests = [
        test_binary_classification_metrics,
        test_ami_metrics,
        test_evaluate_model_predictions,
        test_optimal_threshold,
        test_mcnemar_test,
        test_model_agreement
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests

    print("=" * 50)
    print(f"Evaluation Metrics Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("PASS: All evaluation metrics tests passed!")
        return 0
    else:
        print("FAIL: Some evaluation metrics tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())