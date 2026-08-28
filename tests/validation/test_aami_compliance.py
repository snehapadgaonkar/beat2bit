"""
AAMI EC57 compliant validation tests for Beat2Bit arrhythmia detection.
Implements standards from ANSI/AAMI/ISO 5720-2012 and EC57:2012 for
performance reporting of cardiac electrocardiogram monitors.
"""

import unittest
import numpy as np
from typing import Tuple, Dict
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.benchmarking.model_evaluator import (
    calculate_binary_classification_metrics,
    calculate_ami_metrics,
    calculate_optimal_threshold,
    evaluate_model_predictions
)


class TestAMICompliance(unittest.TestCase):
    """Test AAMI EC57 compliance for arrhythmia detection performance metrics."""

    def setUp(self):
        """Set up test data."""
        # Create predictable test data
        self.y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1])
        self.y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0])

        # Calculate expected values manually for verification
        # Confusion matrix for these arrays is TN: 5, FP: 1, FN: 2, TP: 4.
        self.expected_tn = 5
        self.expected_fp = 1
        self.expected_fn = 2
        self.expected_tp = 4

    def test_confusion_matrix_calculation(self):
        """Test that confusion matrix is calculated correctly."""
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()

        self.assertEqual(tn, self.expected_tn)
        self.assertEqual(fp, self.expected_fp)
        self.assertEqual(fn, self.expected_fn)
        self.assertEqual(tp, self.expected_tp)

    def test_ami_sensitivity_calculation(self):
        """Test AMI sensitivity (Se) calculation: TP / (TP + FN)"""
        metrics = calculate_ami_metrics(self.y_true, self.y_pred)
        expected_sensitivity = self.expected_tp / (self.expected_tp + self.expected_fn)

        self.assertAlmostEqual(metrics['ami_sensitivity'], expected_sensitivity, places=4)
        self.assertAlmostEqual(metrics['ami_sensitivity'], 4 / 6, places=4)  # 4/(4+2)

    def test_ami_positive_predictivity_calculation(self):
        """Test AMI positive predictivity (+P) calculation: TP / (TP + FP)"""
        metrics = calculate_ami_metrics(self.y_true, self.y_pred)
        expected_ppv = self.expected_tp / (self.expected_tp + self.expected_fp)

        self.assertAlmostEqual(metrics['ami_positive_predictivity'], expected_ppv, places=4)
        self.assertAlmostEqual(metrics['ami_positive_predictivity'], 4 / 5, places=4)  # 4/(4+1)

    def test_ami_effectiveness_calculation(self):
        """Test AMI effectiveness (E) calculation: sqrt(Se * +P)"""
        metrics = calculate_ami_metrics(self.y_true, self.y_pred)
        se = self.expected_tp / (self.expected_tp + self.expected_fn)
        ppv = self.expected_tp / (self.expected_tp + self.expected_fp)
        expected_effectiveness = np.sqrt(se * ppv)

        self.assertAlmostEqual(metrics['ami_effectiveness'], expected_effectiveness, places=4)
        self.assertAlmostEqual(metrics['ami_effectiveness'], np.sqrt(4 / 6 * 4 / 5), places=4)  # sqrt(Se * +P)

    def test_binary_classification_metrics(self):
        """Test standard binary classification metrics."""
        metrics = calculate_binary_classification_metrics(self.y_true, self.y_pred)

        # Accuracy = (TP + TN) / Total
        expected_accuracy = (self.expected_tp + self.expected_tn) / len(self.y_true)
        self.assertAlmostEqual(metrics['accuracy'], expected_accuracy, places=4)

        # Precision = TP / (TP + FP)
        expected_precision = self.expected_tp / (self.expected_tp + self.expected_fp)
        self.assertAlmostEqual(metrics['precision'], expected_precision, places=4)

        # Recall/Sensitivity = TP / (TP + FN)
        expected_recall = self.expected_tp / (self.expected_tp + self.expected_fn)
        self.assertAlmostEqual(metrics['recall'], expected_recall, places=4)
        self.assertEqual(metrics['sensitivity'], expected_recall)  # Alias check

        # Specificity = TN / (TN + FP)
        expected_specificity = self.expected_tn / (self.expected_tn + self.expected_fp)
        self.assertAlmostEqual(metrics['specificity'], expected_specificity, places=4)

        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        self.assertAlmostEqual(metrics['f1_score'], expected_f1, places=4)

    def test_evaluate_model_predictions_with_probabilities(self):
        """Test evaluation with probability outputs."""
        # Create probability predictions
        y_pred_prob = np.array([0.1, 0.2, 0.6, 0.3, 0.8, 0.9, 0.4, 0.7, 0.2, 0.8, 0.3, 0.4])

        metrics = evaluate_model_predictions(self.y_true, y_pred_prob, threshold=0.5)

        # Should produce same results as binary predictions with threshold 0.5
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)
        binary_metrics = calculate_binary_classification_metrics(self.y_true, y_pred_binary)
        ami_metrics = calculate_ami_metrics(self.y_true, y_pred_binary)

        # Check that key metrics match
        self.assertAlmostEqual(metrics['accuracy'], binary_metrics['accuracy'], places=4)
        self.assertAlmostEqual(metrics['f1_score'], binary_metrics['f1_score'], places=4)
        self.assertAlmostEqual(metrics['ami_sensitivity'], ami_metrics['ami_sensitivity'], places=4)
        self.assertAlmostEqual(metrics['ami_positive_predictivity'], ami_metrics['ami_positive_predictivity'], places=4)

    def test_optimal_threshold_calculation(self):
        """Test optimal threshold calculation using Youden's J statistic."""
        # Create probability predictions with clear separation
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        optimal_threshold, optimal_metrics = calculate_optimal_threshold(y_true, y_pred_prob)

        # The data is perfectly separable, so any threshold between 0.3 and 0.7
        # yields sensitivity = specificity = 1. The algorithm returns the first
        # threshold that maximizes Youden's J (0.7 for this ordering).
        self.assertGreaterEqual(optimal_threshold, 0.3)
        self.assertLessEqual(optimal_threshold, 0.7)

        # At the optimal threshold the classification must be perfect.
        self.assertEqual(optimal_metrics['ami_sensitivity'], 1.0)
        self.assertEqual(optimal_metrics['specificity'], 1.0)

    def test_edge_cases(self):
        """Test edge cases like all positive or all negative predictions."""
        # All negative predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])

        metrics = calculate_ami_metrics(y_true, y_pred)

        # Sensitivity should be 0 (no true positives)
        self.assertEqual(metrics['ami_sensitivity'], 0.0)
        # Positive predictivity should be 0 (no positive predictions)
        self.assertEqual(metrics['ami_positive_predictivity'], 0.0)

        # All positive predictions
        y_pred = np.array([1, 1, 1, 1])
        metrics = calculate_ami_metrics(y_true, y_pred)

        # Sensitivity should be 1 (all actual positives caught)
        self.assertEqual(metrics['ami_sensitivity'], 1.0)
        # Positive predictivity should be 0.5 (2 TP out of 4 predictions)
        self.assertEqual(metrics['ami_positive_predictivity'], 0.5)

    def test_ami_metrics_with_perfect_predictions(self):
        """Test AMI metrics when predictions are perfect."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        metrics = calculate_ami_metrics(y_true, y_pred)

        # All metrics should be 1.0 for perfect predictions
        self.assertEqual(metrics['ami_sensitivity'], 1.0)
        self.assertEqual(metrics['ami_positive_predictivity'], 1.0)
        self.assertEqual(metrics['ami_effectiveness'], 1.0)

    def test_ami_metrics_with_worst_predictions(self):
        """Test AMI metrics when predictions are completely wrong."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        metrics = calculate_ami_metrics(y_true, y_pred)

        # All metrics should be 0.0 for completely wrong predictions
        self.assertEqual(metrics['ami_sensitivity'], 0.0)
        self.assertEqual(metrics['ami_positive_predictivity'], 0.0)
        self.assertEqual(metrics['ami_effectiveness'], 0.0)


class TestClinicalValidationStandards(unittest.TestCase):
    """Test compliance with clinical validation standards for arrhythmia detectors."""

    def test_ami_thresholds_interpretation(self):
        """Test interpretation of AMI EC57 thresholds for acceptable performance."""
        # According to AAMI EC57, acceptable performance for arrhythmia detectors:
        # Sensitivity (Se) >= 0.75
        # Positive Predictivity (+P) >= 0.70

        # Create test case that meets minimum acceptable performance
        # We need: TP/(TP+FN) >= 0.75 and TP/(TP+FP) >= 0.70
        # Let's design: TP=6, FN=2 (Se=0.75), TP=6, FP=2.57 -> need FP<=2 for +P>=0.70
        # So: TP=6, FN=2, FP=2 -> Se=0.75, +P=0.75

        y_true = np.array([0]*4 + [1]*8)  # 4 negatives, 8 positives
        y_pred = np.array([0,0,1,1] + [0,0,1,1,1,1,1,1])  # 2FP, 6TP

        metrics = calculate_ami_metrics(y_true, y_pred)

        self.assertGreaterEqual(metrics['ami_sensitivity'], 0.75)
        self.assertGreaterEqual(metrics['ami_positive_predictivity'], 0.70)

    def test_effectiveness_interpretation(self):
        """Test that effectiveness combines sensitivity and predictivity appropriately."""
        # High sensitivity, low predictivity
        y_true = np.array([0,0,0,0,1,1,1,1])  # 4N, 4P
        y_pred = np.array([1,1,0,0,1,1,1,1])  # 2FP, 2TN, 4TP, 0FN

        metrics = calculate_ami_metrics(y_true, y_pred)

        se = metrics['ami_sensitivity']  # Should be 1.0 (4TP/4P)
        ppv = metrics['ami_positive_predictivity']  # Should be 0.667 (4TP/6P)
        eff = metrics['ami_effectiveness']  # Should be sqrt(1.0 * 0.667) = 0.816

        self.assertAlmostEqual(se, 1.0, places=3)
        self.assertAlmostEqual(ppv, 0.667, places=3)
        self.assertAlmostEqual(eff, 0.816, places=3)

        # Effectiveness should be geometric mean, not arithmetic mean
        arithmetic_mean = (se + ppv) / 2
        self.assertLess(eff, arithmetic_mean)  # Geometric <= Arithmetic

    def test_performance_reporting_format(self):
        """Test that performance reporting follows AAMI EC57 format."""
        y_true = np.array([0,0,0,0,1,1,1,1])
        y_pred = np.array([0,0,1,0,1,1,0,1])

        metrics = calculate_ami_metrics(y_true, y_pred)

        # AAMI EC57 requires reporting of:
        # - Sensitivity (Se)
        # - Positive Predictivity (+P)
        # - Often also reports Effectiveness (E)

        self.assertIn('ami_sensitivity', metrics)
        self.assertIn('ami_positive_predictivity', metrics)
        self.assertIn('ami_effectiveness', metrics)

        # Values should be in [0, 1] range
        self.assertGreaterEqual(metrics['ami_sensitivity'], 0.0)
        self.assertLessEqual(metrics['ami_sensitivity'], 1.0)
        self.assertGreaterEqual(metrics['ami_positive_predictivity'], 0.0)
        self.assertLessEqual(metrics['ami_positive_predictivity'], 1.0)
        self.assertGreaterEqual(metrics['ami_effectiveness'], 0.0)
        self.assertLessEqual(metrics['ami_effectiveness'], 1.0)


if __name__ == '__main__':
    unittest.main()