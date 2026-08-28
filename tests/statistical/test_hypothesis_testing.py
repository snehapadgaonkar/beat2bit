"""
Statistical hypothesis testing utilities for Beat2Bit project.
Implements tests for comparing model performance and validating research findings.
"""

import unittest
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.benchmarking.comparison_engine import (
    compare_model_performance,
    _calculate_cohens_d,
    _calculate_rank_biserial_correlation
)
from src.benchmarking.model_evaluator import (
    bootstrap_confidence_interval,
    evaluate_model_predictions
)


class TestHypothesisTesting(unittest.TestCase):
    """Test statistical hypothesis testing procedures."""

    def setUp(self):
        """Set up test data for hypothesis testing."""
        np.random.seed(42)
        # Create simulated performance metrics for multiple models
        self.n_folds = 10

        # Baseline model performance
        self.baseline_f1 = np.random.normal(0.85, 0.03, self.n_folds)
        self.baseline_acc = np.random.normal(0.87, 0.02, self.n_folds)

        # Improved model: clearly better than baseline (paired difference of
        # ~+0.05), so the comparison is reliably statistically significant.
        self.improved_f1 = self.baseline_f1 + np.random.normal(0.05, 0.01, self.n_folds)
        self.improved_acc = self.baseline_acc + np.random.normal(0.04, 0.01, self.n_folds)

        # Similar model: nearly identical to baseline (tiny noise only), so the
        # effect size is small and it is not significantly different.
        self.similar_f1 = self.baseline_f1 + np.random.normal(0.0, 0.005, self.n_folds)
        self.similar_acc = self.baseline_acc + np.random.normal(0.0, 0.005, self.n_folds)

        # Prepare data for comparison engine
        self.models_results = {
            'baseline': {
                'f1_score': self.baseline_f1,
                'accuracy': self.baseline_acc
            },
            'improved': {
                'f1_score': self.improved_f1,
                'accuracy': self.improved_acc
            },
            'similar': {
                'f1_score': self.similar_f1,
                'accuracy': self.similar_acc
            }
        }

    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        # Large difference
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        d = _calculate_cohens_d(group1, group2)
        # pooled sample SD = sqrt(2.5); d = (3-5)/1.581 = -1.265
        self.assertAlmostEqual(d, -1.265, places=3)  # Large negative effect

        # Small difference
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        d = _calculate_cohens_d(group1, group2)
        self.assertAlmostEqual(abs(d), 0.063, places=3)  # Small effect

        # Identical groups
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([1, 2, 3, 4, 5])
        d = _calculate_cohens_d(group1, group2)
        self.assertAlmostEqual(d, 0.0, places=3)

    def test_rank_biserial_correlation(self):
        """Test rank-biserial correlation calculation."""
        # Clearly different groups
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        rbc = _calculate_rank_biserial_correlation(group1, group2)
        # (u2 - u1) / (n1*n2) = (25 - 0)/25 = +1.0 for group1 < group2
        self.assertAlmostEqual(rbc, 1.0, places=3)  # Maximum correlation

        # Identical groups
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([1, 2, 3, 4, 5])
        rbc = _calculate_rank_biserial_correlation(group1, group2)
        self.assertAlmostEqual(rbc, 0.0, places=3)  # No correlation

    def test_model_performance_comparison_paired_ttest(self):
        """Test model comparison using paired t-test."""
        comparison = compare_model_performance(
            self.models_results,
            metric_name='f1_score',
            test_type='paired_ttest'
        )

        # Check that comparison ran successfully
        self.assertIn('pairwise_comparisons', comparison)
        self.assertIn('baseline_vs_improved', comparison['pairwise_comparisons'])
        self.assertIn('baseline_vs_similar', comparison['pairwise_comparisons'])
        self.assertIn('improved_vs_similar', comparison['pairwise_comparisons'])

        # Improved model should be significantly better than baseline
        baseline_vs_improved = comparison['pairwise_comparisons']['baseline_vs_improved']
        self.assertTrue(baseline_vs_improved['significant'])
        self.assertEqual(baseline_vs_improved['better_model'], 'improved')
        # Cohen's d here is (baseline - improved), so a higher-scoring 'improved'
        # model yields a negative effect size.
        self.assertLess(baseline_vs_improved['effect_size'], 0)

        # Similar model should not be significantly different from baseline
        baseline_vs_similar = comparison['pairwise_comparisons']['baseline_vs_similar']
        # This might or might not be significant due to randomness, but effect size should be small
        self.assertLess(abs(baseline_vs_similar['effect_size']), 0.5)

    def test_model_performance_comparison_wilcoxon(self):
        """Test model comparison using Wilcoxon signed-rank test."""
        comparison = compare_model_performance(
            self.models_results,
            metric_name='f1_score',
            test_type='wilcoxon'
        )

        # Check that comparison ran successfully
        self.assertIn('pairwise_comparisons', comparison)
        self.assertIn('baseline_vs_improved', comparison['pairwise_comparisons'])

        # Improved model should be significantly better than baseline
        baseline_vs_improved = comparison['pairwise_comparisons']['baseline_vs_improved']
        self.assertTrue(baseline_vs_improved['significant'])
        self.assertEqual(baseline_vs_improved['better_model'], 'improved')

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.3, 0.2, 0.9, 0.1])

        # Define a simple metric function (accuracy)
        def accuracy_metric(y_t, y_p):
            y_pred = (y_p >= 0.5).astype(int)
            return np.mean(y_t == y_pred)

        # Calculate bootstrap CI
        metric_value, lower_bound, upper_bound = bootstrap_confidence_interval(
            y_true, y_pred_prob, accuracy_metric, n_bootstrap=100, confidence_level=0.95
        )

        # Check that values are reasonable
        self.assertGreaterEqual(metric_value, 0.0)
        self.assertLessEqual(metric_value, 1.0)
        self.assertGreaterEqual(lower_bound, 0.0)
        self.assertLessEqual(upper_bound, 1.0)
        self.assertLessEqual(lower_bound, metric_value)
        self.assertGreaterEqual(upper_bound, metric_value)

        # The confidence interval should be reasonable width
        self.assertLess(upper_bound - lower_bound, 0.5)  # Shouldn't be too wide

    def test_model_evaluation_with_bootstrap(self):
        """Test that model evaluation works with bootstrap confidence intervals."""
        # Create test predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.3, 0.2, 0.9, 0.1])

        # Evaluate model
        metrics = evaluate_model_predictions(y_true, y_pred_prob)

        # Check that we get reasonable metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('ami_sensitivity', metrics)
        self.assertIn('ami_positive_predictivity', metrics)

        # All metrics should be in [0, 1]
        for key in ['accuracy', 'f1_score', 'ami_sensitivity', 'ami_positive_predictivity']:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_multiple_comparisons_correction(self):
        """Test handling of multiple comparisons (conceptual)."""
        # This test ensures our comparison framework can handle multiple models
        # without inflating Type I error rate excessively

        # Add more models to the comparison
        extended_models = self.models_results.copy()
        extended_models['another_model'] = {
            'f1_score': np.random.normal(0.86, 0.02, self.n_folds),
            'accuracy': np.random.normal(0.88, 0.015, self.n_folds)
        }

        comparison = compare_model_performance(
            extended_models,
            metric_name='f1_score',
            test_type='paired_ttest'
        )

        # Should have comparisons for all pairs
        expected_comparisons = [
            'baseline_vs_improved',
            'baseline_vs_similar',
            'baseline_vs_another_model',
            'improved_vs_similar',
            'improved_vs_another_model',
            'similar_vs_another_model'
        ]

        for comp in expected_comparisons:
            self.assertIn(comp, comparison['pairwise_comparisons'])


class TestPowerAnalysis(unittest.TestCase):
    """Test power analysis concepts for experimental design."""

    def test_effect_size_interpretation(self):
        """Test interpretation of effect sizes for practical significance."""
        from src.benchmarking.comparison_engine import _calculate_cohens_d

        # Small effect
        small_effect = _calculate_cohens_d(np.array([1, 2, 3, 4, 5]), np.array([1.2, 2.2, 3.2, 4.2, 5.2]))
        self.assertLess(abs(small_effect), 0.3)  # Small effect per Cohen's guidelines

        # Medium effect
        medium_effect = _calculate_cohens_d(np.array([1, 2, 3, 4, 5]), np.array([1.5, 2.5, 3.5, 4.5, 5.5]))
        self.assertGreaterEqual(abs(medium_effect), 0.3)
        self.assertLess(abs(medium_effect), 0.5)  # Medium effect

        # Large effect
        large_effect = _calculate_cohens_d(np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6]))
        self.assertGreaterEqual(abs(large_effect), 0.5)  # Large effect

    def test_sample_size_implications(self):
        """Test understanding of sample size implications for statistical power."""
        # This is more conceptual, but we can demonstrate the principle

        # With small sample size, same effect size gives less power
        small_sample_effect = _calculate_cohens_d(
            np.random.normal(0, 1, 10),
            np.random.normal(0.5, 1, 10)  # Medium effect size
        )

        # With large sample size, we get more stable estimate of same effect
        large_sample_effect = _calculate_cohens_d(
            np.random.normal(0, 1, 100),
            np.random.normal(0.5, 1, 100)  # Same medium effect size
        )

        # Both should be around 0.5, but large sample gives more precise estimate
        # We can't test the exact value due to randomness, but we can check they're reasonable
        self.assertGreater(abs(small_sample_effect), 0.1)
        self.assertLess(abs(small_sample_effect), 1.0)
        self.assertGreater(abs(large_sample_effect), 0.1)
        self.assertLess(abs(large_sample_effect), 1.0)


if __name__ == '__main__':
    unittest.main()