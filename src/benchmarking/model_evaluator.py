"""
Model evaluation utilities for Beat2Bit project.
Implements AAMI EC57 compliant metrics for arrhythmia detection.
"""

import numpy as np
from typing import Tuple, Dict, List
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import logging

logger = logging.getLogger(__name__)


def calculate_binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Dictionary containing accuracy, precision, recall, f1, specificity
    """
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate specificity (true negative rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(recall),  # Alias for clarity in medical context
        'specificity': float(specificity),
        'f1_score': float(f1),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


def calculate_ami_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate AAMI EC57 compliant metrics for arrhythmia detection.

    According to AAMI EC57:2012, performance should be reported as:
    - Sensitivity (Se) = TP / (TP + FN)
    - Positive Predictivity (+P) = TP / (TP + FP)

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Dictionary containing AAMI-compliant metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity (Se) - ability to detect abnormal beats
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Positive Predictivity (+P) - probability that a positive prediction is correct
    positive_predictivity = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Effectiveness (E) - geometric mean of Se and +P
    effectiveness = np.sqrt(sensitivity * positive_predictivity) if (sensitivity * positive_predictivity) > 0 else 0.0

    return {
        'ami_sensitivity': float(sensitivity),
        'ami_positive_predictivity': float(positive_predictivity),
        'ami_effectiveness': float(effectiveness),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def evaluate_model_predictions(y_true: np.ndarray, y_pred_prob: np.ndarray,
                              threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model predictions with probability outputs.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred_prob: Predicted probabilities for positive class
        threshold: Threshold for converting probabilities to binary predictions

    Returns:
        Dictionary containing all evaluation metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Calculate all metrics
    binary_metrics = calculate_binary_classification_metrics(y_true, y_pred)
    ami_metrics = calculate_ami_metrics(y_true, y_pred)

    # Combine metrics
    metrics = {**binary_metrics, **ami_metrics}

    # Add additional useful metrics
    metrics['threshold_used'] = float(threshold)
    metrics['total_samples'] = int(len(y_true))
    metrics['positive_samples'] = int(np.sum(y_true))
    metrics['negative_samples'] = int(len(y_true) - np.sum(y_true))

    return metrics


def calculate_optimal_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold based on Youden's J statistic (sensitivity + specificity - 1).

    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities

    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal_threshold)
    """
    from sklearn.metrics import roc_curve

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    # Youden's J statistic: sensitivity + specificity - 1
    # specificity = 1 - fpr
    j_scores = tpr + (1 - fpr) - 1

    # Find index of maximum J statistic
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate metrics at optimal threshold
    optimal_metrics = evaluate_model_predictions(y_true, y_pred_prob, optimal_threshold)

    return float(optimal_threshold), optimal_metrics


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred_prob: np.ndarray,
                                 metric_func: callable, n_bootstrap: int = 1000,
                                 confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities
        metric_func: Function that takes (y_true, y_pred_prob) and returns scalar metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (metric_value, lower_bound, upper_bound)
    """
    # Calculate metric on original data
    original_metric = metric_func(y_true, y_pred_prob)

    # Bootstrap sampling
    n_samples = len(y_true)
    bootstrap_metrics = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_prob_boot = y_pred_prob[indices]

        # Calculate metric on bootstrap sample
        boot_metric = metric_func(y_true_boot, y_pred_prob_boot)
        bootstrap_metrics.append(boot_metric)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
    upper_bound = np.percentile(bootstrap_metrics, upper_percentile)

    return float(original_metric), float(lower_bound), float(upper_bound)


if __name__ == "__main__":
    # Example usage
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.3, 0.2, 0.9, 0.1])

    metrics = evaluate_model_predictions(y_true, y_pred_prob)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value:.4f}")

    optimal_thresh, opt_metrics = calculate_optimal_threshold(y_true, y_pred_prob)
    print(f"\nOptimal threshold: {optimal_thresh:.4f}")
    print(f"Metrics at optimal threshold:")
    for key, value in opt_metrics.items():
        if key not in ['confusion_matrix', 'threshold_used']:
            print(f"  {key}: {value:.4f}")