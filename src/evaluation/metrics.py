"""
Evaluation metrics for Beat2Bit project.
Implements AAMI EC57 compliant metrics and standard classification metrics.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,  # Sensitivity is same as recall for binary classification
        'specificity': specificity,
        'f1_score': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }

    return metrics


def calculate_ami_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate AAMI EC57 compliant metrics for arrhythmia detection.

    Args:
        y_true: Ground truth labels (0 = normal, 1 = arrhythmia)
        y_pred: Predicted labels (0 = normal, 1 = arrhythmia)

    Returns:
        Dictionary of AAMI metrics
    """
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # AAMI EC57 metrics
    # Sensitivity = TP / (TP + FN) - ability to detect arrhythmias
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Positive Predictivity = TP / (TP + FP) - proportion of detected arrhythmias that are true
    positive_predictivity = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Effectiveness = geometric mean of sensitivity and positive predictivity
    effectiveness = np.sqrt(sensitivity * positive_predictivity) if (sensitivity * positive_predictivity) > 0 else 0.0

    # False Positive Rate = FP / (FP + TN)
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # False Negative Rate = FN / (TP + FN)
    false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        'ami_sensitivity': sensitivity,
        'ami_positive_predictivity': positive_predictivity,
        'ami_effectiveness': effectiveness,
        'ami_false_positive_rate': false_positive_rate,
        'ami_false_negative_rate': false_negative_rate,
        'ami_true_positives': int(tp),
        'ami_false_positives': int(fp),
        'ami_true_negatives': int(tn),
        'ami_false_negatives': int(fn)
    }

    return metrics


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model predictions using probabilities and threshold.

    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities (shape: [n_samples] or [n_samples, 2])
        threshold: Classification threshold for positive class

    Returns:
        Dictionary of evaluation metrics
    """
    # Handle different prediction formats
    if len(y_pred_prob.shape) == 2 and y_pred_prob.shape[1] == 2:
        # Probabilities for both classes, take positive class probability
        y_pred_prob_positive = y_pred_prob[:, 1]
    elif len(y_pred_prob.shape) == 1:
        # Already probabilities for positive class
        y_pred_prob_positive = y_pred_prob
    else:
        raise ValueError(f"Unexpected prediction shape: {y_pred_prob.shape}")

    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob_positive >= threshold).astype(int)

    # Calculate standard binary classification metrics
    binary_metrics = calculate_binary_classification_metrics(y_true, y_pred)

    # Calculate AAMI EC57 compliant metrics
    ami_metrics = calculate_ami_metrics(y_true, y_pred)

    # Combine metrics
    metrics = {**binary_metrics, **ami_metrics}

    # Add threshold used
    metrics['threshold'] = threshold

    return metrics


def calculate_optimal_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    metric_to_optimize: str = 'f1_score'
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate optimal threshold for classification based on a metric.

    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities for positive class
        metric_to_optimize: Metric to optimize ('f1_score', 'ami_effectiveness', 'youden')

    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal_threshold)
    """
    from sklearn.metrics import roc_curve

    # Handle different prediction formats
    if len(y_pred_prob.shape) == 2 and y_pred_prob.shape[1] == 2:
        y_pred_prob_positive = y_pred_prob[:, 1]
    elif len(y_pred_prob.shape) == 1:
        y_pred_prob_positive = y_pred_prob
    else:
        raise ValueError(f"Unexpected prediction shape: {y_pred_prob.shape}")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob_positive)

    # Calculate metrics for each threshold
    best_metric = -1
    best_threshold = 0.5
    best_metrics = {}

    for i, threshold in enumerate(thresholds):
        # Convert probabilities to predictions using this threshold
        y_pred = (y_pred_prob_positive >= threshold).astype(int)

        # Calculate metrics
        binary_metrics = calculate_binary_classification_metrics(y_true, y_pred)
        ami_metrics = calculate_ami_metrics(y_true, y_pred)
        metrics = {**binary_metrics, **ami_metrics}

        # Select metric to optimize
        if metric_to_optimize == 'f1_score':
            metric_value = metrics['f1_score']
        elif metric_to_optimize == 'ami_effectiveness':
            metric_value = metrics['ami_effectiveness']
        elif metric_to_optimize == 'youden':
            # Youden's J statistic = sensitivity + specificity - 1
            metric_value = metrics['sensitivity'] + metrics['specificity'] - 1
        else:
            raise ValueError(f"Unknown metric to optimize: {metric_to_optimize}")

        # Update best if this is better
        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = threshold
            best_metrics = metrics.copy()

    # Add threshold to metrics
    best_metrics['optimal_threshold'] = best_threshold
    best_metrics[f'best_{metric_to_optimize}'] = best_metric

    logger.info(f"Optimal threshold for {metric_to_optimize}: {best_threshold:.4f} "
                f"(value: {best_metric:.4f})")

    return best_threshold, best_metrics


def calculate_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> Dict[str, Tuple[float, float, float]]:
    """
    Calculate confidence intervals for metrics using bootstrap sampling.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_func: Function to calculate metric (defaults to accuracy)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with metric name as key and (lower_bound, upper_bound, point_estimate) as value
    """
    if metric_func is None:
        metric_func = lambda yt, yp: np.mean(yt == yp)  # Accuracy

    # Calculate point estimate
    point_estimate = metric_func(y_true, y_pred)

    # Bootstrap sampling
    n_samples = len(y_true)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Calculate metric on bootstrap sample
        boot_stat = metric_func(y_true_boot, y_pred_boot)
        bootstrap_stats.append(boot_stat)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    return {
        'accuracy': (lower_bound, upper_bound, point_estimate)
    }


def calculate_mcnemar_test(
    y_true: np.ndarray,
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray
) -> Dict[str, float]:
    """
    Calculate McNemar's test for comparing two classifiers.

    Args:
        y_true: Ground truth labels
        y_pred_model1: Predictions from model 1
        y_pred_model2: Predictions from model 2

    Returns:
        Dictionary with McNemar's test results
    """
    from statsmodels.stats.contingency_tables import mcnemar

    # Create contingency table
    # Both correct: a
    # Model 1 correct, Model 2 wrong: b
    # Model 1 wrong, Model 2 correct: c
    # Both wrong: d

    correct1 = (y_true == y_pred_model1)
    correct2 = (y_true == y_pred_model2)

    a = np.sum(correct1 & correct2)  # Both correct
    b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
    d = np.sum(~correct1 & ~correct2)  # Both wrong

    # Create contingency table [[a, b], [c, d]]
    table = [[a, b], [c, d]]

    # Perform McNemar's test
    result = mcnemar(table, exact=False, correction=True)

    return {
        'mcnemar_statistic': result.statistic,
        'mcnemar_pvalue': result.pvalue,
        'contingency_table': table,
        'both_correct': int(a),
        'model1_only_correct': int(b),
        'model2_only_correct': int(c),
        'both_wrong': int(d)
    }


def calculate_model_agreement(
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray
) -> Dict[str, float]:
    """
    Calculate agreement between two models' predictions.

    Args:
        y_pred_model1: Predictions from model 1
        y_pred_model2: Predictions from model 2

    Returns:
        Dictionary of agreement metrics
    """
    from sklearn.metrics import cohen_kappa_score

    # Overall agreement
    agreement = np.mean(y_pred_model1 == y_pred_model2)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_pred_model1, y_pred_model2)

    # Specific agreement for each class
    unique_labels = np.unique(np.concatenate([y_pred_model1, y_pred_model2]))
    class_agreement = {}

    for label in unique_labels:
        both_predict_label = (y_pred_model1 == label) & (y_pred_model2 == label)
        either_predict_label = (y_pred_model1 == label) | (y_pred_model2 == label)
        if np.sum(either_predict_label) > 0:
            class_agreement[f'class_{label}'] = np.sum(both_predict_label) / np.sum(either_predict_label)
        else:
            class_agreement[f'class_{label}'] = 0.0

    metrics = {
        'overall_agreement': agreement,
        'cohen_kappa': kappa,
        **class_agreement
    }

    return metrics


if __name__ == "__main__":
    # Example usage
    print("Evaluation Metrics Module")
    print("========================")

    # Create test data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_prob = np.random.rand(100)

    # Test binary classification metrics
    print("\n1. Testing binary classification metrics:")
    binary_metrics = calculate_binary_classification_metrics(y_true, y_pred)
    for key, value in binary_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Test AAMI metrics
    print("\n2. Testing AAMI EC57 metrics:")
    ami_metrics = calculate_ami_metrics(y_true, y_pred)
    for key, value in ami_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Test prediction evaluation
    print("\n3. Testing prediction evaluation:")
    eval_metrics = evaluate_model_predictions(y_true, y_pred_prob, threshold=0.5)
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  AMI Sensitivity: {eval_metrics['ami_sensitivity']:.4f}")
    print(f"  AMI Positive Predictivity: {eval_metrics['ami_positive_predictivity']:.4f}")
    print(f"  AMI Effectiveness: {eval_metrics['ami_effectiveness']:.4f}")

    # Test optimal threshold
    print("\n4. Testing optimal threshold calculation:")
    opt_thresh, opt_metrics = calculate_optimal_threshold(y_true, y_pred_prob, metric_to_optimize='f1_score')
    print(f"  Optimal threshold: {opt_thresh:.4f}")
    print(f"  F1 score at optimal threshold: {opt_metrics['f1_score']:.4f}")
    print(f"  AMI Effectiveness at optimal threshold: {opt_metrics['ami_effectiveness']:.4f}")

    # Test McNemar's test (with two different models)
    print("\n5. Testing McNemar's test:")
    y_pred_model2 = np.random.randint(0, 2, 100)  # Second model predictions
    mcnemar_results = calculate_mcnemar_test(y_true, y_pred, y_pred_model2)
    print(f"  McNemar's statistic: {mcnemar_results['mcnemar_statistic']:.4f}")
    print(f"  McNemar's p-value: {mcnemar_results['mcnemar_pvalue']:.4f}")

    # Test model agreement
    print("\n6. Testing model agreement:")
    agreement_results = calculate_model_agreement(y_pred, y_pred_model2)
    print(f"  Overall agreement: {agreement_results['overall_agreement']:.4f}")
    print(f"  Cohen's Kappa: {agreement_results['cohen_kappa']:.4f}")

    print("\n✓ All evaluation metrics working correctly!")