"""
Model comparison engine for Beat2Bit project.
Provides statistical comparison and visualization of model performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from scipy import stats
import json
from collections import Counter

logger = logging.getLogger(__name__)


def compare_model_performance(models_results: Dict[str, Dict[str, float]],
                            metric_name: str = 'f1_score',
                            test_type: str = 'paired_ttest') -> Dict[str, Any]:
    """
    Compare performance of multiple models using statistical tests.

    Args:
        models_results: Dictionary mapping model names to lists of metric values
                       (e.g., from cross-validation folds)
        metric_name: Name of the metric to compare
        test_type: Type of statistical test ('paired_ttest', 'wilcoxon', 'anova')

    Returns:
        Dictionary containing comparison results
    """
    model_names = list(models_results.keys())
    n_models = len(model_names)

    if n_models < 2:
        return {'error': 'Need at least 2 models for comparison'}

    # Extract metric values for each model
    metric_values = {}
    for model_name in model_names:
        if metric_name in models_results[model_name]:
            # Assuming models_results[model_name][metric_name] is a list of values
            metric_values[model_name] = np.array(models_results[model_name][metric_name])
        else:
            # If it's a single value, convert to list
            metric_values[model_name] = np.array([models_results[model_name][metric_name]])

    # Check that all models have the same number of measurements (for paired tests)
    lengths = [len(vals) for vals in metric_values.values()]
    if len(set(lengths)) > 1 and test_type in ['paired_ttest', 'wilcoxon']:
        logger.warning("Models have different numbers of measurements. Using unpaired tests.")
        test_type = 'ttest_ind' if test_type == 'paired_ttest' else 'mannwhitneyu'

    results = {
        'metric_name': metric_name,
        'model_names': model_names,
        'n_models': n_models,
        'descriptive_stats': {},
        'pairwise_comparisons': {},
        'overall_test': None,
        'recommendations': []
    }

    # Calculate descriptive statistics
    for model_name in model_names:
        vals = metric_values[model_name]
        results['descriptive_stats'][model_name] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'median': float(np.median(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'n_samples': int(len(vals)),
            'confidence_interval_95': _calculate_confidence_interval(vals, 0.95)
        }

    # Perform pairwise comparisons
    if test_type in ['paired_ttest', 'ttest_ind']:
        results['pairwise_comparisons'] = _perform_pairwise_ttests(metric_values, test_type)
    elif test_type in ['wilcoxon', 'mannwhitneyu']:
        results['pairwise_comparisons'] = _perform_pairwise_nonparametric(metric_values, test_type)

    # Overall test (ANOVA or Kruskal-Wallis)
    if n_models > 2:
        if test_type in ['paired_ttest', 'ttest_ind'] and len(set(lengths)) == 1:
            # Repeated measures ANOVA would be ideal, but using one-way ANOVA for simplicity
            f_stat, p_value = stats.f_oneway(*[metric_values[name] for name in model_names])
            results['overall_test'] = {
                'test_type': 'one_way_anova',
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        else:
            # Kruskal-Wallis test (non-parametric alternative to ANOVA)
            h_stat, p_value = stats.kruskal(*[metric_values[name] for name in model_names])
            results['overall_test'] = {
                'test_type': 'kruskal_wallis',
                'h_statistic': float(h_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

    # Generate recommendations
    results['recommendations'] = _generate_comparison_recommendations(results)

    return results


def _calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for data."""
    n = len(data)
    if n <= 1:
        return (float(np.mean(data)), float(np.mean(data)))

    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return (float(mean - h), float(mean + h))


def _perform_pairwise_ttests(metric_values: Dict[str, np.ndarray],
                           test_type: str) -> Dict[str, Any]:
    """Perform pairwise t-tests between models."""
    model_names = list(metric_values.keys())
    comparisons = {}

    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model_a = model_names[i]
            model_b = model_names[j]
            vals_a = metric_values[model_a]
            vals_b = metric_values[model_b]

            if test_type == 'paired_ttest':
                # Paired t-test assumes same order of measurements
                min_len = min(len(vals_a), len(vals_b))
                t_stat, p_value = stats.ttest_rel(vals_a[:min_len], vals_b[:min_len])
            else:  # ttest_ind
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)  # Welch's t-test

            comparison_key = f'{model_a}_vs_{model_b}'
            comparisons[comparison_key] = {
                'test_type': test_type,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'better_model': model_a if np.mean(vals_a) > np.mean(vals_b) else model_b,
                'effect_size': _calculate_cohens_d(vals_a, vals_b)
            }

    return comparisons


def _perform_pairwise_nonparametric(metric_values: Dict[str, np.ndarray],
                                  test_type: str) -> Dict[str, Any]:
    """Perform pairwise non-parametric tests between models."""
    model_names = list(metric_values.keys())
    comparisons = {}

    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model_a = model_names[i]
            model_b = model_names[j]
            vals_a = metric_values[model_a]
            vals_b = metric_values[model_b]

            if test_type == 'wilcoxon':
                # Wilcoxon signed-rank test (paired)
                min_len = min(len(vals_a), len(vals_b))
                if min_len > 0:
                    try:
                        stat, p_value = stats.wilcoxon(vals_a[:min_len], vals_b[:min_len])
                    except ValueError:
                        # Handle case where all differences are zero
                        stat, p_value = 0.0, 1.0
                else:
                    stat, p_value = 0.0, 1.0
            else:  # mannwhitneyu
                # Mann-Whitney U test (independent)
                stat, p_value = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')

            comparison_key = f'{model_a}_vs_{model_b}'
            comparisons[comparison_key] = {
                'test_type': test_type,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'better_model': model_a if np.median(vals_a) > np.median(vals_b) else model_b,
                'effect_size': _calculate_rank_biserial_correlation(vals_a, vals_b)
            }

    return comparisons


def _calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def _calculate_rank_biserial_correlation(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate rank-biserial correlation as effect size for Mann-Whitney U test."""
    # Simplified approximation
    n1, n2 = len(group1), len(group2)
    u1, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    u2 = n1 * n2 - u1
    return (u2 - u1) / (n1 * n2)


def compare_optimization_approaches(baseline_results: Dict[str, float],
                                  optimized_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare baseline model against multiple optimized variants.

    Args:
        baseline_results: Dictionary of baseline model metrics
        optimized_results: List of dictionaries, each containing optimized model metrics
                          with keys like 'name', 'sparsity', 'quantization', etc.

    Returns:
        Dictionary containing comparison analysis
    """
    analysis = {
        'baseline': baseline_results,
        'optimized_variants': optimized_results,
        'trade_analysis': {},
        'pareto_front': [],
        'recommendations': []
    }

    # Extract key metrics for comparison
    metrics_to_compare = ['accuracy', 'f1_score', 'ami_sensitivity', 'ami_positive_predictivity']
    size_metrics = ['model_size_kb', 'parameters']
    speed_metrics = ['latency_ms', 'fps']

    # For each optimized variant, calculate improvements/regressions
    for variant in optimized_results:
        variant_name = variant.get('name', 'unknown')
        trade_offs = {}

        for metric in metrics_to_compare:
            if metric in baseline_results and metric in variant:
                baseline_val = baseline_results[metric]
                variant_val = variant[metric]
                improvement = ((variant_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
                trade_offs[f'{metric}_improvement_pct'] = float(improvement)

        for metric in size_metrics:
            if metric in baseline_results and metric in variant:
                baseline_val = baseline_results[metric]
                variant_val = variant[metric]
                # For size metrics, negative improvement means size reduction
                change = ((variant_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
                trade_offs[f'{metric}_change_pct'] = float(change)

        for metric in speed_metrics:
            if metric in baseline_results and metric in variant:
                baseline_val = baseline_results[metric]
                variant_val = variant[metric]
                # For latency, negative improvement means speedup
                improvement = ((baseline_val - variant_val) / baseline_val * 100) if baseline_val != 0 else 0
                trade_offs[f'{metric}_improvement_pct'] = float(improvement)

        variant['trade_offs'] = trade_offs

    # Identify Pareto-optimal variants (those that are not dominated in accuracy-size space)
    if 'accuracy' in baseline_results and 'model_size_kb' in baseline_results:
        pareto_candidates = []

        for variant in optimized_results:
            if 'accuracy' in variant and 'model_size_kb' in variant:
                pareto_candidates.append({
                    'name': variant.get('name', 'unknown'),
                    'accuracy': variant['accuracy'],
                    'model_size_kb': variant['model_size_kb'],
                    'variant_data': variant
                })

        # Add baseline to candidates
        pareto_candidates.append({
            'name': 'baseline',
            'accuracy': baseline_results['accuracy'],
            'model_size_kb': baseline_results['model_size_kb'],
            'variant_data': baseline_results
        })

        # Find Pareto front (maximize accuracy, minimize size)
        pareto_front = []
        for candidate in pareto_candidates:
            is_dominated = False
            for other in pareto_candidates:
                # Check if other dominates candidate (better or equal accuracy AND better or equal size)
                if (other['accuracy'] >= candidate['accuracy'] and
                    other['model_size_kb'] <= candidate['model_size_kb'] and
                    (other['accuracy'] > candidate['accuracy'] or
                     other['model_size_kb'] < candidate['model_size_kb'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(candidate)

        # Sort Pareto front by accuracy (descending)
        pareto_front.sort(key=lambda x: x['accuracy'], reverse=True)
        analysis['pareto_front'] = [item['name'] for item in pareto_front]

    # Generate recommendations
    analysis['recommendations'] = _generate_optimization_recommendations(baseline_results, optimized_results)

    return analysis


def _generate_optimization_recommendations(baseline_results: Dict[str, float],
                                         optimized_results: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on optimization comparison."""
    recommendations = []

    if not optimized_results:
        return ["No optimized variants provided for comparison"]

    # Find best accuracy variant
    accuracy_variants = [v for v in optimized_results if 'accuracy' in v]
    if accuracy_variants:
        best_accuracy = max(accuracy_variants, key=lambda x: x['accuracy'])
        acc_improvement = ((best_accuracy['accuracy'] - baseline_results['accuracy']) /
                          baseline_results['accuracy'] * 100) if baseline_results['accuracy'] != 0 else 0
        if acc_improvement > 0:
            recommendations.append(
                f"Best accuracy: {best_accuracy.get('name', 'unknown')} "
                f"({best_accuracy['accuracy']:.4f}, +{acc_improvement:.2f}%)"
            )
        elif acc_improvement < -1:  # Only mention if significantly worse
            recommendations.append(
                f"All variants show accuracy degradation. Best: "
                f"{best_accuracy.get('name', 'unknown')} ({best_accuracy['accuracy']:.4f}, {acc_improvement:.2f}%)"
            )

    # Find smallest model variant
    size_variants = [v for v in optimized_results if 'model_size_kb' in v]
    if size_variants:
        smallest = min(size_variants, key=lambda x: x['model_size_kb'])
        size_reduction = ((baseline_results['model_size_kb'] - smallest['model_size_kb']) /
                         baseline_results['model_size_kb'] * 100) if baseline_results['model_size_kb'] != 0 else 0
        if size_reduction > 0:
            recommendations.append(
                f"Smallest model: {smallest.get('name', 'unknown')} "
                f"({smallest['model_size_kb']:.2f} KB, -{size_reduction:.2f}%)"
            )

    # Find best trade-off (accuracy vs size)
    if accuracy_variants and size_variants:
        # Simple score: accuracy improvement - size penalty
        best_tradeoff = None
        best_score = -float('inf')

        for variant in optimized_results:
            if 'accuracy' in variant and 'model_size_kb' in variant:
                acc_imp = ((variant['accuracy'] - baseline_results['accuracy']) /
                          baseline_results['accuracy'] * 100) if baseline_results['accuracy'] != 0 else 0
                size_change = ((variant['model_size_kb'] - baseline_results['model_size_kb']) /
                              baseline_results['model_size_kb'] * 100) if baseline_results['model_size_kb'] != 0 else 0
                # Score: favor accuracy improvement, penalize size increase
                score = acc_imp - (size_change * 0.5)  # Weight size less than accuracy
                if score > best_score:
                    best_score = score
                    best_tradeoff = variant

        if best_tradeoff:
            recommendations.append(
                f"Best accuracy-size trade-off: {best_tradeoff.get('name', 'unknown')} "
                f"(Acc: {best_tradeoff['accuracy']:.4f}, Size: {best_tradeoff['model_size_kb']:.2f} KB)"
            )

    # Latency recommendations
    latency_variants = [v for v in optimized_results if 'latency_ms' in v]
    if latency_variants:
        fastest = min(latency_variants, key=lambda x: x['latency_ms'])
        speedup = ((baseline_results['latency_ms'] - fastest['latency_ms']) /
                  baseline_results['latency_ms'] * 100) if baseline_results['latency_ms'] != 0 else 0
        if speedup > 0:
            recommendations.append(
                f"Fastest inference: {fastest.get('name', 'unknown')} "
                f"({fastest['latency_ms']:.2f} ms, {speedup:.2f}% speedup)"
            )

    if not recommendations:
        recommendations.append("No clear recommendations - variants show mixed trade-offs")

    return recommendations


def generate_latex_table(model_comparison: Dict[str, Any],
                        metrics_to_show: List[str] = None) -> str:
    """
    Generate LaTeX table for model comparison results.

    Args:
        model_comparison: Output from compare_model_performance function
        metrics_to_show: List of metrics to include in table

    Returns:
        LaTeX table string
    """
    if metrics_to_show is None:
        metrics_to_show = ['mean', 'std', 'median']

    model_names = model_comparison['model_names']
    descriptive_stats = model_comparison['descriptive_stats']

    # Start table
    latex = "\\begin{table}[htbp]\n\\centering\n"
    latex += "\\caption{Model Performance Comparison}\n"
    latex += "\\label{tab:model_comparison}\n"
    latex += "\\begin{tabular}{l" + "c" * len(metrics_to_show) + "}\n"
    latex += "\\hline\n"

    # Header row
    header = "Model"
    for metric in metrics_to_show:
        header += " & " + metric.upper()
    header += " \\\\ \\hline\n"
    latex += header

    # Data rows
    for model_name in model_names:
        row = model_name
        for metric in metrics_to_show:
            if metric in descriptive_stats[model_name]:
                value = descriptive_stats[model_name][metric]
                if isinstance(value, float):
                    row += f" & {value:.4f}"
                else:
                    row += f" & {value}"
            else:
                row += " & N/A"
        row += " \\\\ \n"
        latex += row

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"

    return latex


def generate_markdown_report(comparison_results: Dict[str, Any]) -> str:
    """
    Generate a markdown report from comparison results.

    Args:
        comparison_results: Dictionary containing comparison analysis

    Returns:
        Markdown formatted report string
    """
    report = []

    report.append("# Model Comparison Report\n")

    # Basic info
    if 'metric_name' in comparison_results:
        report.append(f"**Metric Compared:** {comparison_results['metric_name']}\n")
    if 'model_names' in comparison_results:
        report.append(f"**Models Compared:** {', '.join(comparison_results['model_names'])}\n")
    report.append("\n")

    # Descriptive statistics
    report.append("## Descriptive Statistics\n")
    if 'descriptive_stats' in comparison_results:
        report.append("| Model | Mean | Std | Median | 95% CI |")
        report.append("|-------|------|-----|--------|--------|")
        for model_name, stats in comparison_results['descriptive_stats'].items():
            ci = stats.get('confidence_interval_95', (0, 0))
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            report.append(
                f"| {model_name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                f"{stats['median']:.4f} | {ci_str} |"
            )
    report.append("\n")

    # Overall test
    if comparison_results.get('overall_test'):
        ot = comparison_results['overall_test']
        report.append("## Overall Significance Test\n")
        report.append(f"- **Test Type:** {ot['test_type']}\n")
        report.append(f"- **Statistic:** {ot.get('f_statistic', ot.get('h_statistic', 'N/A')):.4f}\n")
        report.append(f"- **P-value:** {ot['p_value']:.4f}\n")
        report.append(f"- **Significant (α=0.05):** {ot['significant']}\n")
        report.append("\n")

    # Pairwise comparisons
    if comparison_results.get('pairwise_comparisons'):
        report.append("## Pairwise Comparisons\n")
        report.append("| Comparison | Test | Statistic | P-value | Significant | Better Model | Effect Size |")
        report.append("|------------|------|-----------|---------|-------------|--------------|-------------|")
        for comp_key, comp in comparison_results['pairwise_comparisons'].items():
            stat_key = 't_statistic' if 't_statistic' in comp else 'statistic'
            stat_val = comp.get(stat_key, 0.0)
            report.append(
                f"| {comp_key} | {comp['test_type']} | {stat_val:.4f} | "
                f"{comp['p_value']:.4f} | {comp['significant']} | "
                f"{comp['better_model']} | {comp.get('effect_size', 0.0):.4f} |"
            )
    report.append("\n")

    # Recommendations
    if comparison_results.get('recommendations'):
        report.append("## Recommendations\n")
        for rec in comparison_results['recommendations']:
            report.append(f"- {rec}")
        report.append("\n")

    return "\n".join(report)


def _generate_comparison_recommendations(results: Dict[str, Any]) -> List[str]:
    """
    Generate recommendations based on model comparison results.
    """
    recommendations = []

    # Check if there is a significant difference between models
    if results.get('overall_test'):
        ot = results['overall_test']
        if ot.get('significant', False):
            recommendations.append("There is a statistically significant difference between models.")
            # Find the best performing model based on mean performance
            if results.get('descriptive_stats'):
                best_model = max(results['descriptive_stats'].items(),
                               key=lambda x: x[1]['mean'])[0]
                recommendations.append(f"Best performing model: {best_model}")
        else:
            recommendations.append("No statistically significant difference detected between models.")

    # Check pairwise comparisons for specific insights
    if results.get('pairwise_comparisons'):
        significant_comparisons = []
        for comp_key, comp in results['pairwise_comparisons'].items():
            if comp.get('significant', False):
                significant_comparisons.append(comp['better_model'])

        if significant_comparisons:
            # Count how many times each model was identified as better
            from collections import Counter
            better_counts = Counter(significant_comparisons)
            if better_counts:
                best_model = better_counts.most_common(1)[0][0]
                recommendations.append(f"Model '{best_model}' performed significantly better in most comparisons.")
        else:
            recommendations.append("No significant differences found in pairwise comparisons.")

    if not recommendations:
        recommendations.append("Consider collecting more data or increasing statistical power to detect differences.")

    return recommendations


if __name__ == "__main__":
    # Example usage
    # Simulate some model results
    np.random.seed(42)
    models_results = {
        'baseline': {
            'f1_score': np.random.normal(0.85, 0.03, 10),
            'accuracy': np.random.normal(0.87, 0.02, 10)
        },
        'pruned_50': {
            'f1_score': np.random.normal(0.83, 0.025, 10),
            'accuracy': np.random.normal(0.85, 0.02, 10)
        },
        'quantized_int8': {
            'f1_score': np.random.normal(0.84, 0.02, 10),
            'accuracy': np.random.normal(0.86, 0.015, 10)
        }
    }

    # Compare models
    comparison = compare_model_performance(models_results, 'f1_score', 'paired_ttest')
    print("Model Comparison Results:")
    print(json.dumps(comparison, indent=2))

    # Generate markdown report
    markdown_report = generate_markdown_report(comparison)
    print("\nMarkdown Report:")
    print(markdown_report)