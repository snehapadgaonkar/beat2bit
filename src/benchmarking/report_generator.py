"""
Report generation utilities for Beat2Bit benchmarking suite.
Creates comprehensive reports combining model evaluation, complexity analysis, and benchmarking results.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_comprehensive_report(model_evaluation: Dict[str, float],
                                complexity_analysis: Dict[str, Any],
                                latency_benchmarking: Dict[str, Any],
                                model_name: str = "model",
                                dataset_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive report combining all benchmarking aspects.

    Args:
        model_evaluation: Output from model_evaluator.py
        complexity_analysis: Output from complexity_analyzer.py
        latency_benchmarking: Output from latency_benchmarker.py
        model_name: Name of the model being evaluated
        dataset_info: Information about the dataset used

    Returns:
        Dictionary containing the comprehensive report
    """
    report = {
        'metadata': {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'report_version': '1.0.0',
            'generator': 'Beat2Bit Benchmarking Suite'
        },
        'dataset_info': dataset_info or {},
        'model_evaluation': model_evaluation,
        'complexity_analysis': complexity_analysis,
        'latency_benchmarking': latency_benchmarking,
        'summary': {},
        'recommendations': []
    }

    # Generate summary section
    report['summary'] = _generate_summary(model_evaluation, complexity_analysis, latency_benchmarking)

    # Generate recommendations
    report['recommendations'] = _generate_recommendations(model_evaluation, complexity_analysis, latency_benchmarking)

    return report


def _generate_summary(model_evaluation: Dict[str, float],
                     complexity_analysis: Dict[str, Any],
                     latency_benchmarking: Dict[str, Any]) -> Dict[str, Any]:
    """Generate executive summary of the benchmarking results."""
    summary = {}

    # Model performance summary
    if model_evaluation:
        summary['model_performance'] = {
            'accuracy': model_evaluation.get('accuracy', 0.0),
            'f1_score': model_evaluation.get('f1_score', 0.0),
            'ami_sensitivity': model_evaluation.get('ami_sensitivity', 0.0),
            'ami_positive_predictivity': model_evaluation.get('ami_positive_predictivity', 0.0),
            'ami_effectiveness': model_evaluation.get('ami_effectiveness', 0.0)
        }

    # Model complexity summary
    if complexity_analysis:
        summary['model_complexity'] = {
            'total_parameters': complexity_analysis.get('parameters', {}).get('total_parameters', 0),
            'model_size_fp32_mb': complexity_analysis.get('memory_size', {}).get('fp32_mb', 0.0),
            'model_size_int8_mb': complexity_analysis.get('memory_size', {}).get('int8_mb', 0.0),
            'compression_ratio': complexity_analysis.get('memory_size', {}).get('compression_ratio_fp32_to_int8', 0.0),
            'total_flops': complexity_analysis.get('computational_complexity', {}).get('total_flops', 0),
            'gflops': complexity_analysis.get('computational_complexity', {}).get('gflops', 0.0)
        }

    # Latency summary
    if latency_benchmarking:
        # Extract key latency metrics
        latency_stats = latency_benchmarking.get('latency_stats', {})
        throughput_stats = latency_benchmarking.get('throughput_stats', {})
        summary_info = latency_benchmarking.get('summary', {})

        summary['latency_performance'] = {
            'single_sample_latency_ms': latency_stats.get('batch_size_1', {}).get('mean_latency_ms', 0.0),
            'latency_p95_ms': latency_stats.get('batch_size_1', {}).get('p95_latency_ms', 0.0),
            'throughput_samples_per_sec': throughput_stats.get('batch_size_1', {}).get('throughput_samples_per_sec', 0.0),
            'optimal_batch_size_latency': summary_info.get('optimal_batch_size_for_latency', 1),
            'optimal_batch_size_throughput': summary_info.get('optimal_batch_size_for_throughput', 1)
        }

    return summary


def _generate_recommendations(model_evaluation: Dict[str, float],
                            complexity_analysis: Dict[str, Any],
                            latency_benchmarking: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on benchmarking results."""
    recommendations = []

    # Performance-based recommendations
    accuracy = model_evaluation.get('accuracy', 0.0)
    f1_score = model_evaluation.get('f1_score', 0.0)
    ami_sensitivity = model_evaluation.get('ami_sensitivity', 0.0)
    ami_ppv = model_evaluation.get('ami_positive_predictivity', 0.0)

    if accuracy < 0.80:
        recommendations.append(
            f"Model accuracy ({accuracy:.2%}) is below the 80% threshold. "
            f"Consider adjusting model architecture or training hyperparameters."
        )
    elif accuracy > 0.95:
        recommendations.append(
            f"Model accuracy ({accuracy:.2%}) is excellent. "
            f"Focus on optimization for deployment rather than accuracy improvements."
        )

    if f1_score < 0.75:
        recommendations.append(
            f"F1-score ({f1_score:.2%}) indicates room for improvement in balancing precision and recall."
        )

    if ami_sensitivity < 0.80:
        recommendations.append(
            f"AMI sensitivity ({ami_sensitivity:.2%}) is below recommended 80% for arrhythmia detection. "
            f"Consider adjusting classification threshold or model architecture."
        )

    if ami_ppv < 0.70:
        recommendations.append(
            f"Positive predictivity ({ami_ppv:.2%}) suggests high false positive rate. "
            f"This may lead to unnecessary alerts in clinical settings."
        )

    # Complexity-based recommendations
    params = complexity_analysis.get('parameters', {}).get('total_parameters', 0)
    model_size_mb = complexity_analysis.get('memory_size', {}).get('fp32_mb', 0.0)

    if params > 100000:  # More than 100K parameters
        recommendations.append(
            f"Model has {params:,} parameters, which may be large for edge deployment. "
            f"Consider pruning or architecture simplification."
        )

    if model_size_mb > 1.0:  # Larger than 1MB FP32
        recommendations.append(
            f"FP32 model size is {model_size_mb:.2f} MB. "
            f"INT8 quantization could reduce this to approximately "
            f"{complexity_analysis.get('memory_size', {}).get('int8_mb', 0.0):.2f} MB."
        )

    # Latency-based recommendations
    single_latency = latency_benchmarking.get('latency_stats', {}).get('batch_size_1', {}).get('mean_latency_ms', 0.0)
    throughput = latency_benchmarking.get('throughput_stats', {}).get('batch_size_1', {}).get('throughput_samples_per_sec', 0.0)

    if single_latency > 20.0:  # Slower than 50 FPS
        recommendations.append(
            f"Single-sample latency ({single_latency:.2f} ms) may be too slow for real-time applications. "
            f"Consider model optimization or hardware acceleration."
        )
    elif single_latency < 5.0:  # Faster than 200 FPS
        recommendations.append(
            f"Single-sample latency ({single_latency:.2f} ms) is excellent for real-time processing. "
            f"You have headroom for additional features or more complex models."
        )

    if throughput < 50.0:  # Less than 50 samples/sec
        recommendations.append(
            f"Throughput ({throughput:.1f} samples/sec) may limit batch processing applications. "
            f"Consider optimizing for higher throughput if batch processing is needed."
        )

    # Combined recommendations
    if recommendations:
        # Prioritize recommendations
        priority_recs = []
        other_recs = []

        for rec in recommendations:
            if any(keyword in rec.lower() for keyword in ['accuracy', 'sensitivity', 'latency']):
                priority_recs.append(rec)
            else:
                other_recs.append(rec)

        # Return priority recommendations first, then others
        return priority_recs + other_recs[:3]  # Limit total recommendations
    else:
        recommendations.append(
            "Model performance appears balanced across accuracy, complexity, and latency metrics. "
            "Consider conducting ablation studies to understand contribution of each component."
        )

    return recommendations


def save_report_to_file(report: Dict[str, Any],
                       output_dir: str,
                       filename: str = None) -> str:
    """
    Save report to JSON and Markdown files.

    Args:
        report: Report dictionary to save
        output_dir: Directory to save the report
        filename: Base filename (without extension)

    Returns:
        Path to the saved JSON file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = report.get('metadata', {}).get('model_name', 'model')
        filename = f"{model_name}_benchmark_report_{timestamp}"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON report
    json_path = os.path.join(output_dir, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"JSON report saved to {json_path}")

    # Save Markdown report
    md_path = os.path.join(output_dir, f"{filename}.md")
    markdown_content = generate_markdown_report(report)
    with open(md_path, 'w') as f:
        f.write(markdown_content)
    logger.info(f"Markdown report saved to {md_path}")

    return json_path


def generate_comparison_report(models_data: Dict[str, Dict[str, Any]],
                             output_filename: str = "model_comparison_report") -> str:
    """
    Generate a comparison report for multiple models.

    Args:
        models_data: Dictionary mapping model names to their data
                    Each model's data should contain:
                    - 'metrics': model evaluation results
                    - 'complexity': complexity analysis results
                    - 'latency': latency benchmarking results
        output_filename: Base filename for the report (without extension)

    Returns:
        Path to the generated report file
    """
    # Generate comprehensive reports for each model
    individual_reports = {}
    for model_name, data in models_data.items():
        individual_reports[model_name] = generate_comprehensive_report(
            model_evaluation=data.get('metrics', {}),
            complexity_analysis=data.get('complexity', {}),
            latency_benchmarking=data.get('latency', {}),
            model_name=model_name
        )

    # Create a comparison report
    comparison_report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'report_version': '1.0.0',
            'generator': 'Beat2Bit Benchmarking Suite',
            'report_type': 'comparison'
        },
        'models': individual_reports,
        'comparison_summary': _generate_comparison_summary(individual_reports)
    }

    # Save the comparison report
    output_dir = "./comparison_reports"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"{output_filename}_{timestamp}.json")

    with open(json_path, 'w') as f:
        json.dump(comparison_report, f, indent=2, default=str)

    # Also generate markdown version
    md_path = os.path.join(output_dir, f"{output_filename}_{timestamp}.md")
    markdown_content = generate_markdown_report(comparison_report)
    with open(md_path, 'w') as f:
        f.write(markdown_content)

    logger.info(f"Comparison report saved to {json_path} and {md_path}")
    return json_path


def _generate_comparison_summary(models_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary comparison across multiple model reports.
    """
    summary = {
        'model_count': len(models_reports),
        'model_names': list(models_reports.keys()),
        'performance_comparison': {},
        'complexity_comparison': {},
        'latency_comparison': {}
    }

    # Compare performance metrics
    perf_metrics = ['accuracy', 'f1_score', 'ami_sensitivity', 'ami_positive_predictivity']
    for metric in perf_metrics:
        summary['performance_comparison'][metric] = {}
        for model_name, report in models_reports.items():
            value = report.get('model_evaluation', {}).get(metric, 0.0)
            summary['performance_comparison'][metric][model_name] = value

    # Compare complexity metrics
    summary['complexity_comparison']['parameters'] = {}
    summary['complexity_comparison']['model_size_fp32_mb'] = {}
    summary['complexity_comparison']['model_size_int8_mb'] = {}
    for model_name, report in models_reports.items():
        complexity = report.get('complexity_analysis', {})
        summary['complexity_comparison']['parameters'][model_name] = complexity.get('parameters', {}).get('total_parameters', 0)
        summary['complexity_comparison']['model_size_fp32_mb'][model_name] = complexity.get('memory_size', {}).get('fp32_mb', 0.0)
        summary['complexity_comparison']['model_size_int8_mb'][model_name] = complexity.get('memory_size', {}).get('int8_mb', 0.0)

    # Compare latency metrics
    for model_name, report in models_reports.items():
        latency = report.get('latency_benchmarking', {})
        single_sample_lat = latency.get('latency_stats', {}).get('batch_size_1', {}).get('mean_latency_ms', 0.0)
        if 'single_sample_latency_ms' not in summary['latency_comparison']:
            summary['latency_comparison']['single_sample_latency_ms'] = {}
        summary['latency_comparison']['single_sample_latency_ms'][model_name] = single_sample_lat

    return summary


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """
    Generate a human-readable Markdown report from the comprehensive report.

    Args:
        report: Comprehensive report dictionary

    Returns:
        Markdown formatted report string
    """
    md_lines = []

    # Header
    model_name = report.get('metadata', {}).get('model_name', 'Unknown Model')
    timestamp = report.get('metadata', {}).get('timestamp', 'Unknown')
    md_lines.append(f"# Benchmark Report: {model_name}")
    md_lines.append(f"**Generated:** {timestamp}")
    md_lines.append("")

    # Dataset Info
    dataset_info = report.get('dataset_info', {})
    if dataset_info:
        md_lines.append("## Dataset Information")
        for key, value in dataset_info.items():
            md_lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        md_lines.append("")

    # Model Performance
    model_eval = report.get('model_evaluation', {})
    if model_eval:
        md_lines.append("## Model Performance Metrics")
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")

        # Key metrics to display
        key_metrics = [
            ('accuracy', 'Accuracy'),
            ('precision', 'Precision'),
            ('recall', 'Recall (Sensitivity)'),
            ('f1_score', 'F1-Score'),
            ('ami_sensitivity', 'AMI Sensitivity'),
            ('ami_positive_predictivity', 'AMI +P'),
            ('ami_effectiveness', 'AMI Effectiveness'),
            ('specificity', 'Specificity')
        ]

        for key, label in key_metrics:
            value = model_eval.get(key)
            if value is not None:
                if isinstance(value, float):
                    md_lines.append(f"| {label} | {value:.4f} |")
                else:
                    md_lines.append(f"| {label} | {value} |")
        md_lines.append("")

        # Confusion Matrix
        conf_matrix = model_eval.get('confusion_matrix', {})
        if conf_matrix:
            md_lines.append("### Confusion Matrix")
            md_lines.append("|          | Predicted Negative | Predicted Positive |")
            md_lines.append("|----------|-------------------|-------------------|")
            md_lines.append(f"| Actual Negative | {conf_matrix.get('tn', 0)} | {conf_matrix.get('fp', 0)} |")
            md_lines.append(f"| Actual Positive | {conf_matrix.get('fn', 0)} | {conf_matrix.get('tp', 0)} |")
            md_lines.append("")

    # Model Complexity
    complexity = report.get('complexity_analysis', {})
    if complexity:
        md_lines.append("## Model Complexity Analysis")

        params = complexity.get('parameters', {}).get('total_parameters', 0)
        md_lines.append(f"- **Total Parameters:** {params:,}")

        mem_size = complexity.get('memory_size', {})
        fp32_size = mem_size.get('fp32_mb', 0.0)
        int8_size = mem_size.get('int8_mb', 0.0)
        compression = mem_size.get('compression_ratio_fp32_to_int8', 0.0)
        md_lines.append(f"- **Model Size (FP32):** {fp32_size:.2f} MB")
        md_lines.append(f"- **Model Size (INT8):** {int8_size:.2f} MB")
        md_lines.append(f"- **Compression Ratio:** {compression:.2f}x")

        flops = complexity.get('computational_complexity', {})
        total_flops = flops.get('total_flops', 0)
        gflops = flops.get('gflops', 0.0)
        md_lines.append(f"- **Total FLOPs:** {total_flops:,}")
        md_lines.append(f"- **GFLOPs:** {gflops:.3f}")
        md_lines.append("")

    # Latency Performance
    latency = report.get('latency_benchmarking', {})
    if latency:
        md_lines.append("## Latency & Throughput Performance")

        # Single sample latency (most important for edge devices)
        single_lat = latency.get('latency_stats', {}).get('batch_size_1', {})
        if single_lat:
            md_lines.append("### Single-Sample Latency (Batch Size = 1)")
            md_lines.append("| Metric | Value |")
            md_lines.append("|--------|-------|")
            md_lines.append(f"| Mean Latency | {single_lat.get('mean_latency_ms', 0.0):.2f} ms |")
            md_lines.append(f"| Median Latency | {single_lat.get('median_latency_ms', 0.0):.2f} ms |")
            md_lines.append(f"| P95 Latency | {single_lat.get('p95_latency_ms', 0.0):.2f} ms |")
            md_lines.append(f"| P99 Latency | {single_lat.get('p99_latency_ms', 0.0):.2f} ms |")
            md_lines.append(f"| Min Latency | {single_lat.get('min_latency_ms', 0.0):.2f} ms |")
            md_lines.append(f"| Max Latency | {single_lat.get('max_latency_ms', 0.0):.2f} ms |")
            md_lines.append("")

        # Throughput
        throughput = latency.get('throughput_stats', {}).get('batch_size_1', {})
        if throughput:
            md_lines.append("### Throughput (Batch Size = 1)")
            md_lines.append(f"- **Samples/Second:** {throughput.get('throughput_samples_per_sec', 0.0):.1f}")
            md_lines.append(f"- **Latency/Sample:** {throughput.get('latency_per_sample_ms', 0.0):.2f} ms")
            md_lines.append("")

        # Summary info
        summary = latency.get('summary', {})
        if summary:
            md_lines.append("### Latency Optimization")
            md_lines.append(f"- **Optimal Batch Size (Lowest Latency):** {summary.get('optimal_batch_size_for_latency', 'N/A')}")
            md_lines.append(f"- **Optimal Batch Size (Highest Throughput):** {summary.get('optimal_batch_size_for_throughput', 'N/A')}")
            md_lines.append("")

    # Summary Section
    summary_section = report.get('summary', {})
    if summary_section:
        md_lines.append("## Executive Summary")

        if 'model_performance' in summary_section:
            perf = summary_section['model_performance']
            md_lines.append("### Model Performance")
            md_lines.append(f"- **Accuracy:** {perf.get('accuracy', 0.0):.2%}")
            md_lines.append(f"- **F1-Score:** {perf.get('f1_score', 0.0):.2%}")
            md_lines.append(f"- **AMI Sensitivity:** {perf.get('ami_sensitivity', 0.0):.2%}")
            md_lines.append(f"- **AMI +P:** {perf.get('ami_positive_predictivity', 0.0):.2%}")
            md_lines.append(f"- **AMI Effectiveness:** {perf.get('ami_effectiveness', 0.0):.2%}")
            md_lines.append("")

        if 'model_complexity' in summary_section:
            comp = summary_section['model_complexity']
            md_lines.append("### Model Complexity")
            md_lines.append(f"- **Parameters:** {comp.get('total_parameters', 0):,}")
            md_lines.append(f"- **FP32 Size:** {comp.get('model_size_fp32_mb', 0.0):.2f} MB")
            md_lines.append(f"- **INT8 Size:** {comp.get('model_size_int8_mb', 0.0):.2f} MB")
            md_lines.append(f"- **Compression:** {comp.get('compression_ratio', 0.0):.2f}x")
            md_lines.append(f"- **FLOPs:** {comp.get('total_flops', 0):,}")
            md_lines.append("")

        if 'latency_performance' in summary_section:
            lat = summary_section['latency_performance']
            md_lines.append("### Latency Performance")
            md_lines.append(f"- **Single-Sample Latency:** {lat.get('single_sample_latency_ms', 0.0):.2f} ms")
            md_lines.append(f"- **P95 Latency:** {lat.get('latency_p95_ms', 0.0):.2f} ms")
            md_lines.append(f"- **Throughput:** {lat.get('throughput_samples_per_sec', 0.0):.1f} samples/sec")
            md_lines.append("")

    # Recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        md_lines.append("## Recommendations")
        for i, rec in enumerate(recommendations, 1):
            md_lines.append(f"{i}. {rec}")
        md_lines.append("")

    # Footer
    md_lines.append("---")
    md_lines.append(f"*Report generated by Beat2Bit Benchmarking Suite v{report.get('metadata', {}).get('report_version', '1.0.0')}*")

    return "\n".join(md_lines)


def load_report_from_file(json_path: str) -> Dict[str, Any]:
    """
    Load a report from JSON file.

    Args:
        json_path: Path to the JSON report file

    Returns:
        Report dictionary
    """
    with open(json_path, 'r') as f:
        report = json.load(f)
    return report


def compare_reports(report_paths: List[str]) -> Dict[str, Any]:
    """
    Compare multiple benchmark reports.

    Args:
        report_paths: List of paths to JSON report files

    Returns:
        Dictionary containing comparison analysis
    """
    reports = []
    for path in report_paths:
        try:
            report = load_report_from_file(path)
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed to load report from {path}: {e}")

    if len(reports) < 2:
        return {'error': 'Need at least 2 reports for comparison'}

    # Extract model names
    model_names = []
    for report in reports:
        name = report.get('metadata', {}).get('model_name', f'Model_{len(model_names)+1}')
        model_names.append(name)

    # Prepare data for comparison engine
    model_evaluations = {}
    model_complexities = {}
    model_latencies = {}

    for i, report in enumerate(reports):
        name = model_names[i]
        model_evaluations[name] = report.get('model_evaluation', {})
        model_complexities[name] = report.get('complexity_analysis', {})
        model_latencies[name] = report.get('latency_benchmarking', {})

    # Create comparison using existing engines
    # Note: This would need integration with comparison_engine.py
    # For now, return basic comparison structure

    comparison = {
        'models_compared': model_names,
        'n_models': len(model_names),
        'individual_reports': {
            name: {
                'evaluation': model_evaluations[name],
                'complexity': model_complexities[name],
                'latency': model_latencies[name]
            } for name in model_names
        },
        'comparison_timestamp': datetime.now().isoformat()
    }

    # Add summary comparison
    comparison['summary_comparison'] = _compare_summaries(
        [report.get('summary', {}) for report in reports],
        model_names
    )

    return comparison


def _compare_summaries(summaries: List[Dict[str, Any]], model_names: List[str]) -> Dict[str, Any]:
    """Compare summary sections across multiple reports."""
    comparison = {}

    # Compare model performance
    perf_metrics = ['accuracy', 'f1_score', 'ami_sensitivity', 'ami_positive_predictivity', 'ami_effectiveness']
    perf_comparison = {}
    for metric in perf_metrics:
        perf_comparison[metric] = {}
        for i, name in enumerate(model_names):
            value = summaries[i].get('model_performance', {}).get(metric, 0.0) if i < len(summaries) else 0.0
            perf_comparison[metric][name] = value
    comparison['model_performance'] = perf_comparison

    # Compare model complexity
    comp_metrics = ['total_parameters', 'model_size_fp32_mb', 'model_size_int8_mb', 'compression_ratio', 'total_flops']
    comp_comparison = {}
    for metric in comp_metrics:
        comp_comparison[metric] = {}
        for i, name in enumerate(model_names):
            value = summaries[i].get('model_complexity', {}).get(metric, 0.0) if i < len(summaries) else 0.0
            comp_comparison[metric][name] = value
    comparison['model_complexity'] = comp_comparison

    # Compare latency performance
    lat_metrics = ['single_sample_latency_ms', 'latency_p95_ms', 'throughput_samples_per_sec']
    lat_comparison = {}
    for metric in lat_metrics:
        lat_comparison[metric] = {}
        for i, name in enumerate(model_names):
            value = summaries[i].get('latency_performance', {}).get(metric, 0.0) if i < len(summaries) else 0.0
            lat_comparison[metric][name] = value
    comparison['latency_performance'] = lat_comparison

    return comparison


if __name__ == "__main__":
    # Example usage
    # Create dummy data for demonstration
    model_evaluation = {
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.89,
        'f1_score': 0.87,
        'ami_sensitivity': 0.89,
        'ami_positive_predictivity': 0.84,
        'ami_effectiveness': 0.86,
        'specificity': 0.86,
        'confusion_matrix': {'tn': 860, 'fp': 140, 'fn': 110, 'tp': 890}
    }

    complexity_analysis = {
        'parameters': {
            'total_parameters': 12450,
            'trainable_parameters': 12450,
            'non_trainable_parameters': 0
        },
        'memory_size': {
            'fp32_mb': 0.047,
            'int8_mb': 0.012,
            'compression_ratio_fp32_to_int8': 3.92
        },
        'computational_complexity': {
            'total_flops': 2840000,
            'gflops': 0.00284,
            'mflops': 2.84
        }
    }

    latency_benchmarking = {
        'batch_sizes': [1, 4, 8, 16],
        'latency_stats': {
            'batch_size_1': {
                'mean_latency_ms': 2.45,
                'median_latency_ms': 2.38,
                'std_latency_ms': 0.32,
                'min_latency_ms': 2.15,
                'max_latency_ms': 3.45,
                'p95_latency_ms': 2.98,
                'p99_latency_ms': 3.25,
                'n_measurements': 100
            }
        },
        'throughput_stats': {
            'batch_size_1': {
                'throughput_samples_per_sec': 408.2,
                'latency_per_sample_ms': 2.45
            }
        },
        'summary': {
            'optimal_batch_size_for_latency': 1,
            'optimal_batch_size_for_throughput': 16,
            'latency_range_ms': {'min': 2.45, 'max': 8.20},
            'throughput_range_samples_per_sec': {'min': 408.2, 'max': 1280.5}
        }
    }

    # Generate comprehensive report
    report = generate_comprehensive_report(
        model_evaluation=model_evaluation,
        complexity_analysis=complexity_analysis,
        latency_benchmarking=latency_benchmarking,
        model_name="baseline_cnn",
        dataset_info={
            'dataset': 'MIT-BIH Arrhythmia Database',
            'samples': 2000,
            'features': 180,
            'classes': ['Normal', 'Abnormal'],
            'split': 'Patient-independent AAMI'
        }
    )

    # Save report
    output_dir = "./benchmark_reports"
    json_path = save_report_to_file(report, output_dir)

    print(f"Report saved to: {json_path}")
    print("\nReport Summary:")
    print(f"Model: {report['metadata']['model_name']}")
    print(f"Accuracy: {report['model_evaluation']['accuracy']:.2%}")
    print(f"F1-Score: {report['model_evaluation']['f1_score']:.2%}")
    print(f"Parameters: {report['complexity_analysis']['parameters']['total_parameters']:,}")
    print(f"Latency: {report['latency_benchmarking']['latency_stats']['batch_size_1']['mean_latency_ms']:.2f} ms")
    print(f"Recommendations: {len(report['recommendations'])} items")