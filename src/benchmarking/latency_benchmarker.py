"""
Latency benchmarking utilities for Beat2Bit project.
Measures inference latency simulating edge device constraints.
"""

import time
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def benchmark_inference_latency(model: tf.keras.Model,
                              input_data: np.ndarray,
                              batch_sizes: List[int] = [1, 4, 8, 16, 32],
                              n_warmup: int = 10,
                              n_measurements: int = 100) -> Dict[str, Any]:
    """
    Benchmark inference latency for a model with different batch sizes.

    Args:
        model: TensorFlow/Keras model to benchmark
        input_data: Input data for benchmarking (will be sliced for different batch sizes)
        batch_sizes: List of batch sizes to test
        n_warmup: Number of warmup runs before measurements
        n_measurements: Number of measurement runs for each batch size

    Returns:
        Dictionary containing latency metrics for each batch size
    """
    results = {
        'batch_sizes': batch_sizes,
        'latency_stats': {},
        'throughput_stats': {},
        'summary': {}
    }

    # Ensure we have enough input data
    max_batch_size = max(batch_sizes)
    if len(input_data) < max_batch_size:
        # Repeat input data if necessary
        repeat_factor = int(np.ceil(max_batch_size / len(input_data)))
        input_data = np.tile(input_data, (repeat_factor, 1, 1))[:max_batch_size]
        logger.warning(f"Repeated input data to achieve batch size {max_batch_size}")

    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size {batch_size}")

        # Prepare batch data
        batch_data = input_data[:batch_size]

        # Warmup runs
        for _ in range(n_warmup):
            _ = model.predict(batch_data, verbose=0)

        # Measurement runs
        latencies = []

        for _ in range(n_measurements):
            start_time = time.perf_counter()
            _ = model.predict(batch_data, verbose=0)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

        latencies = np.array(latencies)

        # Calculate statistics
        results['latency_stats'][f'batch_size_{batch_size}'] = {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'n_measurements': n_measurements
        }

        # Calculate throughput (samples/second)
        mean_latency_sec = np.mean(latencies) / 1000
        throughput = batch_size / mean_latency_sec if mean_latency_sec > 0 else 0

        results['throughput_stats'][f'batch_size_{batch_size}'] = {
            'throughput_samples_per_sec': float(throughput),
            'latency_per_sample_ms': float(np.mean(latencies) / batch_size) if batch_size > 0 else 0
        }

    # Calculate summary statistics
    if batch_sizes:
        # Find optimal batch size for latency (minimum mean latency per sample)
        latencies_per_sample = {}
        for batch_size in batch_sizes:
            key = f'batch_size_{batch_size}'
            mean_latency = results['latency_stats'][key]['mean_latency_ms']
            latency_per_sample = mean_latency / batch_size if batch_size > 0 else float('inf')
            latencies_per_sample[batch_size] = latency_per_sample

        optimal_batch_size = min(latencies_per_sample, key=latencies_per_sample.get) if latencies_per_sample else 1

        # Find optimal batch size for throughput (maximum throughput)
        throughputs = {}
        for batch_size in batch_sizes:
            key = f'batch_size_{batch_size}'
            throughput = results['throughput_stats'][key]['throughput_samples_per_sec']
            throughputs[batch_size] = throughput

        optimal_batch_size_throughput = max(throughputs, key=throughputs.get) if throughputs else 1

        results['summary'] = {
            'optimal_batch_size_for_latency': optimal_batch_size,
            'optimal_batch_size_for_throughput': optimal_batch_size_throughput,
            'latency_range_ms': {
                'min': min(results['latency_stats'][f'batch_size_{bs}']['mean_latency_ms'] for bs in batch_sizes),
                'max': max(results['latency_stats'][f'batch_size_{bs}']['mean_latency_ms'] for bs in batch_sizes)
            },
            'throughput_range_samples_per_sec': {
                'min': min(results['throughput_stats'][f'batch_size_{bs}']['throughput_samples_per_sec'] for bs in batch_sizes),
                'max': max(results['throughput_stats'][f'batch_size_{bs}']['throughput_samples_per_sec'] for bs in batch_sizes)
            }
        }

    return results


def benchmark_single_inference_latency(model: tf.keras.Model,
                                     input_data: np.ndarray,
                                     n_warmup: int = 50,
                                     n_measurements: int = 200) -> Dict[str, float]:
    """
    Benchmark single-sample inference latency (batch size = 1).
    This simulates real-time edge device operation where samples are processed one at a time.

    Args:
        model: TensorFlow/Keras model to benchmark
        input_data: Input data (should have at least one sample)
        n_warmup: Number of warmup runs
        n_measurements: Number of measurement runs

    Returns:
        Dictionary containing single-sample latency metrics
    """
    # Ensure we have at least one sample
    if len(input_data) == 0:
        raise ValueError("Input data must contain at least one sample")

    # Use first sample for single-inference benchmarking
    single_sample = input_data[0:1]  # Keep batch dimension

    # Warmup runs
    for _ in range(n_warmup):
        _ = model.predict(single_sample, verbose=0)

    # Measurement runs
    latencies = []

    for _ in range(n_measurements):
        start_time = time.perf_counter()
        _ = model.predict(single_sample, verbose=0)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    latencies = np.array(latencies)

    return {
        'mean_latency_ms': float(np.mean(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'n_measurements': n_measurements
    }


def simulate_edge_device_latency(model: tf.keras.Model,
                               input_data: np.ndarray,
                               target_fps: float = 50.0,
                               n_measurements: int = 100) -> Dict[str, Any]:
    """
    Simulate edge device operation by measuring if the model can meet target FPS.

    Args:
        model: TensorFlow/Keras model to benchmark
        input_data: Input data for benchmarking
        target_fps: Target frames per second (e.g., 50 FPS for ECG processing)
        n_measurements: Number of measurement runs

    Returns:
        Dictionary containing edge device suitability metrics
    """
    target_latency_ms = 1000.0 / target_fps  # Maximum latency per sample to achieve target FPS

    # Benchmark single-sample latency
    single_latency = benchmark_single_inference_latency(
        model, input_data, n_warmup=20, n_measurements=n_measurements
    )

    mean_latency = single_latency['mean_latency_ms']
    achievable_fps = 1000.0 / mean_latency if mean_latency > 0 else 0

    # Calculate how much headroom we have
    latency_headroom_ms = target_latency_ms - mean_latency
    fps_headroom = achievable_fps - target_fps

    # Determine if model meets real-time requirements
    meets_realtime = mean_latency <= target_latency_ms

    return {
        'target_fps': target_fps,
        'target_latency_ms': target_latency_ms,
        'achieved_mean_latency_ms': mean_latency,
        'achieved_fps': achievable_fps,
        'latency_headroom_ms': latency_headroom_ms,
        'fps_headroom': fps_headroom,
        'meets_realtime_requirement': meets_realtime,
        'detailed_latency_stats': single_latency
    }


def benchmark_model_variants_latency(models: Dict[str, tf.keras.Model],
                                   input_data: np.ndarray,
                                   batch_sizes: List[int] = [1, 4, 8, 16],
                                   n_warmup: int = 10,
                                   n_measurements: int = 50) -> Dict[str, Any]:
    """
    Benchmark latency across multiple model variants for comparison.

    Args:
        models: Dictionary mapping model names to model instances
        input_data: Input data for benchmarking
        batch_sizes: List of batch sizes to test
        n_warmup: Number of warmup runs
        n_measurements: Number of measurement runs

    Returns:
        Dictionary containing latency comparison results
    """
    results = {
        'models': list(models.keys()),
        'batch_sizes': batch_sizes,
        'model_latency': {},
        'comparison': {}
    }

    # Benchmark each model
    for name, model in models.items():
        logger.info(f"Benchmarking model: {name}")
        model_results = benchmark_inference_latency(
            model, input_data, batch_sizes, n_warmup, n_measurements
        )
        results['model_latency'][name] = model_results

    # Add comparative analysis
    if len(models) > 1:
        # Compare single-sample latency (most relevant for edge devices)
        single_sample_latencies = {}
        for name in models.keys():
            key = f'batch_size_1'
            if key in results['model_latency'][name]['latency_stats']:
                latency = results['model_latency'][name]['latency_stats'][key]['mean_latency_ms']
                single_sample_latencies[name] = latency

        # Compare throughput at batch size 1 (samples/sec)
        throughput_at_bs1 = {}
        for name in models.keys():
            key = f'batch_size_1'
            if key in results['model_latency'][name]['throughput_stats']:
                throughput = results['model_latency'][name]['throughput_stats'][key]['throughput_samples_per_sec']
                throughput_at_bs1[name] = throughput

        results['comparison'] = {
            'single_sample_latency_ms': single_sample_latencies,
            'throughput_samples_per_sec_bs1': throughput_at_bs1,
            'fastest_model_latency': min(single_sample_latencies, key=single_sample_latencies.get) if single_sample_latencies else None,
            'highest_throughput_model': max(throughput_at_bs1, key=throughput_at_bs1.get) if throughput_at_bs1 else None
        }

    return results


def estimate_energy_consumption(latency_ms: float,
                              power_consumption_mw: float = 100.0) -> Dict[str, float]:
    """
    Estimate energy consumption for inference based on latency and power consumption.
    This is a simplified model - actual energy consumption depends on specific hardware.

    Args:
        latency_ms: Inference latency in milliseconds
        power_consumption_mw: Power consumption during inference in milliwatts

    Returns:
        Dictionary containing energy consumption estimates
    """
    # Energy = Power × Time
    latency_sec = latency_ms / 1000.0
    energy_joules = (power_consumption_mw / 1000.0) * latency_sec  # Convert mW to W
    energy_microjoules = energy_joules * 1e6

    return {
        'latency_ms': latency_ms,
        'power_consumption_mw': power_consumption_mw,
        'energy_joules': energy_joules,
        'energy_microjoules': energy_microjoules,
        'assumption': 'Based on constant power consumption during inference'
    }


if __name__ == "__main__":
    # Example usage
    # Create a simple model for demonstration
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

    # Create dummy input data
    input_data = np.random.randn(32, 180, 1).astype(np.float32)

    print("Benchmarking model latency...")
    latency_results = benchmark_inference_latency(
        model, input_data, batch_sizes=[1, 4, 8, 16], n_warmup=5, n_measurements=20
    )

    print("\nLatency Results:")
    for batch_size in [1, 4, 8, 16]:
        key = f'batch_size_{batch_size}'
        if key in latency_results['latency_stats']:
            stats = latency_results['latency_stats'][key]
            print(f"Batch Size {batch_size}:")
            print(f"  Mean Latency: {stats['mean_latency_ms']:.2f} ms")
            print(f"  P95 Latency: {stats['p95_latency_ms']:.2f} ms")
            print(f"  Throughput: {latency_results['throughput_stats'][key]['throughput_samples_per_sec']:.1f} samples/sec")

    # Single sample benchmark
    print("\nSingle Sample Latency (Edge Device Simulation):")
    single_latency = benchmark_single_inference_latency(model, input_data, n_warmup=5, n_measurements=20)
    print(f"Mean Latency: {single_latency['mean_latency_ms']:.2f} ms")
    print(f"P95 Latency: {single_latency['p95_latency_ms']:.2f} ms")

    # Edge device suitability
    print("\nEdge Device Suitability (Target: 50 FPS):")
    edge_metrics = simulate_edge_device_latency(model, input_data, target_fps=50.0, n_measurements=20)
    print(f"Target Latency: {edge_metrics['target_latency_ms']:.2f} ms")
    print(f"Achieved Latency: {edge_metrics['achieved_mean_latency_ms']:.2f} ms")
    print(f"Achieved FPS: {edge_metrics['achieved_fps']:.1f}")
    print(f"Meets Realtime: {edge_metrics['meets_realtime_requirement']}")