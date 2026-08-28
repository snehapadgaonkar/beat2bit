"""
Model complexity analysis utilities for Beat2Bit project.
Calculates model size, parameter count, and computational complexity.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def count_model_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """
    Count total, trainable, and non-trainable parameters in a model.

    Args:
        model: TensorFlow/Keras model

    Returns:
        Dictionary with parameter counts
    """
    total_params = model.count_params()

    # Count trainable and non-trainable parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

    return {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'non_trainable_parameters': int(non_trainable_params)
    }


def estimate_model_size_mb(model: tf.keras.Model, precision: str = 'float32') -> Dict[str, float]:
    """
    Estimate model size in memory based on parameter count and precision.

    Args:
        model: TensorFlow/Keras model
        precision: Numerical precision ('float32', 'float16', 'int8')

    Returns:
        Dictionary with size estimates in different units
    """
    param_counts = count_model_parameters(model)
    total_params = param_counts['total_parameters']

    # Bytes per parameter based on precision
    precision_bytes = {
        'float32': 4,
        'float16': 2,
        'int8': 1,
        'uint8': 1
    }

    bytes_per_param = precision_bytes.get(precision, 4)
    size_bytes = total_params * bytes_per_param

    return {
        'size_bytes': size_bytes,
        'size_kb': size_bytes / 1024,
        'size_mb': size_bytes / (1024 * 1024),
        'precision_used': precision
    }


def calculate_flops(model: tf.keras.Model, batch_size: int = 1) -> Dict[str, Any]:
    """
    Calculate floating point operations (FLOPs) for a model.

    Note: This is an approximation as TensorFlow doesn't provide direct FLOPs counting
    for all layer types. For more accurate measurements, consider using
    tensorflow.profiler or thop library.

    Args:
        model: TensorFlow/Keras model
        batch_size: Batch size for FLOPs calculation

    Returns:
        Dictionary with FLOPs information
    """
    try:
        # Try to use TensorFlow's profiler if available
        import tensorflow as tf

        # Create a concrete function from the model
        concrete_func = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete_func.get_concrete_function(
            tf.TensorSpec([batch_size] + list(model.input_shape[1:]), model.inputs[0].dtype)
        )

        # Get graph definition
        graph_def = concrete_func.graph.as_graph_def()

        # Calculate FLOPs (this is simplified - real implementation would be more complex)
        # For now, we'll provide a basic approximation based on layer types
        flops_dict = _approximate_flops(model, batch_size)
        flops_dict['method'] = 'approximation'

    except Exception as e:
        logger.warning(f"Could not calculate precise FLOPs: {e}. Using approximation.")
        flops_dict = _approximate_flops(model, batch_size)
        flops_dict['method'] = 'approximation'

    return flops_dict


def _approximate_flops(model: tf.keras.Model, batch_size: int = 1) -> Dict[str, Any]:
    """
    Approximate FLOPs by analyzing layer types and dimensions.
    This is a simplified implementation for common layer types.
    """
    total_flops = 0
    layer_details = []

    # Get input shape
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]  # Assume single input for simplicity

    # Remove batch dimension
    if input_shape[0] is None:  # Variable batch size
        current_shape = [batch_size] + list(input_shape[1:])
    else:
        current_shape = list(input_shape)

    # Analyze each layer
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        layer_type = type(layer).__name__

        flops = 0

        if isinstance(layer, tf.keras.layers.Conv1D):
            # Conv1D: kernel_size * input_channels * output_channels * output_length
            kernel_size = layer.kernel_size[0]
            in_channels = current_shape[-1]
            out_channels = layer.filters

            # Calculate output length
            stride = layer.strides[0] if hasattr(layer, 'strides') else 1
            padding = layer.padding
            input_length = current_shape[-2]

            if padding == 'same':
                output_length = np.ceil(input_length / stride)
            else:  # 'valid'
                output_length = np.ceil((input_length - kernel_size + 1) / stride)

            flops = int(kernel_size * in_channels * out_channels * output_length)

            # Update shape for next layer
            current_shape = [current_shape[0], int(output_length), out_channels]

        elif isinstance(layer, tf.keras.layers.MaxPooling1D) or isinstance(layer, tf.keras.layers.AveragePooling1D):
            # Pooling: comparison operations (typically less expensive, often ignored in FLOPs)
            # For simplicity, we'll count as input_elements * pool_size
            pool_size = layer.pool_size[0] if hasattr(layer, 'pool_size') else 2
            stride = layer.strides[0] if hasattr(layer, 'strides') else pool_size
            padding = layer.padding
            input_length = current_shape[-2]
            in_channels = current_shape[-1]

            if padding == 'same':
                output_length = np.ceil(input_length / stride)
            else:  # 'valid'
                output_length = np.ceil((input_length - pool_size + 1) / stride)

            flops = int(input_length * in_channels)  # Simplified: one comparison per input element

            # Update shape for next layer
            current_shape = [current_shape[0], int(output_length), in_channels]

        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling1D):
            # Global average pooling: sum and divide for each channel
            in_channels = current_shape[-1]
            sequence_length = current_shape[-2]
            flops = int(2 * in_channels * sequence_length)  # sum + divide per element

            # Update shape for next layer
            current_shape = [current_shape[0], 1, in_channels]

        elif isinstance(layer, tf.keras.layers.Flatten):
            # Flatten: no FLOPs, just reshaping
            flops = 0
            # Update shape for next layer
            total_elements = np.prod(current_shape[1:])
            current_shape = [current_shape[0], int(total_elements)]

        elif isinstance(layer, tf.keras.layers.Dense):
            # Dense: input_size * output_size * 2 (multiply + add)
            input_size = current_shape[-1]
            output_size = layer.units
            flops = int(2 * input_size * output_size)

            # Update shape for next layer
            current_shape = [current_shape[0], output_size]

        else:
            # For other layers, assume minimal FLOPs or skip
            flops = 0
            # Try to preserve shape if possible
            if hasattr(layer, 'compute_output_shape'):
                try:
                    current_shape = list(layer.compute_output_shape(tuple(current_shape)))
                except:
                    pass  # Keep current shape if we can't compute

        total_flops += flops

        layer_details.append({
            'layer_index': i,
            'layer_name': layer_name,
            'layer_type': layer_type,
            'flops': flops,
            'output_shape': tuple(current_shape) if len(current_shape) > 1 else current_shape[0]
        })

    return {
        'total_flops': int(total_flops),
        'flops_per_sample': int(total_flops // batch_size) if batch_size > 0 else int(total_flops),
        'gflops': total_flops / 1e9,
        'mflops': total_flops / 1e6,
        'kflops': total_flops / 1e3,
        'layer_details': layer_details,
        'input_shape': tuple(model.input_shape) if isinstance(model.input_shape, tuple) else model.input_shape,
        'output_shape': tuple(model.output_shape) if isinstance(model.output_shape, tuple) else model.output_shape
    }


def analyze_model_complexity(model: tf.keras.Model, batch_size: int = 1) -> Dict[str, Any]:
    """
    Perform comprehensive model complexity analysis.

    Args:
        model: TensorFlow/Keras model
        batch_size: Batch size for analysis

    Returns:
        Dictionary containing all complexity metrics
    """
    # Parameter analysis
    param_counts = count_model_parameters(model)

    # Size analysis (FP32 baseline)
    size_fp32 = estimate_model_size_mb(model, 'float32')
    size_int8 = estimate_model_size_mb(model, 'int8')

    # FLOPs analysis
    flops_info = calculate_flops(model, batch_size)

    # Additional metrics - compute output shapes properly
    layers_info = []

    # Get input shape
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]  # Assume single input for simplicity

    # Remove batch dimension
    if input_shape[0] is None:  # Variable batch size
        current_shape = [batch_size] + list(input_shape[1:])
    else:
        current_shape = list(input_shape)

    # Analyze each layer
    for layer in model.layers:
        layer_name = layer.name
        layer_type = type(layer).__name__

        # Compute output shape for this layer (similar to _approximate_flops)
        output_shape = None
        if isinstance(layer, tf.keras.layers.Conv1D):
            # Conv1D: kernel_size * input_channels * output_channels * output_length
            kernel_size = layer.kernel_size[0]
            in_channels = current_shape[-1]
            out_channels = layer.filters

            # Calculate output length
            stride = layer.strides[0] if hasattr(layer, 'strides') else 1
            padding = layer.padding
            input_length = current_shape[-2]

            if padding == 'same':
                output_length = np.ceil(input_length / stride)
            else:  # 'valid'
                output_length = np.ceil((input_length - kernel_size + 1) / stride)

            output_shape = [current_shape[0], int(output_length), out_channels]
            # Update shape for next layer
            current_shape = output_shape

        elif isinstance(layer, tf.keras.layers.MaxPooling1D) or isinstance(layer, tf.keras.layers.AveragePooling1D):
            # Pooling
            pool_size = layer.pool_size[0] if hasattr(layer, 'pool_size') else 2
            stride = layer.strides[0] if hasattr(layer, 'strides') else pool_size
            padding = layer.padding
            input_length = current_shape[-2]
            in_channels = current_shape[-1]

            if padding == 'same':
                output_length = np.ceil(input_length / stride)
            else:  # 'valid'
                output_length = np.ceil((input_length - pool_size + 1) / stride)

            output_shape = [current_shape[0], int(output_length), in_channels]
            # Update shape for next layer
            current_shape = output_shape

        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling1D):
            # Global average pooling
            output_shape = [current_shape[0], 1, current_shape[-1]]
            # Update shape for next layer
            current_shape = output_shape

        elif isinstance(layer, tf.keras.layers.Flatten):
            # Flatten
            total_elements = np.prod(current_shape[1:])
            output_shape = [current_shape[0], int(total_elements)]
            # Update shape for next layer
            current_shape = output_shape

        elif isinstance(layer, tf.keras.layers.Dense):
            # Dense
            output_shape = [current_shape[0], layer.units]
            # Update shape for next layer
            current_shape = output_shape

        else:
            # For other layers, try to compute output shape if possible
            try:
                if hasattr(layer, 'compute_output_shape'):
                    computed_shape = layer.compute_output_shape(tuple(current_shape))
                    if isinstance(computed_shape, list):
                        output_shape = list(computed_shape)
                    else:
                        output_shape = list(computed_shape)
                    # Update shape for next layer
                    current_shape = output_shape
                else:
                    # If we can't compute, keep the same shape (conservative)
                    output_shape = current_shape[:]
            except:
                # If computation fails, keep the same shape
                output_shape = current_shape[:]

        layers_info.append({
            'name': layer_name,
            'type': layer_type,
            'output_shape': str(output_shape) if output_shape is not None else 'unknown'
        })

    return {
        'parameters': param_counts,
        'memory_size': {
            'fp32_mb': size_fp32['size_mb'],
            'int8_mb': size_int8['size_mb'],
            'compression_ratio_fp32_to_int8': size_fp32['size_mb'] / size_int8['size_mb'] if size_int8['size_mb'] > 0 else 0
        },
        'computational_complexity': flops_info,
        'architecture': {
            'total_layers': len(model.layers),
            'layer_types': [type(layer).__name__ for layer in model.layers],
            'layers_detail': layers_info
        },
        'batch_size_used': batch_size
    }


def compare_models_complexity(models: Dict[str, tf.keras.Model],
                            batch_size: int = 1) -> Dict[str, Any]:
    """
    Compare complexity metrics across multiple models.

    Args:
        models: Dictionary mapping model names to model instances
        batch_size: Batch size for analysis

    Returns:
        Dictionary containing comparison results
    """
    comparison_results = {}

    # Analyze each model
    for name, model in models.items():
        comparison_results[name] = analyze_model_complexity(model, batch_size)

    # Add comparative analysis
    if len(models) > 1:
        model_names = list(models.keys())

        # Parameter comparison
        param_comparison = {}
        for name in model_names:
            param_comparison[name] = comparison_results[name]['parameters']['total_parameters']

        # Size comparison
        size_comparison_fp32 = {}
        size_comparison_int8 = {}
        for name in model_names:
            size_comparison_fp32[name] = comparison_results[name]['memory_size']['fp32_mb']
            size_comparison_int8[name] = comparison_results[name]['memory_size']['int8_mb']

        # FLOPs comparison
        flops_comparison = {}
        for name in model_names:
            flops_comparison[name] = comparison_results[name]['computational_complexity']['total_flops']

        comparison_results['_comparative_analysis'] = {
            'parameter_counts': param_comparison,
            'model_sizes_fp32_mb': size_comparison_fp32,
            'model_sizes_int8_mb': size_comparison_int8,
            'total_flops': flops_comparison,
            'baseline_model': list(models.keys())[0] if model_names else None
        }

    return comparison_results


if __name__ == "__main__":
    # Example usage with a simple model
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

    print("Model Summary:")
    model.summary()

    print("\nComplexity Analysis:")
    complexity = analyze_model_complexity(model, batch_size=16)

    print(f"Parameters: {complexity['parameters']['total_parameters']:,}")
    print(f"Model Size (FP32): {complexity['memory_size']['fp32_mb']:.2f} MB")
    print(f"Model Size (INT8): {complexity['memory_size']['int8_mb']:.2f} MB")
    print(f"Compression Ratio: {complexity['memory_size']['compression_ratio_fp32_to_int8']:.2f}x")
    print(f"FLOPs: {complexity['computational_complexity']['total_flops']:,}")
    print(f"GFLOPs: {complexity['computational_complexity']['gflops']:.3f}")