"""
ECG arrhythmia detection model architectures for Beat2Bit project.
Implements baseline 1D CNN architectures suitable for edge deployment.
"""

import tensorflow as tf
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def create_baseline_1dcnn(
    input_shape: Tuple[int, int] = (180, 1),
    num_classes: int = 2,
    filters: list = [16, 32, 64],
    kernel_sizes: list = [7, 5, 3],
    pool_sizes: list = [2, 2, 2],
    dense_units: list = [64, 32],
    dropout_rate: float = 0.5,
    use_batch_norm: bool = True,
    activation: str = 'relu',
    final_activation: str = 'softmax'
) -> tf.keras.Model:
    """
    Create a baseline 1D CNN for ECG arrhythmia detection.

    Args:
        input_shape: Shape of input ECG windows (samples, channels)
        num_classes: Number of output classes
        filters: List of filter counts for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        pool_sizes: List of pool sizes for each pooling layer
        dense_units: List of units for each dense layer
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        activation: Activation function for hidden layers
        final_activation: Activation function for output layer

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name='ecg_input')
    x = inputs

    # Convolutional blocks
    for i, (filt, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
        x = tf.keras.layers.Conv1D(
            filters=filt,
            kernel_size=kernel_size,
            activation=activation,
            padding='same',
            name=f'conv1d_{i+1}'
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)

        x = tf.keras.layers.MaxPooling1D(
            pool_size=pool_size,
            name=f'maxpool_{i+1}'
        )(x)

        # Optional dropout after pooling
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)

    # Global average pooling to reduce dimensionality
    x = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

    # Dense layers
    for i, units in enumerate(dense_units):
        x = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            name=f'dense_{i+1}'
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)

        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)

    # Output layer
    outputs = tf.keras.layers.Dense(
        units=num_classes,
        activation=final_activation,
        name='predictions'
    )(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='baseline_1dcnn_ecg')

    logger.info(f"Created baseline 1D CNN model with {model.count_params():,} parameters")
    return model


def create_deeper_1dcnn(
    input_shape: Tuple[int, int] = (180, 1),
    num_classes: int = 2,
    filters: list = [32, 64, 128, 256],
    kernel_sizes: list = [9, 7, 5, 3],
    pool_sizes: list = [2, 2, 2, 2],
    dense_units: list = [128, 64],
    dropout_rate: float = 0.5,
    use_batch_norm: bool = True,
    activation: str = 'relu',
    final_activation: str = 'softmax'
) -> tf.keras.Model:
    """
    Create a deeper 1D CNN for ECG arrhythmia detection.

    Args:
        input_shape: Shape of input ECG windows (samples, channels)
        num_classes: Number of output classes
        filters: List of filter counts for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        pool_sizes: List of pool sizes for each pooling layer
        dense_units: List of units for each dense layer
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        activation: Activation function for hidden layers
        final_activation: Activation function for output layer

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name='ecg_input')
    x = inputs

    # Convolutional blocks with increasing filters
    for i, (filt, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
        x = tf.keras.layers.Conv1D(
            filters=filt,
            kernel_size=kernel_size,
            activation=activation,
            padding='same',
            name=f'conv1d_{i+1}'
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)

        x = tf.keras.layers.MaxPooling1D(
            pool_size=pool_size,
            name=f'maxpool_{i+1}'
        )(x)

        # Optional dropout after pooling
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

    # Dense layers
    for i, units in enumerate(dense_units):
        x = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            name=f'dense_{i+1}'
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)

        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)

    # Output layer
    outputs = tf.keras.layers.Dense(
        units=num_classes,
        activation=final_activation,
        name='predictions'
    )(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='deeper_1dcnn_ecg')

    logger.info(f"Created deeper 1D CNN model with {model.count_params():,} parameters")
    return model


def create_residual_1dcnn(
    input_shape: Tuple[int, int] = (180, 1),
    num_classes: int = 2,
    filters: list = [32, 64, 128],
    kernel_size: int = 7,
    pool_size: int = 2,
    dense_units: list = [64],
    dropout_rate: float = 0.5,
    use_batch_norm: bool = True,
    activation: str = 'relu',
    final_activation: str = 'softmax'
) -> tf.keras.Model:
    """
    Create a residual 1D CNN for ECG arrhythmia detection.

    Args:
        input_shape: Shape of input ECG windows (samples, channels)
        num_classes: Number of output classes
        filters: List of filter counts for each conv block
        kernel_size: Kernel size for conv layers
        pool_size: Pool size for max pooling
        dense_units: List of units for each dense layer
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        activation: Activation function for hidden layers
        final_activation: Activation function for output layer

    Returns:
        Compiled Keras model
    """
    def residual_block(x, filters, kernel_size, stride=1, use_batch_norm=True, activation='relu', name=None):
        """Create a residual block with two conv layers."""
        shortcut = x

        # First conv layer
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            activation=None,
            name=f'{name}_conv1' if name else None
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1' if name else None)(x)

        x = tf.keras.layers.Activation(activation)(x)

        # Second conv layer
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation=None,
            name=f'{name}_conv2' if name else None
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2' if name else None)(x)

        # Adjust shortcut if needed (for dimension matching)
        if shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=1,
                strides=stride,
                padding='same',
                name=f'{name}_shortcut' if name else None
            )(shortcut)

        # Add shortcut connection
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation(activation)(x)

        return x

    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name='ecg_input')
    x = inputs

    # Initial conv layer
    x = tf.keras.layers.Conv1D(
        filters=filters[0],
        kernel_size=kernel_size,
        activation=activation,
        padding='same',
        name='initial_conv'
    )(x)

    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name='initial_batch_norm')(x)

    # Residual blocks
    for i, filt in enumerate(filters):
        x = residual_block(
            x,
            filters=filt,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            activation=activation,
            name=f'residual_block_{i+1}'
        )

        # Add pooling after each residual block (except last)
        if i < len(filters) - 1:
            x = tf.keras.layers.MaxPooling1D(
                pool_size=pool_size,
                name=f'residual_pool_{i+1}'
            )(x)

            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate, name=f'residual_dropout_{i+1}')(x)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

    # Dense layers
    for i, units in enumerate(dense_units):
        x = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            name=f'dense_{i+1}'
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)

        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)

    # Output layer
    outputs = tf.keras.layers.Dense(
        units=num_classes,
        activation=final_activation,
        name='predictions'
    )(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='residual_1dcnn_ecg')

    logger.info(f"Created residual 1D CNN model with {model.count_params():,} parameters")
    return model


def create_lightweight_1dcnn(
    input_shape: Tuple[int, int] = (180, 1),
    num_classes: int = 2,
    filters: list = [8, 16],
    kernel_sizes: list = [5, 3],
    pool_sizes: list = [2, 2],
    dense_units: list = [16],
    dropout_rate: float = 0.3,
    use_batch_norm: bool = False,
    activation: str = 'relu',
    final_activation: str = 'softmax'
) -> tf.keras.Model:
    """
    Create a lightweight 1D CNN for edge deployment.

    Args:
        input_shape: Shape of input ECG windows (samples, channels)
        num_classes: Number of output classes
        filters: List of filter counts for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        pool_sizes: List of pool sizes for each pooling layer
        dense_units: List of units for each dense layer
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        activation: Activation function for hidden layers
        final_activation: Activation function for output layer

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name='ecg_input')
    x = inputs

    # Convolutional blocks
    for i, (filt, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
        x = tf.keras.layers.Conv1D(
            filters=filt,
            kernel_size=kernel_size,
            activation=activation,
            padding='same',
            name=f'conv1d_{i+1}'
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)

        x = tf.keras.layers.MaxPooling1D(
            pool_size=pool_size,
            name=f'maxpool_{i+1}'
        )(x)

        # Optional dropout after pooling
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

    # Dense layers
    for i, units in enumerate(dense_units):
        x = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            name=f'dense_{i+1}'
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)

        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)

    # Output layer
    outputs = tf.keras.layers.Dense(
        units=num_classes,
        activation=final_activation,
        name='predictions'
    )(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='lightweight_1dcnn_ecg')

    logger.info(f"Created lightweight 1D CNN model with {model.count_params():,} parameters")
    return model


def model_factory(
    architecture: str = 'baseline',
    input_shape: Tuple[int, int] = (180, 1),
    num_classes: int = 2,
    **kwargs
) -> tf.keras.Model:
    """
    Factory function to create different ECG CNN architectures.

    Args:
        architecture: Type of architecture ('baseline', 'deeper', 'residual', 'lightweight')
        input_shape: Shape of input ECG windows (samples, channels)
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to specific architecture functions

    Returns:
        Compiled Keras model
    """
    architectures = {
        'baseline': create_baseline_1dcnn,
        'deeper': create_deeper_1dcnn,
        'residual': create_residual_1dcnn,
        'lightweight': create_lightweight_1dcnn
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Available options: {list(architectures.keys())}")

    model_func = architectures[architecture]
    model = model_func(
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )

    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = 'adam',
    loss: str = 'sparse_categorical_crossentropy',
    metrics: list = None
) -> tf.keras.Model:
    """
    Compile a Keras model with specified optimizer and metrics.

    Args:
        model: Uncompiled Keras model
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        loss: Loss function
        metrics: List of metrics to track

    Returns:
        Compiled Keras model
    """
    if metrics is None:
        metrics = ['accuracy']

    # Create optimizer
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )

    logger.info(f"Compiled model with {optimizer} optimizer, lr={learning_rate}")
    return model


if __name__ == "__main__":
    # Example usage and testing
    print("ECG Model Architectures")
    print("======================")

    # Test baseline model
    print("\n1. Testing baseline 1D CNN:")
    baseline_model = create_baseline_1dcnn(input_shape=(180, 1), num_classes=2)
    baseline_model = compile_model(baseline_model)
    baseline_model.summary()

    # Test deeper model
    print("\n2. Testing deeper 1D CNN:")
    deeper_model = create_deeper_1dcnn(input_shape=(180, 1), num_classes=2)
    deeper_model = compile_model(deeper_model)
    deeper_model.summary()

    # Test residual model
    print("\n3. Testing residual 1D CNN:")
    residual_model = create_residual_1dcnn(input_shape=(180, 1), num_classes=2)
    residual_model = compile_model(residual_model)
    residual_model.summary()

    # Test lightweight model
    print("\n4. Testing lightweight 1D CNN:")
    lightweight_model = create_lightweight_1dcnn(input_shape=(180, 1), num_classes=2)
    lightweight_model = compile_model(lightweight_model)
    lightweight_model.summary()

    # Test model factory
    print("\n5. Testing model factory:")
    factory_model = model_factory(
        architecture='baseline',
        input_shape=(180, 1),
        num_classes=2,
        filters=[32, 64],
        dropout_rate=0.3
    )
    factory_model = compile_model(factory_model)
    print(f"Factory model parameters: {factory_model.count_params():,}")

    print("\n✓ All model architectures created successfully!")