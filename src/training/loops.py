"""
Training loops for Beat2Bit project.
Implements standard training, pruning-aware training, and quantization-aware training.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any, Callable
import logging
from pathlib import Path
import datetime

# Try to import tensorflow_model_optimization, make it optional
try:
    import tensorflow_model_optimization as tfmot
    TFMOT_AVAILABLE = True
except ImportError:
    TFMOT_AVAILABLE = False
    tfmot = None
    logger = logging.getLogger(__name__)
    logger.warning("tensorflow_model_optimization not available. Pruning and quantization features will be disabled.")

logger = logging.getLogger(__name__)

def create_callbacks(
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    reduce_lr_patience: int = 5,
    early_stopping_patience: int = 10,
    monitor: str = 'val_loss',
    save_best_only: bool = True
) -> list:
    """
    Create standard training callbacks.

    Args:
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
        reduce_lr_patience: Patience for ReduceLROnPlateau
        early_stopping_patience: Patience for EarlyStopping
        monitor: Metric to monitor for callbacks
        save_best_only: Whether to save only the best model

    Returns:
        List of Keras callbacks
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'model_{timestamp}_{{epoch:02d}}-{{val_loss:.2f}}.keras'),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),

        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, f'run_{timestamp}'),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        ),

        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ),

        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    return callbacks


def train_standard(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    callbacks: Optional[list] = None,
    verbose: int = 1
) -> tf.keras.callbacks.History:
    """
    Standard training loop.

    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        callbacks: List of Keras callbacks (optional)
        verbose: Verbosity mode

    Returns:
        Training history
    """
    if callbacks is None:
        callbacks = create_callbacks()

    logger.info(f"Starting standard training for {epochs} epochs")
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    logger.info(f"Batch size: {batch_size}")

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose
    )

    logger.info("Standard training completed")
    return history


def apply_pruning(
    model: tf.keras.Model,
    pruning_schedule: Any = None,
    block_size: Tuple[int, int] = (1, 1),
    block_pooling_type: str = 'AVG'
) -> tf.keras.Model:
    """
    Apply pruning to a Keras model.

    Args:
        model: Keras model to prune
        pruning_schedule: Pruning schedule (defaults to polynomial decay)
        block_size: Block size for pruning
        block_pooling_type: Pooling type for pruning blocks

    Returns:
        Pruned model
    """
    if not TFMOT_AVAILABLE:
        logger.warning("TensorFlow Model Optimization not available. Returning original model.")
        return model

    if pruning_schedule is None:
        # Default pruning schedule: start at 50% sparsity, end at 80% over training
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.50,
            final_sparsity=0.80,
            begin_step=0,
            end_step=1000  # Will be adjusted based on training steps
        )

    # Define pruning parameters
    pruning_params = {
        'pruning_schedule': pruning_schedule,
        'block_size': block_size,
        'block_pooling_type': block_pooling_type
    }

    # Apply pruning to the model
    try:
        # For TensorFlow Model Optimization toolkit
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        logger.info("Applied pruning to model")
        return pruned_model
    except Exception as e:
        logger.error(f"Failed to apply pruning: {e}")
        # Return original model if pruning fails
        return model


def train_with_pruning(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    initial_sparsity: float = 0.50,
    final_sparsity: float = 0.80,
    callbacks: Optional[list] = None,
    verbose: int = 1
) -> Tuple[tf.keras.callbacks.History, tf.keras.Model]:
    """
    Training loop with pruning support.

    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        initial_sparsity: Initial sparsity fraction (0.0 to 1.0)
        final_sparsity: Final sparsity fraction (0.0 to 1.0)
        callbacks: List of Keras callbacks (optional)
        verbose: Verbosity mode

    Returns:
        Tuple of (training history, pruned model)
    """
    if not TFMOT_AVAILABLE:
        logger.warning("TensorFlow Model Optimization not available. Falling back to standard training.")
        history = train_standard(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=verbose
        )
        return history, model

    # Calculate training steps for pruning schedule
    steps_per_epoch = len(X_train) // batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    total_steps = steps_per_epoch * epochs

    # Create pruning schedule
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        begin_step=0,
        end_step=total_steps
    )

    # Apply pruning to model
    pruned_model = apply_pruning(model, pruning_schedule)

    # Pruning-specific callbacks
    pruning_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='logs/pruning', profile_batch=0)
    ]

    # Combine with standard callbacks
    if callbacks is None:
        all_callbacks = create_callbacks() + pruning_callbacks
    else:
        all_callbacks = callbacks + pruning_callbacks

    logger.info(f"Starting pruning-aware training for {epochs} epochs")
    logger.info(f"Initial sparsity: {initial_sparsity}, Final sparsity: {final_sparsity}")

    # Compile pruned model (needed after pruning wrapper)
    pruned_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.metrics
    )

    history = pruned_model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=all_callbacks,
        verbose=verbose
    )

    # Strip pruning wrappers for deployment
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    logger.info("Pruning-aware training completed and pruning wrappers stripped")

    return history, final_model


def apply_quantization_aware_training(
    model: tf.keras.Model
) -> tf.keras.Model:
    """
    Apply quantization-aware training to a Keras model.

    Args:
        model: Keras model to prepare for QAT

    Returns:
        Model prepared for quantization-aware training
    """
    if not TFMOT_AVAILABLE:
        logger.warning("TensorFlow Model Optimization not available. Returning original model.")
        return model

    try:
        # Define QAT annotation scope
        qat_annotate_model = tfmot.quantization.keras.quantize_model

        # Apply QAT annotation
        q_aware_model = qat_annotate_model(model)
        logger.info("Applied quantization-aware training annotation to model")
        return q_aware_model
    except Exception as e:
        logger.error(f"Failed to apply quantization-aware training: {e}")
        return model


def train_with_quantization_aware(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    callbacks: Optional[list] = None,
    verbose: int = 1
) -> Tuple[tf.keras.callbacks.History, tf.keras.Model]:
    """
    Training loop with quantization-aware training support.

    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        callbacks: List of Keras callbacks (optional)
        verbose: Verbosity mode

    Returns:
        Tuple of (training history, QAT model)
    """
    # Apply QAT annotation
    q_aware_model = apply_quantization_aware_training(model)

    # Recompile model (required after QAT annotation)
    q_aware_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.metrics
    )

    # Standard callbacks
    if callbacks is None:
        callbacks = create_callbacks()

    logger.info(f"Starting quantization-aware training for {epochs} epochs")

    history = q_aware_model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose
    )

    logger.info("Quantization-aware training completed")
    return history, q_aware_model


def convert_to_tflite(
    model: tf.keras.Model,
    output_path: str = "model.tflite",
    optimizations: list = None,
    representative_data: Optional[np.ndarray] = None,
    inference_type: str = "float"  # "float", "int8", "uint8"
) -> bool:
    """
    Convert Keras model to TensorFlow Lite format.

    Args:
        model: Trained Keras model
        output_path: Path to save TFLite model
        optimizations: List of optimizations to apply
        representative_data: Dataset for quantization calibration
        inference_type: Type of inference ("float", "int8", "uint8")

    Returns:
        True if conversion successful
    """
    try:
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Set optimizations
        if optimizations is None:
            optimizations = [tf.lite.Optimize.DEFAULT]

        converter.optimizations = optimizations

        # Set inference type
        if inference_type == "int8":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif inference_type == "uint8":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        # For float, use default settings

        # Set representative dataset for quantization
        if representative_data is not None and inference_type in ["int8", "uint8"]:
            def representative_data_gen():
                for i in range(min(100, len(representative_data))):
                    yield [representative_data[i:i+1].astype(np.float32)]

            converter.representative_dataset = representative_data_gen

        # Convert model
        tflite_model = converter.convert()

        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        model_size = os.path.getsize(output_path)
        logger.info(f"Model converted to TFLite and saved to {output_path}")
        logger.info(f"TFLite model size: {model_size} bytes ({model_size/1024:.2f} KB)")

        return True

    except Exception as e:
        logger.error(f"Failed to convert model to TFLite: {e}")
        return False


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    training_mode: str = "standard",
    pruning_sparsity: Tuple[float, float] = (0.5, 0.8),
    qat_inference_type: str = "int8",
    tflite_output_path: str = "model.tflite",
    save_tflite: bool = True,
    callbacks: Optional[list] = None,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Unified training function that supports different training modes.

    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        training_mode: Type of training ("standard", "pruning", "qat")
        pruning_sparsity: Tuple of (initial, final) sparsity for pruning
        qat_inference_type: Inference type for QAT ("int8", "uint8", "float")
        tflite_output_path: Path to save TFLite model
        save_tflite: Whether to convert and save TFLite model
        callbacks: List of Keras callbacks (optional)
        verbose: Verbosity mode

    Returns:
        Dictionary containing training results and artifacts
    """
    results = {
        'model': model,
        'history': None,
        'tflite_path': None,
        'training_mode': training_mode
    }

    # Train based on mode
    if training_mode == "standard":
        history = train_standard(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=verbose
        )
        results['history'] = history
        results['model'] = model

    elif training_mode == "pruning":
        initial_sparsity, final_sparsity = pruning_sparsity
        history, pruned_model = train_with_pruning(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            callbacks=callbacks, verbose=verbose
        )
        results['history'] = history
        results['model'] = pruned_model

    elif training_mode == "qat":
        history, qat_model = train_with_quantization_aware(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=verbose
        )
        results['history'] = history
        results['model'] = qat_model

    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

    # Convert to TFLite if requested
    if save_tflite:
        # Use a subset of training data for representative dataset
        rep_data = X_train[:min(100, len(X_train))] if len(X_train) > 0 else None

        success = convert_to_tflite(
            results['model'],
            output_path=tflite_output_path,
            representative_data=rep_data,
            inference_type="int8" if training_mode == "qat" else "float"
        )

        if success:
            results['tflite_path'] = tflite_output_path
        else:
            logger.warning("TFLite conversion failed")

    return results


if __name__ == "__main__":
    # Example usage
    print("Training Loops Module")
    print("====================")

    # Create a simple model for testing
    from src.models.architectures import create_baseline_1dcnn, compile_model

    model = create_baseline_1dcnn(input_shape=(180, 1), num_classes=2)
    model = compile_model(model, learning_rate=0.001)

    # Create dummy data
    X_train = np.random.randn(100, 180, 1).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(20, 180, 1).astype(np.float32)
    y_val = np.random.randint(0, 2, 20)

    print(f"Created model with {model.count_params():,} parameters")
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")

    # Test standard training (short run for testing)
    print("\nTesting standard training (2 epochs)...")
    try:
        results = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=2, batch_size=16,
            training_mode="standard",
            save_tflite=False,  # Skip TFLite conversion for quick test
            verbose=1
        )
        print(f"Standard training completed. Final val accuracy: {results['history'].history['val_accuracy'][-1]:.4f}")
    except Exception as e:
        print(f"Standard training failed: {e}")

    print("\n✓ Training loops module ready!")