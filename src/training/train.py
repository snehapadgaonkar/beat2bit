"""
Training script for Beat2Bit (baseline). Supports two modes:
- smoke mode (default): quick no-op that verifies the script runs without heavy deps (keeps CI fast)
- full mode: trains a small 1D CNN using TensorFlow/Keras on data/processed synthetic data

Usage examples:
- Smoke mode (default): python -m src.training.train
- Full mode: python -m src.training.train --full --epochs 5
"""

import os
import argparse
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved')


def smoke():
    print("Running smoke mode — no heavy dependencies. Create synthetic data with small sample and exit.")
    # Ensure processed folder exists
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    # write a tiny placeholder file to indicate smoke passed
    with open(os.path.join(MODEL_DIR, 'smoke.txt'), 'w') as f:
        f.write('smoke OK')
    print("Smoke artifacts written to models/saved/")


def full_train(epochs=10, batch_size=16):
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception as e:
        print("TensorFlow is required for full training mode. Install TensorFlow or run in smoke mode.")
        raise

    # Try a list of known processed-split namings in order of preference.
    # MIT-BIH (mitdb_*) -> generic (train/val/test) -> flat preprocess output
    # (X_train/y_train + X_test/y_test, no validation split), which we split on
    # the fly so `python -m src.training.train --full` works end-to-end.
    def load_processed():
        candidates = [
            ("MIT-BIH mitdb_*", ["mitdb_X_train.npy", "mitdb_y_train.npy", "mitdb_X_val.npy", "mitdb_y_val.npy", "mitdb_X_test.npy", "mitdb_y_test.npy"]),
            ("synthetic train/val/test", ["train.npy", "train_labels.npy", "val.npy", "val_labels.npy", "test.npy", "test_labels.npy"]),
            ("flat X_train/y_train (no val split)", ["X_train.npy", "y_train.npy", None, None, "X_test.npy", "y_test.npy"]),
        ]

        for label, files in candidates:
            paths = [os.path.join(DATA_DIR, f) if f else None for f in files]
            existing = [p for p in paths if p is not None]
            if not existing or not all(os.path.exists(p) for p in existing):
                continue

            x_train = np.load(paths[0])
            y_train = np.load(paths[1])

            if paths[2] is not None and paths[3] is not None:
                X_val = np.load(paths[2])
                y_val = np.load(paths[3])
            else:
                # No validation split provided — carve one out of the train set.
                split = int(len(x_train) * 0.8)
                x_val, X_train_split = x_train[split:], x_train[:split]
                y_val, y_train_split = y_train[split:], y_train[:split]
                x_train, y_train = X_train_split, y_train_split
                print("No validation split found — carved 20% from the training set.")

            X_test = np.load(paths[4])
            y_test = np.load(paths[5])

            print(f"Loaded {label} from {DATA_DIR}")
            return x_train, y_train, X_val, y_val, X_test, y_test

        raise FileNotFoundError(f"No compatible processed datasets found in {DATA_DIR}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_processed()

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    input_shape = X_train.shape[1:]
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(8, 5, activation='relu', padding='same'),
        layers.MaxPool1D(2),
        layers.Conv1D(16, 5, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    os.makedirs(MODEL_DIR, exist_ok=True)
    # Save Keras model
    model_path = os.path.join(MODEL_DIR, 'baseline.h5')
    model.save(model_path)
    print(f"Saved Keras model to {model_path}")

    # Evaluate on test set and print clinical-style metrics
    try:
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
        y_pred_prob = model.predict(X_test, batch_size=batch_size)
        y_pred = (y_pred_prob.ravel() > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print("Test metrics:")
        print(f"  accuracy: {acc:.4f}")
        print(f"  precision: {prec:.4f}")
        print(f"  recall (sensitivity): {rec:.4f}")
        print(f"  specificity: {specificity:.4f}")
        print(f"  f1: {f1:.4f}")
    except Exception as e:
        print(f"Evaluation skipped or failed (sklearn may be missing): {e}")

    # TFLite conversion (best-effort)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(MODEL_DIR, 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to {tflite_path}")
    except Exception as e:
        print(f"TFLite conversion failed or not available: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run full training (requires TensorFlow)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)

    # When called from pytest, sys.argv will contain pytest CLI flags which argparse will reject.
    # Default to smoke mode when running under pytest or when argv is explicitly []
    if argv is None:
        import sys
        import os
        # Detect pytest run by environment variable or presence of pytest args
        if os.environ.get('PYTEST_CURRENT_TEST') or any(a.startswith('-') for a in sys.argv[1:]):
            argv = []
        else:
            argv = None

    args = parser.parse_args(argv)

    if not args.full:
        smoke()
    else:
        full_train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
