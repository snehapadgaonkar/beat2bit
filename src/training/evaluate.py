"""
Evaluation script for Beat2Bit.
Loads a trained Keras model (models/saved/baseline.h5) and evaluates on processed test split (MIT-BIH or synthetic), computing accuracy, precision, recall, specificity, F1, and saves a JSON/text report under experiments/.

Usage:
  python -m src.training.evaluate
"""

import os
import json
import numpy as np
from datetime import datetime

EXPERIMENT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')


def load_test_split():
    # Try MIT-BIH processed naming first
    mit_x_test = os.path.join(DATA_DIR, 'mitdb_X_test.npy')
    mit_y_test = os.path.join(DATA_DIR, 'mitdb_y_test.npy')
    if os.path.exists(mit_x_test) and os.path.exists(mit_y_test):
        X_test = np.load(mit_x_test)
        y_test = np.load(mit_y_test)
        print('Loaded MIT-BIH test split')
        return X_test, y_test
    # Fallback to synthetic naming
    gen_x_test = os.path.join(DATA_DIR, 'test.npy')
    gen_y_test = os.path.join(DATA_DIR, 'test_labels.npy')
    if os.path.exists(gen_x_test) and os.path.exists(gen_y_test):
        X_test = np.load(gen_x_test)
        y_test = np.load(gen_y_test)
        print('Loaded synthetic test split')
        return X_test, y_test
    raise FileNotFoundError('No test split found in data/processed/')


def evaluate_keras_model(model_path, X_test, y_test):
    try:
        from tensorflow import keras
    except Exception:
        raise RuntimeError('TensorFlow is required to evaluate Keras model')
    X_test = X_test[..., np.newaxis]
    model = keras.models.load_model(model_path)
    preds = model.predict(X_test).ravel()
    y_pred = (preds > 0.5).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': specificity,
        'f1': f1,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def main():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    X_test, y_test = load_test_split()

    model_path = os.path.join(MODEL_DIR, 'baseline.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    print('Evaluating model at', model_path)
    metrics = evaluate_keras_model(model_path, X_test, y_test)

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    report_json = os.path.join(EXPERIMENT_DIR, f'report_{ts}.json')
    report_txt = os.path.join(EXPERIMENT_DIR, f'report_{ts}.txt')

    with open(report_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(report_txt, 'w') as f:
        f.write('Beat2Bit evaluation report\n')
        for k, v in metrics.items():
            f.write(f'{k}: {v}\n')

    print('Saved evaluation report to', report_json)

if __name__ == '__main__':
    main()
