"""
Post-training quantization script for Beat2Bit.

Converts a trained Keras model (models/saved/baseline.h5) to a fully integer int8 TFLite
using a representative dataset sampled from data/processed (MIT-BIH preferred).

Usage:
  python -m src.training.quantize --model models/saved/baseline.h5 --out models/saved/model_int8.tflite --rep-samples 100

Requirements: tensorflow (for conversion) and numpy
"""

import os
import argparse
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments')


def load_representative_samples(max_samples=100):
    """Load representative samples from MIT-BIH processed splits if available,
    otherwise fall back to synthetic naming. Returns a numpy array of shape (N, L).
    """
    mit_x_train = os.path.join(DATA_DIR, 'mitdb_X_train.npy')
    gen_x_train = os.path.join(DATA_DIR, 'train.npy')
    if os.path.exists(mit_x_train):
        arr = np.load(mit_x_train)
        print(f"Using MIT-BIH training samples for representative dataset ({arr.shape[0]} available)")
    elif os.path.exists(gen_x_train):
        arr = np.load(gen_x_train)
        print(f"Using synthetic training samples for representative dataset ({arr.shape[0]} available)")
    else:
        raise FileNotFoundError('No processed training samples found in data/processed/; run download_and_preprocess or generate_synthetic')

    if arr.ndim == 3:  # in case channel already present
        arr = arr.squeeze(-1)
    n = min(max_samples, arr.shape[0])
    return arr[:n]


def representative_gen(samples):
    # samples: numpy array (N, L)
    def gen():
        for s in samples:
            inp = np.array(s, dtype=np.float32)
            # reshape to [1, L, 1]
            inp = inp.reshape(1, -1, 1)
            yield [inp]
    return gen


def convert_to_int8(keras_model_path, out_path, rep_samples=100):
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError('TensorFlow is required for conversion: pip install tensorflow')

    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f'Keras model not found: {keras_model_path}')

    print('Loading model:', keras_model_path)
    model = tf.keras.models.load_model(keras_model_path)

    print('Loading representative samples...')
    samples = load_representative_samples(max_samples=rep_samples)
    rep_gen = representative_gen(samples)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen()
    # Enforce integer only ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print('Converting model to int8 TFLite (this may take a moment)')
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print('Wrote int8 TFLite model to', out_path)

    # Size report
    k_size = os.path.getsize(keras_model_path)
    t_size = os.path.getsize(out_path)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    report_path = os.path.join(EXPERIMENTS_DIR, 'quantize_report.txt')
    with open(report_path, 'a') as r:
        r.write(f'Model: {keras_model_path}\n')
        r.write(f'Output: {out_path}\n')
        r.write(f'Keras size: {k_size} bytes\n')
        r.write(f'Int8 TFLite size: {t_size} bytes\n')
        r.write('\n')
    print('Quantization report appended to', report_path)
    return out_path, k_size, t_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join(MODEL_DIR, 'baseline.h5'), help='Path to Keras model (.h5)')
    parser.add_argument('--out', default=os.path.join(MODEL_DIR, 'model_int8.tflite'), help='Output TFLite path')
    parser.add_argument('--rep-samples', type=int, default=100, help='Number of representative samples to use')
    args = parser.parse_args()

    out_path, k_size, t_size = convert_to_int8(args.model, args.out, rep_samples=args.rep_samples)
    print('Done. Sizes: keras=%d, int8_tflite=%d' % (k_size, t_size))


if __name__ == '__main__':
    main()
