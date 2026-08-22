"""
Simple inference harness for Beat2Bit.
Provides run_sample(model_path, sample_input) which attempts to run a TFLite interpreter if available,
otherwise returns a deterministic dummy prediction. Meant for smoke tests.
"""

import numpy as np
import os


def run_sample(model_path, sample_input):
    """Run a single sample through a TFLite model if tflite runtime is available.
    model_path: path to .tflite file (may not exist in smoke tests)
    sample_input: 1D iterable of floats (length expected by model)
    Returns: prediction (0 or 1) or None if model can't be executed
    """
    # Try TFLite runtime first
    try:
        from tflite_runtime.interpreter import Interpreter
        have_tflite = True
    except Exception:
        try:
            import tensorflow as tf
            Interpreter = None
            have_tflite = False
        except Exception:
            Interpreter = None
            have_tflite = False

    # If tflite-runtime available and model exists, run it
    if 'Interpreter' in globals() and Interpreter is not None and os.path.exists(model_path):
        try:
            interpreter = Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            inp = np.array(sample_input, dtype=np.float32)
            # reshape based on input_details
            shape = input_details[0]['shape']
            # If shape is [1, L, 1] or [1, L]
            if inp.ndim == 1:
                if len(shape) == 3 and shape[2] == 1:
                    inp = inp.reshape((1, -1, 1))
                else:
                    inp = inp.reshape((1, -1))
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
            # Return binary prediction
            pred = int(np.ravel(out)[0] > 0.5)
            return pred
        except Exception:
            # fall through to dummy
            pass

    # Fallback dummy: simple energy threshold
    arr = np.array(sample_input, dtype=np.float32)
    energy = np.mean(np.abs(arr))
    return 1 if energy > 0.2 else 0
