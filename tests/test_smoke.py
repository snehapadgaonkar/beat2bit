import os
import sys

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training import train
from src.inference import infer


def test_train_runs_without_error():
    # The training script is a lightweight template that should run and print
    train.main()


def test_infer_runs_on_sample():
    # Should execute without throwing
    infer.run_sample('models/saved/model.tflite', [0.0]*250)
