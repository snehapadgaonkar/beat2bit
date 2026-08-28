"""Pytest root configuration for Beat2Bit.

Ensures the project root is importable so ``src`` package imports resolve
regardless of the working directory pytest is invoked from.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
