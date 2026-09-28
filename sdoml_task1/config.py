"""Paths and constants shared by the rest of the package.

Everything is derived from ``PROJECT_DIR``, so the package can be imported
from anywhere (notebook, script, gradio app) without hardcoding paths.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
"""Repository root, i.e. the directory holding ``data/``, ``models/`` and the notebooks."""

DATA_DIR = PROJECT_DIR / "data"
"""Top-level data directory."""

RAW_DATA_DIR = DATA_DIR / "raw"
"""Original dataset as downloaded, never modified in place."""

PROCESSED_DATA_DIR = DATA_DIR / "processed"
"""Feature files written by the notebooks (CSV/Parquet)."""

MODEL_DIR = PROJECT_DIR / "models"
"""Trained checkpoints. Kept out of git."""

N_MFCC = 13
"""Number of MFCCs kept for each clip.

The coefficients are computed per frame and the clip ends up summarized by their
mean and standard deviation, so the input layer of the network has
``2 * N_MFCC = 26`` neurons. Raising it gives a finer spectral description but a
longer feature vector.
"""
