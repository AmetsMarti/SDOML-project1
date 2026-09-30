"""Code to run model inference with trained models.

The pipeline mirrors training: the clip is decoded at its native sample rate,
reduced to the 26-dimensional MFCC vector the network was trained on, and
pushed through the checkpoint as a batch of one.

Classes
-------
ModelLoader
    Loads a trained checkpoint and turns audio paths into class probabilities.

"""

from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from sdoml_task1.config import PROJECT_DIR
from sdoml_task1.features import extract_features
from sdoml_task1.modeling.model import Net

RUNS_DIR = PROJECT_DIR / "runs"
"""Directory holding one subfolder per training run, each with a ``model.pt``."""


class ModelLoader:
    """Inference wrapper around a trained :class:`Net`.

    Parameters
    ----------
    model : Net, optional
        An already-built network. When omitted, the checkpoint resolved by
        ``run_id`` is loaded instead.
    run_id : str, optional
        UUID of the run to load from ``runs/``. Defaults to the most recently
        written checkpoint.
    device : str or torch.device, optional
        Device to run the forward pass on. Defaults to CPU so the loader also
        works on machines without CUDA.


    Raises
    ------
    FileNotFoundError
        If no matching checkpoint exists under ``runs/``.
    """

    def __init__(
        self,
        model: Net | None = None,
        run_id: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if model is None:
            checkpoint = self._latest_checkpoint(run_id)
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = Net(input_dim=state_dict["fc1.weight"].shape[1])
            model.load_state_dict(state_dict)
        self.model = model.to(self.device)
        self.model.eval()

    @staticmethod
    def _latest_checkpoint(run_id: str | None = None) -> Path:
        """Return the ``model.pt`` of a run, by id or most recently written.

        Parameters
        ----------
        run_id : str, optional
            UUID of the run. When ``None``, the newest checkpoint in ``runs/``
            is selected by modification time.

        Returns
        -------
        pathlib.Path
            Path to the checkpoint file.

        Raises
        ------
        FileNotFoundError
            If the run does not exist or ``runs/`` holds no checkpoint.
        """
        if run_id is not None:
            checkpoint = RUNS_DIR / run_id / "model.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"No checkpoint for run {run_id!r}: {checkpoint}")
            return checkpoint

        checkpoints = sorted(RUNS_DIR.glob("*/model.pt"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found under {RUNS_DIR}")
        return checkpoints[-1]

    @torch.no_grad()
    def _predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict class probabilities from a MFCC feature vector.

        Parameters
        ----------
        X : torch.Tensor
            Feature tensor of shape ``(batch_size, 26)``.

        Returns
        -------
        numpy.ndarray
            Probabilities of shape ``(10,)`` for a single sample, or
            ``(batch_size, 10)`` when several samples are passed.
        """
        self.model.eval()
        logits = self.model(X.to(self.device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs[0] if probs.shape[0] == 1 else probs

    def make_prediction(self, audio_path: str) -> np.ndarray:
        """Turn an audio file into digit probabilities.

        - Loads the file at its native sample rate, mixed down to mono
        - Extracts the 26 MFCC features used in training
        - Converts them into a ``(1, 26)`` float32 tensor
        - Returns the probabilities of the 10 digits

        Parameters
        ----------
        audio_path : str
            Path to the audio file, e.g. the filepath a ``gr.Audio`` component
            hands to its callback.

        Returns
        -------
        numpy.ndarray
            Probability vector of shape ``(10,)``.
        """
        audio_array, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        features = extract_features(audio_array, sample_rate)
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return self._predict(X)
