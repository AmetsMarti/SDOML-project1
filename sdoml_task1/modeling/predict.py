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
import pickle
import re
import warnings

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from sdoml_task1.config import MODEL_DIR, PROJECT_DIR
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
        UUID of the run to load from ``runs/``. When omitted,
        ``models/model.pt`` is used if it exists, otherwise the most recently
        written checkpoint under ``runs/``.
    device : str or torch.device, optional
        Device to run the forward pass on. Defaults to CPU so the loader also
        works on machines without CUDA.

    Attributes
    ----------
    is_trained : bool
        ``False`` when no checkpoint was found and the network was built with
        random weights. The UI uses it to warn that predictions are meaningless.
    checkpoint : pathlib.Path or None
        File the weights were read from, or ``None`` when a ``model`` was
        passed in or no checkpoint exists.

    Raises
    ------
    FileNotFoundError
        If an explicit ``run_id`` does not exist. Without ``run_id`` a missing
        checkpoint is not an error: the loader falls back to random weights.
    ValueError
        If the checkpoint does not describe a network :class:`Net` can rebuild.
    """

    def __init__(
        self,
        model: Net | None = None,
        run_id: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint: Path | None = None
        self.is_trained = model is not None

        if model is None:
            try:
                checkpoint = self._resolve_checkpoint(run_id)
                state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model = self._build_model(state_dict, checkpoint)
            except FileNotFoundError:
                # Without an explicit run_id a missing checkpoint is not fatal:
                # the interface must still start, so we fall back to Net() below.
                if run_id is not None:
                    raise
                checkpoint = None
            except (OSError, EOFError, ValueError, RuntimeError, pickle.UnpicklingError) as error:
                warnings.warn(f"{checkpoint} is unusable ({error}); starting untrained.")
                checkpoint = None

            if checkpoint is None:
                model = Net()  # random weights: nothing has been trained yet
            else:
                self.checkpoint = checkpoint
                self.is_trained = True

        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def _resolve_checkpoint(cls, run_id: str | None = None) -> Path:
        """Pick the checkpoint to load: ``run_id``, the published model, then the newest run.

        Parameters
        ----------
        run_id : str, optional
            UUID of a run under ``runs/``. When given, no fallback is used.

        Returns
        -------
        pathlib.Path
            Path to an existing ``model.pt``.

        Raises
        ------
        FileNotFoundError
            If neither the published model nor any run holds a checkpoint.
        """
        if run_id is None:
            published = MODEL_DIR / "model.pt"
            if published.is_file():
                return published
            try:
                return cls._latest_checkpoint()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No checkpoint found: expected {published} or a model.pt under {RUNS_DIR}."
                    " Train a model first (notebooks/01_audio_training.ipynb) and save it with"
                    " sdoml_task1.modeling.train.save_run()."
                ) from None
        return cls._latest_checkpoint(run_id)

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

    @staticmethod
    def _build_model(state_dict: dict[str, torch.Tensor], checkpoint: Path) -> Net:
        """Rebuild a :class:`Net` from the shapes stored in a checkpoint.

        The topology (input size, hidden layers, number of classes) is read
        from the weight matrices, and legacy layer names such as ``fc1``/``fc3``
        are mapped onto the current ``lst.*`` names, so checkpoints written
        before the model was refactored keep loading.

        Parameters
        ----------
        state_dict : dict
            Weights as read from a checkpoint file.
        checkpoint : pathlib.Path
            Only used to point at the offending file in error messages.

        Returns
        -------
        Net
            A network matching the checkpoint, weights already loaded.

        Raises
        ------
        ValueError
            If the tensors do not describe a stack of ``Net`` linear layers.
        """

        def layer_index(key: str) -> int:
            digits = re.findall(r"\d+", key)
            return int(digits[-1]) if digits else -1

        weight_keys = sorted((k for k, v in state_dict.items() if v.ndim == 2), key=layer_index)
        bias_keys = sorted((k for k, v in state_dict.items() if v.ndim == 1), key=layer_index)
        shapes = [tuple(state_dict[k].shape) for k in weight_keys]

        if len(shapes) < 2:
            raise ValueError(
                f"{checkpoint} does not describe a Net: found {len(shapes)} linear layer(s),"
                " at least 2 are required (one hidden layer plus the output layer)."
            )
        if len(bias_keys) != len(weight_keys):
            raise ValueError(
                f"{checkpoint} mixes biased and bias-free layers:"
                f" {len(weight_keys)} weight tensors vs {len(bias_keys)} bias tensors."
            )
        for i, (out_dim, in_dim) in enumerate(shapes):
            if i and in_dim != shapes[i - 1][0]:
                raise ValueError(
                    f"{checkpoint} has inconsistent layer shapes {shapes}: layer {i} takes"
                    f" {in_dim} inputs but layer {i - 1} produces {shapes[i - 1][0]}."
                )

        input_dim = shapes[0][1]
        hidden_sizes = tuple(out_dim for out_dim, _ in shapes[:-1])
        num_classes = shapes[-1][0]

        model = Net(input_dim=input_dim, hidden_sizes=hidden_sizes, num_classes=num_classes)
        canonical = {f"lst.{i}.weight": state_dict[k] for i, k in enumerate(weight_keys)}
        canonical.update({f"lst.{i}.bias": state_dict[k] for i, k in enumerate(bias_keys)})
        model.load_state_dict(canonical)
        return model

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

        Raises
        ------
        RuntimeError
            If no trained model is loaded: with random weights the numbers
            would look like predictions without meaning anything.
        """
        if not self.is_trained:
            raise RuntimeError(
                "No trained model is loaded: train a model and save it with"
                " sdoml_task1.modeling.train.save_run() first."
            )
        audio_array, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        features = extract_features(audio_array, sample_rate)
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return self._predict(X)
