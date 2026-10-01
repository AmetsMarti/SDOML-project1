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
import warnings

import numpy as np
import torch

from sdoml_task1.config import MODEL_DIR
from sdoml_task1.modeling.model import Net

class ModelLoader:
    """Inference wrapper around a trained :class:`Net`.

    Parameters
    ----------
    model : Net, optional
        An already-built network. When omitted, ``models/model.pt`` is loaded.
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
        If ``models/model.pt`` does not exist. The loader falls back to random
        weights when no model has been trained yet.
    ValueError
        If the checkpoint does not contain model weights and configuration.
    """

    def __init__(
        self,
        model: Net | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint: Path | None = None
        self.is_trained = model is not None

        if model is None:
            try:
                checkpoint = self._resolve_checkpoint()
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict) or not {"state_dict", "config"} <= payload.keys():
                    raise ValueError(f"{checkpoint} is not a latest-model checkpoint")

                config = payload["config"]
                model = Net(
                    input_dim=config["input_dim"],
                    hidden_sizes=tuple(config["hidden_sizes"]),
                    num_classes=config.get("num_classes", 10),
                    activation_function=config["activation_function"],
                    regularization=config["regularization"],
                    reg_param=config["reg_param"],
                )
                model.load_state_dict(payload["state_dict"])
            except FileNotFoundError:
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

    @staticmethod
    def _resolve_checkpoint() -> Path:
        """Return the latest model checkpoint.

        Returns
        -------
        pathlib.Path
            Path to ``models/model.pt``.

        Raises
        ------
        FileNotFoundError
            If the latest model has not been saved yet.
        """
        checkpoint = MODEL_DIR / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"No latest model found at {checkpoint}. Train a model first and save it with"
                " sdoml_task1.modeling.train.save_run()."
            )
        return checkpoint

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
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
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
        from sdoml_task1.features import extract_features
        import librosa

        audio_array, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        features = extract_features(audio_array, sample_rate)
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return self._predict(X)
