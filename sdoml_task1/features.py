
import io

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from sdoml_task1.config import N_MFCC


def extract_features(audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract an MFCC feature vector from an audio signal.

    Computes 13 Mel-frequency cepstral coefficients and returns the
    concatenation of their mean and standard deviation across time frames,
    yielding a 26-dimensional feature vector.

    Parameters
    ----------
    audio_array : np.ndarray
        The raw audio waveform as a 1-D numpy array.
    sample_rate : int
        Sample rate of the audio, in Hz.

    Returns
    -------
    np.ndarray
        Feature vector of shape ``(26,)`` containing MFCC mean and std.
    """
    mfcc = librosa.feature.mfcc(y=audio_array.astype(np.float32), sr=sample_rate, n_mfcc=N_MFCC)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    return np.concatenate([mean, std])


def decode_audio(example: dict) -> tuple[np.ndarray, int]:
    """Decode a HuggingFace audio example into a waveform.

    Handles both in-memory audio bytes and file path references.

    Parameters
    ----------
    example : dict
        A dataset example with an ``"audio"`` field containing either
        ``"bytes"`` (raw audio data) or ``"path"`` (file path).

    Returns
    -------
    data : np.ndarray
        The decoded audio waveform as a 1-D numpy array.
    sample_rate : int
        Sample rate of the audio, in Hz.

    """
    audio_field = example["audio"]
    if audio_field["bytes"] is not None:
        data, sample_rate = sf.read(io.BytesIO(audio_field["bytes"]))
    else:
        data, sample_rate = sf.read(audio_field["path"])
    return data, sample_rate


def build_feature_dataset(hf_split) -> tuple[np.ndarray, np.ndarray]:
    """Build a feature matrix and label vector from a dataset split.

    Iterates over all samples in a HuggingFace dataset split, decodes
    the audio, and extracts MFCC features for each sample.

    Parameters
    ----------
    hf_split : datasets.Dataset
        A HuggingFace dataset split with ``"audio"`` and ``"digit"`` fields.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, 26)``.
    y : np.ndarray
        Digit labels of shape ``(n_samples,)``.
    """
    X, y = [], []
    for s in tqdm(hf_split, desc="Extract features"):
        audio_array, sr = decode_audio(s)
        features = extract_features(audio_array, sr)
        X.append(features)
        y.append(s["digit"])
    return np.array(X), np.array(y)