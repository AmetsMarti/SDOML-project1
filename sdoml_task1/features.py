"""Audio feature extraction utilities."""

import io

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from sdoml_task1.config import N_MFCC

def extract_features(audio_array, sample_rate):
    """Extract an MFCC feature vector from an audio signal.

    Parameters
    ----------
    audio_array : np.ndarray
        The raw audio waveform.
    sample_rate : int
        Sample rate of the audio, in Hz.

    Returns
    -------
    np.ndarray
        Feature vector: mean and standard deviation of the MFCCs.
    """
    mfcc = librosa.feature.mfcc(y=audio_array.astype(np.float32), sr=sample_rate, n_mfcc=N_MFCC)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    return np.concatenate([mean, std])

def decode_audio(example):
    """Decode a HuggingFace audio example into a waveform.

    Parameters
    ----------
    example : dict
        A dataset example with an "audio" field (bytes or path).

    Returns
    -------
    data : np.ndarray
        The decoded audio waveform.
    sample_rate : int
        Sample rate of the audio, in Hz.
    """
    audio_field = example["audio"]
    if audio_field["bytes"] is not None:
        data, sample_rate = sf.read(io.BytesIO(audio_field["bytes"]))
    else:
        data, sample_rate = sf.read(audio_field["path"])
    return data, sample_rate

def build_feature_dataset(hf_split):
    """Build a feature matrix and label vector from a dataset split.

    Parameters
    ----------
    hf_split : datasets.Dataset
        A HuggingFace dataset split with "audio" and "digit" fields.

    Returns
    -------
    X : np.ndarray
        Feature vectors, shape (n_samples, n_features).
    y : np.ndarray
        Digit labels, shape (n_samples,).
    """
    X, y = [], []
    for s in tqdm(hf_split, desc="Extract features"):
        audio_array, sr = decode_audio(s)
        features = extract_features(audio_array, sr)
        X.append(features)
        y.append(s["digit"])
    return np.array(X), np.array(y)