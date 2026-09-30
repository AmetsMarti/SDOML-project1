import pickle
from datasets import load_dataset, Audio
from sdoml_task1.config import PROCESSED_DATA_DIR

def load_features():
    """Load the MFCC features, computing and caching them if needed.

    The first call downloads the dataset, extracts the features and saves
    them to ``data/processed/features.pkl``. Later calls simply read that file.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Feature matrices of shape ``(n_samples, 26)`` and digit labels.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = load_features()
    """
    cache_path = PROCESSED_DATA_DIR / "features.pkl"

    try:
        with open(cache_path, "rb") as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
        print(f"Features loaded from {cache_path}")

    except FileNotFoundError:
        ds = load_dataset("gilkeyio/AudioMNIST")
        ds = ds.cast_column("audio", Audio(decode=False))

        X_train, y_train = build_feature_dataset(ds["train"])
        X_test, y_test = build_feature_dataset(ds["test"])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)
        print(f"Features saved to {cache_path}")

    return X_train, y_train, X_test, y_test