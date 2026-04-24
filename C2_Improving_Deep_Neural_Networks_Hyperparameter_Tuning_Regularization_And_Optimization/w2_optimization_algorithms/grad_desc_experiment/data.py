import numpy as np
from sklearn.datasets import fetch_california_housing


def load_and_prepare_data(
    dev_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Load the California Housing dataset, split into train/dev/test,
    and normalize both features and target using training-set statistics.

    Invariants:
        1. Splits are disjoint — no example appears in more than one split.
        2. Normalization statistics (mean, std) are computed from the training
           split ONLY and applied to dev and test. This prevents information
           leakage from evaluation data into training.
        3. y has shape (m, 1), not (m,), to prevent silent broadcasting bugs
           downstream in the neural network code.
        4. All returned arrays are finite — no NaN, no Inf.
        5. Splits are deterministic given `seed`.

    Args:
        dev_frac:  Fraction of the dataset to use for the dev set.
        test_frac: Fraction of the dataset to use for the test set.
                   The training fraction is implicitly 1 - dev_frac - test_frac.
        seed:      Random seed controlling the shuffle used for splitting.

    Returns:
        A dict with the following keys:
            X_train, X_dev, X_test : feature arrays of shape (m_split, 8)
            y_train, y_dev, y_test : target arrays of shape (m_split, 1),
                                     normalized to zero mean / unit variance
                                     using training-set statistics
            x_mean, x_std          : per-feature statistics from the training
                                     split, shape (8,). Exposed for debugging
                                     and for inverse-transforming features.
            y_mean, y_std          : target statistics from the training split,
                                     scalars. Needed to inverse-transform
                                     predictions back to original units
                                     (e.g., reporting MSE in dollars).
    """

    # Load the dataset
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target.reshape(-1, 1)

    # Sample a reprodicible shuffle of indices
    m = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(m)

    # Compute split sizes
    dev_size = int(m * dev_frac)
    test_size = int(m * test_frac)
    train_size = m - dev_size - test_size

    # Slice indices
    train_idx = indices[:train_size]
    dev_idx = indices[train_size : train_size + dev_size]
    test_idx = indices[train_size + dev_size :]

    # Extract splits
    X_train, y_train = X[train_idx], y[train_idx]
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Normalize Features (X)
    # INVARIANT: Compute mean/std from X_train ONLY
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)

    # Fail fast on bad data - ( ok for learning, in prod code, we would add epsilon)-
    # eps = 1e-8
    # x_std = np.where(x_std == 0, eps, x_std)
    assert (x_std > 0).all(), "Constant feature detected in training data"

    X_train = (X_train - x_mean) / x_std
    X_dev = (X_dev - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    # Normalize Target (y) (we want scalar y_mean and y_std for clean predictions
    # INVARIANT: Compute mean/std from y_train ONLY
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    assert y_std > 0, "Constant target detected in training data"

    y_train = (y_train - y_mean) / y_std
    y_dev = (y_dev - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Trust-But-Verify (The NaN/Inf Assertions)
    datasets = {
        "X_train": X_train,
        "y_train": y_train,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "X_test": X_test,
        "y_test": y_test,
    }

    for name, data in datasets.items():
        assert not np.isnan(data).any(), f"NaN detected in {name}"
        assert not np.isinf(data).any(), f"Inf detected in {name}"

    # Return a dictionary of values because dictionaries are self documenting.
    return {
        **datasets,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


if __name__ == "__main__":
    data = load_and_prepare_data()
    # Quick sanity checks
    print(f"Data loaded. X_train shape: {data['X_train'].shape}")
    assert abs(data["X_train"].mean()) < 0.01, "Train features not centered"
    assert abs(data["X_train"].std() - 1.0) < 0.01, "Train features not unit variance"
    # Dev/test should be CLOSE to 0 mean, but not exactly
    assert abs(data["X_dev"].mean()) < 0.1, "Dev features unexpectedly far from 0"
    assert abs(data["X_test"].mean()) < 0.1, "Test features unexpectedly far from 0"
