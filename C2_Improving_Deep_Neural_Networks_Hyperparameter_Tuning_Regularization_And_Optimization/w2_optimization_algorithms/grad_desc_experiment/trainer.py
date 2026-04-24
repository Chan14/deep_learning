"""
trainer.py — Training loop and batch iterators for the three GD methods.

The three methods (Batch GD, True SGD, Mini-batch SGD) differ in exactly
one thing: how data is chunked into gradient-estimation batches. We
express that in code by parameterizing the training loop with a
batch_iterator. The loop itself is method-agnostic.

Batch iterator contract:
    A batch_iterator is a callable (X, y, rng) -> generator of (X_batch, y_batch).
    Each call to the iterator produces one epoch's worth of batches.
    The iterator is responsible for reshuffling (if shuffling is meaningful).

Invariants (see module docstring in train() for more):
    - train() sees only (X_batch, y_batch) pairs from the iterator. It
      does not know or care which method is running.
    - Shuffling is the iterator's responsibility, not train()'s.
    - Batch size is inferred from X_batch.shape[0] at each step and passed
      to optimizer_step as `m`, so the L2 scaling is correct per method.
"""

from typing import Callable, Iterator, Tuple
import numpy as np

BatchIterator = Callable[
    [np.ndarray, np.ndarray, np.random.Generator],
    Iterator[Tuple[np.ndarray, np.ndarray]],
]


# ---------- Batch iterators ----------


def batch_gd_iterator(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """
    Full-batch gradient descent. Yields the entire dataset once per epoch.

    No shuffling — it's meaningless for full-batch (gradient is identical
    regardless of sample order). We accept `rng` to match the common
    iterator signature, but don't use it.

    Yields:
        Exactly one (X, y) pair, shape (m, n_features) and (m, 1).
    """
    yield X, y


def true_sgd_iterator(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """
    True stochastic gradient descent. Yields m single-example batches per epoch.

    Shuffles at the start of each epoch (i.e. each time this is called).

    Yields:
        m pairs of shape (1, n_features) and (1, 1).

    Invariant: the yielded batches have a LEADING DIMENSION of 1, not 0.
    That is, X_batch.shape == (1, n_features), NOT (n_features,).
    This is essential because the model's forward pass expects 2D input.
    Use slicing (X[i:i+1]) rather than indexing (X[i]) to preserve the dim.
    """
    m = X.shape[0]
    indices = rng.permutation(m)
    for i in indices:
        yield X[i : i + 1], y[i : i + 1]  # slicing preserves leading dim


def minibatch_iterator(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, batch_size: int = 64
):
    """
    Mini-batch stochastic gradient descent. Yields ceil(m / batch_size) batches per epoch.

    Shuffles at the start of each epoch. The last batch may be smaller than
    `batch_size` if m isn't divisible by B — we accept that rather than
    dropping the remainder, because on California Housing the remainder is
    meaningful data. The optimizer handles any batch size correctly.

    Args:
        batch_size: Batch size B. Default 64 is a reasonable middle ground.

    Yields:
        ceil(m / B) pairs. All but possibly the last have shape (B, n_features) and (B, 1).
    """
    m = X.shape[0]
    indices = rng.permutation(m)
    for start in range(0, m, batch_size):
        idx = indices[start : start + batch_size]
        yield X[idx], y[idx]


# ---------- Helpers for diagnostics ----------


def _grad_norm(grads: dict) -> float:
    """
    L2 norm across all gradient arrays, flattened together:
        sqrt(sum over all entries of all grad arrays of g^2)
    One scalar for the whole network — useful for spotting gradient
    explosion / vanishing at a glance.
    """
    grad_stack = np.concatenate([g.ravel() for g in grads.values()])
    return np.linalg.norm(grad_stack)


def _param_norm(params: dict) -> float:
    """
    L2 norm across all WEIGHT arrays (not biases). Used to watch
    regularization pulling weights toward zero.
    Matches the convention from losses.py — biases are never part of
    the L2 story.
    """
    p_stack = np.concatenate(
        [v.ravel() for k, v in params.items() if k.startswith("W")]
    )
    return np.linalg.norm(p_stack)


def _snapshot(params: dict) -> dict:
    """
    Deep-copy the params dict. Needed for best-params snapshot because
    params is mutated in place by optimizer_step — without copying, our
    'best' would be a reference that keeps changing.
    """
    return {k: v.copy() for k, v in params.items()}


# ---------- The training loop ----------


def train(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    batch_iterator: BatchIterator,
    learning_rate: float,
    lambda_reg: float,
    max_epochs: int = 150,
    patience: int = 20,
    rng: np.random.Generator = None,
    log_per_update_epochs: tuple = (),
    verbose: bool = False,
) -> dict:
    """
    Train the model. Method (batch GD / true SGD / mini-batch) is entirely
    determined by the batch_iterator passed in.

    Args:
        params:                Initial parameter dict from init_parameters.
                               MUTATED IN PLACE during training (via optimizer_step).
                               Best-dev-loss snapshot is returned separately.
        X_train, y_train:      Training data. Shapes (m_train, 8) and (m_train, 1).
        X_dev,   y_dev:        Dev data. Shapes (m_dev, 8) and (m_dev, 1).
        batch_iterator:        One of batch_gd_iterator, true_sgd_iterator, minibatch_iterator.
                               (For minibatch, wrap in functools.partial to bind batch_size.)
        learning_rate:         Alpha.
        lambda_reg:            L2 strength. Pass 0.0 explicitly if you want no regularization.
        max_epochs:            Hard cap on epochs. Default 150.
        patience:              Stop if dev loss hasn't improved for this many epochs. Default 20.
        rng:                   np.random.Generator, used by the iterator for shuffling.
                               If None, a default-seeded one is created (but pass it
                               explicitly for reproducibility across runs).
        log_per_update_epochs: Tuple of epoch indices (0-indexed) on which to log
                               per-update loss. E.g. (0, 75, 149) for first/middle/last.
                               Default () = no per-update logging.
        verbose:               If True, print a summary each epoch.

    Returns:
        A dict:
            'best_params'       : dict — params from the epoch with lowest dev data loss
            'best_epoch'        : int  — the epoch at which best_params was captured
            'best_dev_loss'     : float
            'stopped_early'     : bool — True if we stopped before max_epochs
            'final_epoch'       : int  — the last epoch that actually ran (0-indexed)
            'history'           : dict with keys:
                'train_data_loss'   : list[float]
                'train_reg_loss'    : list[float]
                'train_total_loss'  : list[float]
                'dev_data_loss'     : list[float]
                'grad_norm'         : list[float]  — final-batch grad norm each epoch
                'param_norm'        : list[float]  — weight norm after each epoch
            'per_update_history': dict[int, list[float]] — maps epoch_idx -> list of
                                  per-update data losses for that epoch. Keys present
                                  only for epochs in log_per_update_epochs.
    """
    from model import forward, backward
    from losses import compute_total_loss, compute_data_loss
    from optimizer import optimizer_step

    if rng is None:
        rng = np.random.default_rng(0)

    history = {
        "train_data_loss": [],
        "train_reg_loss": [],
        "train_total_loss": [],
        "dev_data_loss": [],
        "grad_norm": [],
        "param_norm": [],
    }
    per_update_history = {}

    best_dev_loss = float("inf")
    best_params = _snapshot(params)
    best_epoch = 0
    epochs_since_improvement = 0
    stopped_early = False
    final_epoch = 0

    for epoch in range(max_epochs):
        # Is this an epoch where we log per-update losses?
        log_updates_this_epoch = epoch in log_per_update_epochs
        per_update_losses = [] if log_updates_this_epoch else None

        # ---- Run one epoch's worth of updates ----
        last_grads = None  # for grad_norm logging at end of epoch
        for X_batch, y_batch in batch_iterator(X_train, y_train, rng):
            y_hat, cache = forward(X_batch, params)
            grads = backward(y_hat, y_batch, params, cache)
            if log_updates_this_epoch:
                per_update_losses.append(compute_data_loss(y_hat, y_batch))
            optimizer_step(params, grads, learning_rate, lambda_reg, m=X_batch.shape[0])
            last_grads = grads

        # ---- End-of-epoch diagnostics ----
        # Train loss on the FULL training set with current params (Option A, honest).
        y_hat_train, _ = forward(X_train, params)
        losses = compute_total_loss(y_hat_train, y_train, params, lambda_reg)
        history["train_data_loss"].append(losses["data"])
        history["train_reg_loss"].append(losses["reg"])
        history["train_total_loss"].append(losses["total"])

        # Dev loss — data only, no reg term.
        y_hat_dev, _ = forward(X_dev, params)
        dev_loss = compute_data_loss(y_hat_dev, y_dev)
        history["dev_data_loss"].append(dev_loss)

        # Diagnostics: gradient norm from the last batch, param norm now.
        history["grad_norm"].append(_grad_norm(last_grads))
        history["param_norm"].append(_param_norm(params))

        # Per-update history bookkeeping
        if log_updates_this_epoch:
            per_update_history[epoch] = per_update_losses

        if verbose:
            print(
                f"epoch {epoch:3d}  train={losses['total']:.4f}  "
                f"dev={dev_loss:.4f}  grad_norm={history['grad_norm'][-1]:.3f}"
            )

        final_epoch = epoch

        # ---- Early stopping logic ----
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_params = _snapshot(params)
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            stopped_early = True
            break

    return {
        "best_params": best_params,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "stopped_early": stopped_early,
        "final_epoch": final_epoch,
        "history": history,
        "per_update_history": per_update_history,
    }


# ---------- Smoke tests ----------


def _iterator_smoke_test():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 8))
    y = rng.normal(size=(100, 1))

    # ----- batch_gd_iterator -----
    batches = list(batch_gd_iterator(X, y, rng))
    assert (
        len(batches) == 1
    ), f"batch GD should yield exactly 1 batch, got {len(batches)}"
    Xb, yb = batches[0]
    assert Xb.shape == (100, 8) and yb.shape == (100, 1)
    print(f"batch_gd_iterator:  1 batch of shape {Xb.shape}  OK")

    # ----- true_sgd_iterator -----
    batches = list(true_sgd_iterator(X, y, rng))
    assert len(batches) == 100
    for Xb, yb in batches:
        assert Xb.shape == (1, 8) and yb.shape == (1, 1)
    X_concat = np.concatenate([Xb for Xb, _ in batches], axis=0)
    assert np.allclose(np.sort(X_concat[:, 0]), np.sort(X[:, 0]))
    print(f"true_sgd_iterator:  100 batches of shape (1, 8)  OK")

    # ----- minibatch_iterator -----
    batches = list(minibatch_iterator(X, y, rng, batch_size=32))
    assert len(batches) == 4
    sizes = [Xb.shape[0] for Xb, _ in batches]
    assert sizes == [32, 32, 32, 4]
    X_concat = np.concatenate([Xb for Xb, _ in batches], axis=0)
    assert np.allclose(np.sort(X_concat[:, 0]), np.sort(X[:, 0]))
    print(f"minibatch_iterator: 4 batches with sizes {sizes}  OK")

    # ----- Reshuffling & determinism -----
    rng_a = np.random.default_rng(7)
    first_call = list(true_sgd_iterator(X, y, rng_a))[0][0]
    second_call = list(true_sgd_iterator(X, y, rng_a))[0][0]
    assert not np.array_equal(first_call, second_call), "rng not advancing!"
    print("Reshuffling check:   consecutive calls produce different orders  OK")

    rep_a = list(true_sgd_iterator(X, y, np.random.default_rng(42)))[0][0]
    rep_b = list(true_sgd_iterator(X, y, np.random.default_rng(42)))[0][0]
    assert np.array_equal(rep_a, rep_b)
    print("Determinism check:   same seed produces identical orders  OK")


def _make_synthetic_data(n_train=500, n_dev=100, n_features=8, noise=0.1, seed=0):
    """
    A known-learnable regression problem: y = X @ w_true + noise.
    With n_features=8 matching our California Housing setup, the 3-layer MLP
    should easily fit this — if it doesn't, trainer is broken.
    """
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=(n_features, 1))

    X_train = rng.normal(size=(n_train, n_features))
    y_train = X_train @ w_true + noise * rng.normal(size=(n_train, 1))

    X_dev = rng.normal(size=(n_dev, n_features))
    y_dev = X_dev @ w_true + noise * rng.normal(size=(n_dev, 1))

    return X_train, y_train, X_dev, y_dev


def _training_dynamics_smoke_test():
    """
    The real test. Runs all three methods on synthetic data and asserts
    that training actually works the way we expect.
    """
    from functools import partial
    from model import init_parameters

    X_train, y_train, X_dev, y_dev = _make_synthetic_data()

    # A config that should converge for all three methods on easy synthetic data.
    # We deliberately use slightly different learning rates per method — this
    # mirrors the real experiment where SGD needs a smaller alpha.
    configs = [
        ("batch_gd", batch_gd_iterator, 0.05),
        ("minibatch", partial(minibatch_iterator, batch_size=32), 0.02),
        ("true_sgd", true_sgd_iterator, 0.005),
    ]

    results = {}
    for name, iterator, lr in configs:
        params = init_parameters([8, 32, 16, 1], seed=1)
        result = train(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_dev=X_dev,
            y_dev=y_dev,
            batch_iterator=iterator,
            learning_rate=lr,
            lambda_reg=0.0,
            max_epochs=60,
            patience=100,  # effectively no early stop — we want full runs here
            rng=np.random.default_rng(42),
        )
        results[name] = result

        # ----- Assertion 1: loss actually decreased -----
        start_loss = result["history"]["train_data_loss"][0]
        end_loss = result["history"]["train_data_loss"][-1]
        assert end_loss < start_loss / 2, (
            f"{name}: training didn't reduce loss enough. "
            f"start={start_loss:.4f}, end={end_loss:.4f}"
        )

        # ----- Assertion 2: best_params gives <= current params loss -----
        from model import forward
        from losses import compute_data_loss

        y_hat_best, _ = forward(X_dev, result["best_params"])
        best_loss = compute_data_loss(y_hat_best, y_dev)
        assert best_loss <= result["history"]["dev_data_loss"][-1] + 1e-8, (
            f"{name}: best_params gives worse dev loss than final params — "
            f"snapshot bug. best={best_loss:.6f}, final={result['history']['dev_data_loss'][-1]:.6f}"
        )

        print(
            f"  {name:10s}: start_loss={start_loss:.4f}  end_loss={end_loss:.4f}  "
            f"best_epoch={result['best_epoch']:3d}  best_dev={result['best_dev_loss']:.4f}"
        )

    # ----- Assertion 3: all three methods reach comparable loss -----
    # "Comparable" = within a factor of 5 of each other. A method-specific
    # bug (e.g. wrong shape in SGD path) would show as one method being far worse.
    final_dev_losses = [r["best_dev_loss"] for r in results.values()]
    max_loss = max(final_dev_losses)
    min_loss = min(final_dev_losses)
    assert max_loss / min_loss < 5.0, (
        f"Methods reached very different loss — possible method-specific bug. "
        f"Losses: {dict(zip(results.keys(), final_dev_losses))}"
    )
    print(
        f"Training-decrease & method-parity: OK  "
        f"(dev losses in [{min_loss:.4f}, {max_loss:.4f}])"
    )

    # ----- Assertion 4: early stopping fires -----
    params = init_parameters([8, 32, 16, 1], seed=1)
    result_es = train(
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        batch_iterator=batch_gd_iterator,
        learning_rate=0.05,
        lambda_reg=0.0,
        max_epochs=500,
        patience=5,  # aggressive; should fire well before 500
        rng=np.random.default_rng(42),
    )
    assert result_es["stopped_early"], (
        f"Early stopping did not fire with patience=5, max_epochs=500. "
        f"Final epoch: {result_es['final_epoch']}"
    )
    assert (
        result_es["final_epoch"] < 400
    ), f"Early stopping fired too late (epoch {result_es['final_epoch']})"
    print(f"Early stopping:            OK  (fired at epoch {result_es['final_epoch']})")

    # ----- Assertion 5: per-update logging behaves -----
    params = init_parameters([8, 32, 16, 1], seed=1)
    result_pu = train(
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        batch_iterator=partial(minibatch_iterator, batch_size=32),
        learning_rate=0.02,
        lambda_reg=0.0,
        max_epochs=10,
        patience=100,
        rng=np.random.default_rng(42),
        log_per_update_epochs=(0, 5, 9),
    )
    assert set(result_pu["per_update_history"].keys()) == {0, 5, 9}
    # 500 training examples, batch_size 32 -> ceil(500/32) = 16 updates/epoch.
    for epoch_key in (0, 5, 9):
        assert len(result_pu["per_update_history"][epoch_key]) == 16, (
            f"epoch {epoch_key}: expected 16 per-update losses, got "
            f"{len(result_pu['per_update_history'][epoch_key])}"
        )
    print("Per-update history:        OK  (keys {0, 5, 9}, each with 16 entries)")

    # ----- Assertion 6: determinism -----
    def _run_once():
        p = init_parameters([8, 32, 16, 1], seed=1)
        return train(
            params=p,
            X_train=X_train,
            y_train=y_train,
            X_dev=X_dev,
            y_dev=y_dev,
            batch_iterator=partial(minibatch_iterator, batch_size=32),
            learning_rate=0.02,
            lambda_reg=0.01,
            max_epochs=10,
            patience=100,
            rng=np.random.default_rng(42),
        )

    r1 = _run_once()
    r2 = _run_once()
    # Same seed, same init -> trajectories should be byte-identical.
    assert (
        r1["history"]["dev_data_loss"] == r2["history"]["dev_data_loss"]
    ), "Two runs with same seed produced different trajectories — non-determinism!"
    print("Determinism:               OK  (two seeded runs match exactly)")


if __name__ == "__main__":
    print("=== Stage 1: iterators ===")
    _iterator_smoke_test()
    print("\n=== Stage 2: training dynamics ===")
    _training_dynamics_smoke_test()
    print("\nAll trainer smoke tests passed.")
