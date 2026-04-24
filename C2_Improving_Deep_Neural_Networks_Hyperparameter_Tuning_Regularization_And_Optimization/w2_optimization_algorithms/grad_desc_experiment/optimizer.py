"""
optimizer.py — Vanilla gradient descent step with L2 regularization.

Design notes:
    - Mutates params in place. Returns None. Standard Python in-place idiom.
    - L2 gradient (lambda_reg / m) * W is added to WEIGHT gradients only.
      Biases never receive L2. This matches the loss-side invariant from
      losses.py.
    - The caller MUST pass the m that was used to compute `grads`. This is
      not optional — batch size mismatch between data gradient and L2
      gradient would silently make regularization strength depend on batch
      size, contaminating the Batch/SGD/Mini-batch comparison.
    - `lambda_reg` has no default. Making it required forces the caller to
      be intentional about regularization. A forgotten `lambda_reg` argument
      is a bug; defaulting to 0.0 would let that bug pass silently.
    - Vanilla GD only. No momentum, no Adam. Those come later.

Verified: Smoke test passed.
"""

import numpy as np


def optimizer_step(
    params: dict,
    grads: dict,
    learning_rate: float,
    lambda_reg: float,
    m: int,
) -> None:
    """
    Apply one gradient descent update to the parameters, in place.

    Update rule:
        For weights W_l:
            W_l <- W_l - learning_rate * (grads[W_l] + (lambda_reg / m) * W_l)
        For biases b_l:
            b_l <- b_l - learning_rate * grads[b_l]

    Args:
        params:        Parameter dict. Mutated in place.
        grads:         Gradient dict from model.backward. Must have same keys
                       and shapes as params.
        learning_rate: Step size alpha. Must be > 0.
        lambda_reg:    L2 regularization strength. Zero means no regularization.
                       Pass 0.0 explicitly rather than relying on a default.
        m:             Batch size used to compute `grads`. Needed so the L2
                       gradient (lambda_reg / m) * W is on the same scale as
                       the data-loss gradient, which already has a 2/m factor
                       baked in from backward().

    Returns:
        None. Params are updated in place.

    Invariants:
        1. params.keys() is unchanged after the call.
        2. Each params[k].shape is unchanged.
        3. L2 term is added ONLY to weight gradients (keys starting with 'W').
        4. If lambda_reg == 0, behavior is identical to vanilla GD.
    """
    for name in params:
        grad = grads[name]
        if name.startswith("W"):
            # Use `+`, not `+=`: grads is the caller's data. The variance probe
            # reads these same arrays later and needs the unmodified data gradient.
            grad = grad + (lambda_reg / m) * params[name]
        params[name] -= learning_rate * grad


# ---------- Smoke test ----------

if __name__ == "__main__":
    from model import init_parameters, forward, backward

    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 8))
    y = rng.normal(size=(10, 1))
    params = init_parameters([8, 32, 16, 1], seed=1)

    # Save a snapshot of initial weights and biases for comparison.
    W1_before = params["W1"].copy()
    b1_before = params["b1"].copy()

    # Compute gradients for one step.
    y_hat, cache = forward(X, params)
    grads = backward(y_hat, y, params, cache)

    # Create a copy to preserve the "Pure" data gradients
    grads_snapshot = {k: v.copy() for k, v in grads.items()}

    # Apply one step with nonzero L2.
    optimizer_step(params, grads, learning_rate=0.01, lambda_reg=0.1, m=10)

    # Shape invariants.
    assert params["W1"].shape == W1_before.shape, "W1 shape changed!"
    assert params["b1"].shape == b1_before.shape, "b1 shape changed!"

    # Parameters actually moved.
    assert not np.array_equal(params["W1"], W1_before), "W1 did not update!"
    assert not np.array_equal(params["b1"], b1_before), "b1 did not update!"

    # Same keys.
    assert set(params.keys()) == {"W1", "b1", "W2", "b2", "W3", "b3"}

    # Grads preserved.
    for key in grads:
        assert np.array_equal(
            grads[key], grads_snapshot[key]
        ), f"Invariant Violated: optimizer_step mutated grads['{key}'] in-place!"

    print(
        "Single step — OK. W1 and b1 both moved, shapes preserved, keys preserved. grads preserved."
    )

    # ----- Verify the bias-exemption invariant -----
    # If we run with lambda_reg=0, bias update should be identical to lambda_reg > 0,
    # because L2 doesn't touch biases. Weight updates SHOULD differ.

    params_a = init_parameters([8, 32, 16, 1], seed=1)
    params_b = init_parameters([8, 32, 16, 1], seed=1)

    y_hat_a, cache_a = forward(X, params_a)
    grads_a = backward(y_hat_a, y, params_a, cache_a)
    y_hat_b, cache_b = forward(X, params_b)
    grads_b = backward(y_hat_b, y, params_b, cache_b)

    # grads_a and grads_b are identical (same init, same X, same y).
    optimizer_step(params_a, grads_a, learning_rate=0.01, lambda_reg=0.0, m=10)
    optimizer_step(params_b, grads_b, learning_rate=0.01, lambda_reg=0.1, m=10)

    # Biases: same in both runs (L2 didn't touch them).
    assert np.allclose(
        params_a["b1"], params_b["b1"]
    ), "L2 affected b1 — invariant violated!"
    assert np.allclose(
        params_a["b2"], params_b["b2"]
    ), "L2 affected b2 — invariant violated!"
    assert np.allclose(
        params_a["b3"], params_b["b3"]
    ), "L2 affected b3 — invariant violated!"

    # Weights: different between the two runs (L2 pulled them toward zero more).
    assert not np.allclose(params_a["W1"], params_b["W1"]), "L2 did not affect W1!"

    print(
        "Bias-exemption invariant — OK. L2 changes weight updates but not bias updates."
    )
    print("Optimizer smoke test passed.")
