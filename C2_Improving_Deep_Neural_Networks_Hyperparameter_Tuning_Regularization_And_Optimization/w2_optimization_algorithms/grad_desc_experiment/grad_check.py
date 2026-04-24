"""
grad_check.py — Numerical gradient check for model.py.

Verifies that the analytic gradients from model.backward match centered-
difference numeric gradients to high precision. This is the single most
important unit test for any hand-rolled backprop.

Run: python grad_check.py
"""

import numpy as np
from model import init_parameters, forward, backward


def compute_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    MSE loss (matches the derivation in model.backward):
        L = (1/m) * sum((y_hat - y)^2)
    NOTE: No L2 term here. We're checking gradients of the data loss only,
    consistent with what backward() computes.
    """
    m = y.shape[0]
    L = (1 / m) * np.sum((y_hat - y) ** 2)
    return float(
        L
    )  # defensive: forces scalar even if something else regresses. Note - np.sum without the axis is always a scalar.


def numeric_gradient(
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    key: str,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute the numeric gradient of the loss w.r.t. params[key], via
    centered differences. Returns an array the same shape as params[key].

    For each scalar entry (i, j) in params[key]:
        1. Save original value.
        2. Set params[key][i, j] = original + epsilon.  Compute J_plus.
        3. Set params[key][i, j] = original - epsilon.  Compute J_minus.
        4. Restore original.
        5. numeric_grad[i, j] = (J_plus - J_minus) / (2 * epsilon).

    CAUTION: mutate-and-restore on the params dict. If you forget to restore
    you will silently corrupt the params for subsequent checks.
    """
    param = params[key]
    grad = np.zeros_like(param)

    for idx in np.ndindex(param.shape):
        original = param[idx]
        param[idx] = original + epsilon
        y_hat_plus, _ = forward(X, params)
        J_plus = compute_loss(y_hat_plus, y)

        param[idx] = original - epsilon
        y_hat_minus, _ = forward(X, params)
        J_minus = compute_loss(y_hat_minus, y)

        param[idx] = original  # <-- RESTORE
        # print("J_minus:", J_minus, type(J_minus), getattr(J_minus, "shape", "scalar"))
        grad[idx] = (J_plus - J_minus) / (2 * epsilon)
    return grad


def relative_error(g1: np.ndarray, g2: np.ndarray) -> float:
    """
    Relative error between two gradient arrays:
        ||g1 - g2|| / (||g1|| + ||g2||)
    Uses L2 (Frobenius) norm. Small epsilon in denominator to avoid
    division by zero when both gradients are zero.
    """
    num = np.linalg.norm(g1 - g2)
    den = np.linalg.norm(g1) + np.linalg.norm(g2) + 1e-12
    return num / den


def run_grad_check():
    """
    Small-scale gradient check on random data.

    Why small? Numeric gradient requires 2 forward passes per scalar parameter.
    Our model has ~849 parameters, so that's ~1700 forward passes. Manageable
    on a tiny dataset but would be painful on full California Housing.
    Use m=10 examples — enough to exercise all code paths, small enough to be fast.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 8))
    y = rng.normal(size=(10, 1))

    params = init_parameters([8, 32, 16, 1], seed=1)

    # Analytic gradients
    y_hat, cache = forward(X, params)
    analytic_grads = backward(y_hat, y, params, cache)

    # Numeric gradients, one parameter at a time
    print(f"{'param':<6} {'rel_error':<14} {'status'}")
    print("-" * 40)
    all_passed = True
    for key in params:
        num_grad = numeric_gradient(params, X, y, key)
        err = relative_error(analytic_grads[key], num_grad)
        status = "OK" if err < 1e-5 else "FAIL"
        if err >= 1e-5:
            all_passed = False
        print(f"{key:<6} {err:<14.3e} {status}")

    print("-" * 40)
    print("ALL PASS" if all_passed else "CHECK FAILED — investigate before proceeding")


if __name__ == "__main__":
    run_grad_check()
