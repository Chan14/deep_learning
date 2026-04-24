"""
model.py — Three-layer MLP for California Housing regression.

Architecture:
    Input (8) -> Hidden1 (32, ReLU) -> Hidden2 (16, ReLU) -> Output (1, linear)

Design notes:
    - Parameters stored in a dict for easy iteration (optimizers, norms).
    - Forward returns (y_hat, cache); backward consumes cache. No recomputation.
    - L2 regularization lives in losses.py, NOT here. Model computes data-loss
      gradients only; the regularizer's contribution is added at the optimizer.

Verified: Gradient check passed (see grad_check.py).
Max relative error across all parameters: ~5e-9.
"""

import numpy as np


# ---------- Activations ----------


def relu(Z: np.ndarray) -> np.ndarray:
    """
    Element-wise ReLU: max(0, Z).
    Shape preserved: input (m, k) -> output (m, k).
    """
    return np.maximum(0, Z)


def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Backward pass for ReLU. Given the upstream gradient dA (gradient of loss
    w.r.t. ReLU's OUTPUT) and the cached pre-activation Z (ReLU's INPUT),
    returns the gradient of loss w.r.t. ReLU's INPUT.

    ReLU derivative: 1 where Z > 0, else 0. So dZ = dA * (Z > 0).

    Shapes: dA and Z both (m, k); output (m, k).
    """
    return dA * (Z > 0)


# ---------- Parameter initialization ----------


def init_parameters(
    layer_dims: list[int],
    seed: int = 42,
) -> dict:
    """
    Initialize parameters for an L-layer network.

    Args:
        layer_dims: List of layer sizes, e.g. [8, 32, 16, 1] for input 8,
                    two hidden layers (32, 16), output 1.
                    Length L+1 where L is the number of PARAMETERIZED layers.
        seed:       RNG seed for reproducibility.

    Returns:
        params dict with keys W1, b1, W2, b2, ..., WL, bL.

    Initialization strategy:
        - Hidden layers (layers 1 .. L-1): He init. W ~ N(0, 2/fan_in).
          Reason: ReLU kills half the distribution; doubling variance keeps
          activation variance stable across depth.
        - Output layer (layer L): Xavier init. W ~ N(0, 1/fan_in).
          Reason: linear output, no ReLU correction needed.
        - All biases: zeros.

    Invariant: params[f"W{l}"] has shape (layer_dims[l-1], layer_dims[l]).
               params[f"b{l}"] has shape (1, layer_dims[l]).
    """
    rng = np.random.default_rng(seed)
    params = {}
    L = len(layer_dims) - 1  # number of parameterized layers

    for l in range(1, L + 1):
        fan_in = layer_dims[l - 1]
        fan_out = layer_dims[l]

        is_output = l == L
        variance = 1 / fan_in if is_output else 2 / fan_in
        std = np.sqrt(variance)

        params[f"W{l}"] = rng.normal(0, std, size=(fan_in, fan_out))
        params[f"b{l}"] = np.zeros((1, fan_out))

    return params


# ---------- Forward pass ----------


def forward(X: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """
    Forward pass through the 3-layer MLP.

    Args:
        X:      Input features, shape (m, 8).
        params: Parameter dict from init_parameters.

    Returns:
        y_hat:  Predictions, shape (m, 1).
        cache:  Dict containing all intermediate quantities needed for backward.
                Keys: X, Z1, A1, Z2, A2, Z3.
                (Note: A3 == Z3 since output is linear, so we skip it.)

    Invariant: No in-place modification of params. Forward is a pure function
               of (X, params).
    """

    Z1 = X @ params["W1"] + params["b1"]  # (m, 32)
    A1 = relu(Z1)  # (m, 32)

    Z2 = A1 @ params["W2"] + params["b2"]  # (m, 16)
    A2 = relu(Z2)  # (m, 16)

    Z3 = A2 @ params["W3"] + params["b3"]  # (m, 1)
    y_hat = Z3  # (linear output)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3}
    return y_hat, cache


# ---------- Backward pass ----------


def backward(
    y_hat: np.ndarray,
    y: np.ndarray,
    params: dict,
    cache: dict,
) -> dict:
    """
    Backward pass. Computes gradients of MSE data loss w.r.t. all parameters.

    MSE loss: L = (1/m) * sum((y_hat - y)^2)  [note: NOT (1/2m)]
    So dL/dy_hat = (2/m) * (y_hat - y).
    (We use 2/m, not 1/m — keeps the math clean and matches standard MSE.)

    L2 regularization is NOT added here. The caller adds lambda*W to each
    weight gradient at the optimizer step. Keeps responsibilities clean.

    Args:
        y_hat:  Predictions from forward, shape (m, 1).
        y:      True targets, shape (m, 1).
        params: Parameter dict (needed for weight matrices in backprop).
        cache:  Intermediates from forward.

    Returns:
        grads dict with keys W1, b1, W2, b2, W3, b3.
        Invariant: grads[k].shape == params[k].shape for every k.
    """
    m = y.shape[0]
    grads = {}

    # Unpack cache for readability
    X, Z1, A1, Z2, A2 = cache["X"], cache["Z1"], cache["A1"], cache["Z2"], cache["A2"]

    # ---- Output layer (layer 3, linear) ----
    dZ3 = (2 / m) * (y_hat - y)  # shape (m, 1)
    grads["W3"] = A2.T @ dZ3  # shape (16, 1)
    grads["b3"] = dZ3.sum(axis=0, keepdims=True)  # shape (1, 1)
    dA2 = dZ3 @ params["W3"].T  # shape (m, 16)

    # ---- Hidden layer 2 (ReLU) ----
    dZ2 = relu_backward(dA2, Z2)  # shape (m, 16)
    grads["W2"] = A1.T @ dZ2  # shape (32, 16)
    grads["b2"] = dZ2.sum(axis=0, keepdims=True)  # shape (1, 16)
    dA1 = dZ2 @ params["W2"].T  # shape (m, 32)

    # ---- Hidden layer 1 (ReLU) ----
    dZ1 = relu_backward(dA1, Z1)  # (m, 32)
    grads["W1"] = X.T @ dZ1  # shape (8, 32)
    grads["b1"] = dZ1.sum(axis=0, keepdims=True)  # shape (1, 32)

    # Sanity check — invariant 5
    for k in params:
        assert (
            grads[k].shape == params[k].shape
        ), f"Shape mismatch for {k}: grad {grads[k].shape} vs param {params[k].shape}"

    return grads


# ---------- Smoke test ----------

if __name__ == "__main__":
    # Tiny sanity check: does forward run? does backward run? shapes right?
    rng = np.random.default_rng(0)
    X_fake = rng.normal(size=(5, 8))
    y_fake = rng.normal(size=(5, 1))

    params = init_parameters([8, 32, 16, 1])
    y_hat, cache = forward(X_fake, params)
    grads = backward(y_hat, y_fake, params, cache)

    print(f"y_hat shape: {y_hat.shape}  (expected (5, 1))")
    for k in params:
        print(f"  {k}: param {params[k].shape}, grad {grads[k].shape}")
    print("Forward + backward smoke test passed.")
