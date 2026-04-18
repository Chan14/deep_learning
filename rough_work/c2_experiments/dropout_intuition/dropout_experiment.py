import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
np.random.seed(42)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
LAYER_DIMS = [20, 16, 8, 1]  # input_dim, hidden..., output
N_TRAIN = 80
N_VAL = 40
INPUT_DIM = 20
EPOCHS = 400
LR = 0.01
KEEP_PROB = 0.5  # dropout keep probability
RECORD_EVERY = 5  # record metrics every N epochs


# ─────────────────────────────────────────────
# Data — synthetic, designed to be easy to overfit
# X: (input_dim, m), Y: (1, m)
# ─────────────────────────────────────────────
def make_data(m, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((d, m))
    # nonlinear signal in first 5 dims only; rest is noise
    signal = X[0] * X[1] - X[2] * X[3] + X[4]
    Y = (signal > 0).astype(float).reshape(1, m)
    return X, Y


X_train, Y_train = make_data(N_TRAIN, INPUT_DIM, seed=1)
X_val, Y_val = make_data(N_VAL, INPUT_DIM, seed=2)


# ─────────────────────────────────────────────
# Parameter init — He initialization
# params: dict of W1, b1, W2, b2, ...
# ─────────────────────────────────────────────
def init_params(layer_dims):
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        params[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(
            2 / layer_dims[l - 1]
        )
        params[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return params


# ─────────────────────────────────────────────
# Forward pass
# Invariant: mask is generated here and stored in cache
#            so the SAME mask is reused in backward pass
# train=True  → apply dropout
# train=False → no dropout (inference mode)
# ─────────────────────────────────────────────
def forward(X, params, keep_prob=1.0, train=True):
    cache = {"A0": X, "masks": {}}
    A = X
    L = len(LAYER_DIMS) - 1

    for l in range(1, L + 1):
        W = params[f"W{l}"]
        b = params[f"b{l}"]
        Z = W @ A + b
        cache[f"Z{l}"] = Z

        if l < L:
            A = np.maximum(0, Z)  # ReLU
            if train and keep_prob < 1.0:
                mask = np.random.rand(*A.shape) < keep_prob
                A = A * mask / keep_prob  # inverted dropout
                cache["masks"][l] = mask  # store for backprop
        else:
            A = 1 / (1 + np.exp(-Z))  # sigmoid output

        cache[f"A{l}"] = A

    return cache


# ─────────────────────────────────────────────
# Loss — binary cross entropy
# ─────────────────────────────────────────────
def compute_loss(AL, Y):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-8, 1 - 1e-8)
    return -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))


# ─────────────────────────────────────────────
# Backward pass
# Invariant: same mask from forward pass is applied to dA
#            same 1/p scaling is applied to gradients
# ─────────────────────────────────────────────
def backward(cache, params, Y, keep_prob=1.0, train=True):
    grads = {}
    m = Y.shape[1]
    L = len(LAYER_DIMS) - 1

    # output layer gradient (sigmoid + BCE combined)
    AL = cache[f"A{L}"]
    dZ = AL - Y  # (n_L, m)

    for l in range(L, 0, -1):
        A_prev = cache[f"A{l-1}"]
        grads[f"dW{l}"] = (dZ @ A_prev.T) / m
        grads[f"db{l}"] = np.mean(dZ, axis=1, keepdims=True)

        if l > 1:
            W = params[f"W{l}"]
            dA = W.T @ dZ  # gradient wrt A[l-1]

            # apply same dropout mask backward
            if train and keep_prob < 1.0 and (l - 1) in cache["masks"]:
                mask = cache["masks"][l - 1]
                dA = dA * mask / keep_prob  # same mask, same scale

            # ReLU backward
            Z_prev = cache[f"Z{l-1}"]
            dZ = dA * (Z_prev > 0)

    return grads


# ─────────────────────────────────────────────
# Parameter update — vanilla SGD
# ─────────────────────────────────────────────
def update(params, grads, lr):
    p = {}
    L = len(LAYER_DIMS) - 1
    for l in range(1, L + 1):
        p[f"W{l}"] = params[f"W{l}"] - lr * grads[f"dW{l}"]
        p[f"b{l}"] = params[f"b{l}"] - lr * grads[f"db{l}"]
    return p


# ─────────────────────────────────────────────
# Weight norm — mean Frobenius norm across all W layers
# ─────────────────────────────────────────────
def mean_weight_norm(params):
    L = len(LAYER_DIMS) - 1
    return np.mean([np.linalg.norm(params[f"W{l}"], "fro") for l in range(1, L + 1)])


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(params, keep_prob, epochs, lr, record_every=RECORD_EVERY):
    history = {"train_loss": [], "val_loss": [], "weight_norm": [], "epoch": []}
    p = {k: v.copy() for k, v in params.items()}
    L = len(LAYER_DIMS) - 1

    for e in range(epochs):
        cache = forward(X_train, p, keep_prob=keep_prob, train=True)
        grads = backward(cache, p, Y_train, keep_prob=keep_prob, train=True)
        p = update(p, grads, lr)

        if e % record_every == 0 or e == epochs - 1:
            t_cache = forward(X_train, p, keep_prob=1.0, train=False)
            v_cache = forward(X_val, p, keep_prob=1.0, train=False)
            history["train_loss"].append(compute_loss(t_cache[f"A{L}"], Y_train))
            history["val_loss"].append(compute_loss(v_cache[f"A{L}"], Y_val))
            history["weight_norm"].append(mean_weight_norm(p))
            history["epoch"].append(e + 1)
    # print(p)
    return history


# ─────────────────────────────────────────────
# Run both networks from identical init
# ─────────────────────────────────────────────
init = init_params(LAYER_DIMS)

print("Training without dropout...")
h_no = train(
    {k: v.copy() for k, v in init.items()}, keep_prob=1.0, epochs=EPOCHS, lr=LR
)

print("Training with dropout...")
h_do = train(
    {k: v.copy() for k, v in init.items()}, keep_prob=KEEP_PROB, epochs=EPOCHS, lr=LR
)

print(f"\n{'':─<50}")
print(f"{'':20} {'No Dropout':>12} {'Dropout':>12}")
print(f"{'':─<50}")
print(
    f"{'Final train loss':20} {h_no['train_loss'][-1]:>12.4f} {h_do['train_loss'][-1]:>12.4f}"
)
print(
    f"{'Final val loss':20} {h_no['val_loss'][-1]:>12.4f} {h_do['val_loss'][-1]:>12.4f}"
)
print(
    f"{'Val-Train gap':20} {h_no['val_loss'][-1]-h_no['train_loss'][-1]:>12.4f} {h_do['val_loss'][-1]-h_do['train_loss'][-1]:>12.4f}"
)
print(
    f"{'Mean weight norm':20} {h_no['weight_norm'][-1]:>12.4f} {h_do['weight_norm'][-1]:>12.4f}"
)
print(f"{'':─<50}")

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Dropout Experiment  |  keep_prob={KEEP_PROB}  lr={LR}  epochs={EPOCHS}",
    fontsize=12,
)

epochs_axis = h_no["epoch"]

ax1.plot(
    epochs_axis, h_no["train_loss"], color="#D85A30", lw=1.5, label="no dropout — train"
)
ax1.plot(
    epochs_axis,
    h_no["val_loss"],
    color="#D85A30",
    lw=1.5,
    ls="--",
    label="no dropout — val",
)
ax1.plot(
    epochs_axis, h_do["train_loss"], color="#1D9E75", lw=1.5, label="dropout — train"
)
ax1.plot(
    epochs_axis,
    h_do["val_loss"],
    color="#1D9E75",
    lw=1.5,
    ls="--",
    label="dropout — val",
)
ax1.set_title("Loss curves")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Binary cross-entropy loss")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.2)

ax2.plot(epochs_axis, h_no["weight_norm"], color="#D85A30", lw=1.5, label="no dropout")
ax2.plot(epochs_axis, h_do["weight_norm"], color="#1D9E75", lw=1.5, label="dropout")
ax2.set_title("Mean weight norm (Frobenius) across layers")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean ||W||_F")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("dropout_experiment.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to dropout_experiment.png")
