"""Pure representation-similarity functions: linear CKA and mutual k-NN.

Each takes two matrices X, Y of shape [N, H] (one layer's last-token reps from
two models, same N examples in the same order).
"""

import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:  # torch is a hard dep here, but keep numpy fallback honest
    _TORCH = False


def _matmul_device():
    """Best torch device for matmuls (CUDA > MPS), or None to use NumPy."""
    if not _TORCH:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def _self_gram(A: np.ndarray) -> np.ndarray:
    """A @ A.T on GPU (MPS/CUDA) if available, else NumPy. Returns float64.

    MPS only supports float32, so the matmul runs in float32 and the result is
    promoted to float64 for the downstream centering/sums.
    """
    A = np.asarray(A)
    dev = _matmul_device()
    if dev is None:
        return (A.astype(np.float64) @ A.astype(np.float64).T)
    t = torch.as_tensor(A, dtype=torch.float32, device=dev)
    return (t @ t.T).cpu().numpy().astype(np.float64)


def _center_columns(A: np.ndarray) -> np.ndarray:
    return A - A.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA via the feature-space (cross-covariance) formula.

    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F), on column-centered X, Y.
    Invariant to orthogonal rotation, isotropic scaling, neuron permutation.
    """
    X = _center_columns(np.asarray(X, dtype=np.float64))
    Y = _center_columns(np.asarray(Y, dtype=np.float64))
    xty = X.T @ Y  # [Hx, Hy]
    numerator = np.sum(xty**2)  # ||X^T Y||_F^2
    xtx = X.T @ X
    yty = Y.T @ Y
    denom = np.linalg.norm(xtx, "fro") * np.linalg.norm(yty, "fro")
    if denom == 0.0:
        return 0.0
    return float(numerator / denom)


def _double_center(G: np.ndarray) -> np.ndarray:
    """Apply H G H (centering matrix H = I - 11^T/n) to a Gram matrix."""
    rmean = G.mean(axis=0, keepdims=True)
    cmean = G.mean(axis=1, keepdims=True)
    return G - rmean - cmean + G.mean()


def _cka_from_raw_grams(K: np.ndarray, L: np.ndarray) -> float:
    """Linear CKA from raw (uncentered) N x N Gram matrices K=XX^T, L=YY^T.

    Equivalent to the feature-space formula via tr(K_c L_c)/sqrt(tr(K_c^2)tr(L_c^2))
    where K_c = H K H is the double-centered Gram (matches column-centering X).
    """
    Kc = _double_center(K)
    Lc = _double_center(L)
    denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    if denom == 0.0:
        return 0.0
    return float(np.sum(Kc * Lc) / denom)


def cka_bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 42,
    max_n: int = 1024,
):
    """Fast bootstrapped linear CKA via the Gram-matrix identity.

    Precomputes the raw N x N Gram matrices once, so each bootstrap iteration is
    an O(n^2) submatrix select + double-center instead of recomputing the
    O(n^2 * H) feature contraction. Optionally subsamples to max_n rows first
    (full-N bootstrap is infeasible at H=2048). Returns (mean, lo, hi).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if max_n is not None and n > max_n:
        keep = rng.choice(n, size=max_n, replace=False)
        X, Y = X[keep], Y[keep]
        n = max_n
    K = _self_gram(X)
    L = _self_gram(Y)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sel = np.ix_(idx, idx)
        vals[b] = _cka_from_raw_grams(K[sel], L[sel])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(vals, alpha))
    hi = float(np.quantile(vals, 1.0 - alpha))
    return float(vals.mean()), lo, hi


def _l2_normalize(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return A / norms


def _knn_indices(A: np.ndarray, k: int) -> np.ndarray:
    """Top-k neighbor indices per row by inner product on L2-normalized A.

    Self is excluded by setting the diagonal similarity to -inf.
    Returns an [N, k] int array.
    """
    A = _l2_normalize(np.asarray(A, dtype=np.float64))
    sim = _self_gram(A)  # cosine sim (rows are unit vectors), on GPU if available
    np.fill_diagonal(sim, -np.inf)
    # argpartition for top-k (unordered within the top-k is fine for set overlap)
    nn = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    return nn


def mutual_knn(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    """Mutual k-NN alignment (Platonic Representation Hypothesis metric).

    L2-normalize, rank neighbors by inner product (== cosine == Euclidean order
    for unit vectors), then average the per-example neighbor-set overlap / k.
    """
    nn_x = _knn_indices(X, k)
    nn_y = _knn_indices(Y, k)
    # Vectorized set-overlap: for each row, count shared neighbor indices.
    # nn_x[:, :, None] == nn_y[:, None, :] -> [N, k, k]; any-match over axis 2
    # gives, per x-neighbor, whether it appears in y's set; sum over axis 1.
    matches = (nn_x[:, :, None] == nn_y[:, None, :]).any(axis=2)
    overlaps = matches.sum(axis=1) / k
    return float(overlaps.mean())


def _topk_from_sim(sim: np.ndarray, k: int) -> np.ndarray:
    """Top-k neighbor indices per row of a similarity matrix (self pre-masked)."""
    return np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]


def _mutual_overlap(nn_x: np.ndarray, nn_y: np.ndarray, k: int) -> float:
    matches = (nn_x[:, :, None] == nn_y[:, None, :]).any(axis=2)
    return float((matches.sum(axis=1) / k).mean())


def knn_bootstrap(X, Y, k_values, n_boot=200, ci=0.95, seed=42):
    """Bootstrapped mutual k-NN for several k at once, sharing the matmul.

    Precomputes the N x N cosine-similarity matrices for X and Y once (on GPU if
    available). Each bootstrap resample then only slices the resampled submatrix
    and ranks neighbors -- no per-iteration, per-k matmul. Returns a dict
    k -> (mean, lo, hi).
    """
    Xn = _l2_normalize(np.asarray(X, dtype=np.float64))
    Yn = _l2_normalize(np.asarray(Y, dtype=np.float64))
    simX = _self_gram(Xn)
    simY = _self_gram(Yn)
    n = simX.shape[0]
    rng = np.random.default_rng(seed)
    vals = {k: np.empty(n_boot, dtype=np.float64) for k in k_values}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sel = np.ix_(idx, idx)
        sx = simX[sel].copy()
        sy = simY[sel].copy()
        np.fill_diagonal(sx, -np.inf)
        np.fill_diagonal(sy, -np.inf)
        for k in k_values:
            nn_x = _topk_from_sim(sx, k)
            nn_y = _topk_from_sim(sy, k)
            vals[k][b] = _mutual_overlap(nn_x, nn_y, k)
    alpha = (1.0 - ci) / 2.0
    out = {}
    for k in k_values:
        v = vals[k]
        out[k] = (
            float(v.mean()),
            float(np.quantile(v, alpha)),
            float(np.quantile(v, 1.0 - alpha)),
        )
    return out


def bootstrap_ci(
    fn,
    X: np.ndarray,
    Y: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
):
    """Resample the N examples with replacement; recompute fn(X[idx], Y[idx]).

    The SAME resampled indices are applied to both X and Y so the paired
    example correspondence is preserved. Returns (mean, lo, hi).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = fn(X[idx], Y[idx])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(vals, alpha))
    hi = float(np.quantile(vals, 1.0 - alpha))
    return float(vals.mean()), lo, hi
