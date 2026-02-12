import numpy as np
from numpy.typing import ArrayLike
from math import comb


def binom(n: ArrayLike, k: ArrayLike, out: np.ndarray | None = None) -> np.ndarray:
    r"""Binomial coefficient.

    Computes the binomial coefficient $C(n,k)$ given by:

    $$ {n \choose k} = \frac{n!}{k!(n-k)!} = { n - 1 \choose k - 1} + { n - 1 \choose k} $$

    All parameters `n` and `k` are broadcasted appropriately. Uses $O(|n| \cdot |k|)$ memory.
    Uses the multiplicative formula to compute the coefficients.

    Parameters:
            n: numerator of the binomial coefficient.

    Examples:
            >>> binom(10, 2)
            45
            >>> binom(range(10), 2)

    Pyodide:
            <div class="language-python">
                    <code id="code-snippet-1">print("hello")</code>
                    <button class="run-py-btn" data-code-id="code-snippet-1">Run Code</button>
                    <div id="code-snippet-1-output" class="py-output"></div>
            </div>
    """
    n, k = np.asarray(n), np.asarray(k)
    k = np.broadcast_to(k, n.shape)

    # k = np.minimum(k, n - k)
    # k = np.clip(k, 0, n)
    # i = np.arange(1, k.max() + 1)
    # terms = np.where(i <= k[..., None], (n[..., None] - i + 1) / i, 1.0)
    # result = np.rint(np.prod(terms, axis=-1)).astype(np.uint64)

    ## Keep initialize to zero, as out-of-bounds like C(0,k) = 0
    out = np.zeros_like(n, dtype=np.uint64)
    valid = (k >= 0) & (k <= n)
    if not np.any(valid):
        return out
    nv = n[valid]
    kv = np.minimum(k[valid], nv - k[valid])
    i = np.arange(1, kv.max() + 1)
    terms = np.where(i <= kv[..., None], (nv[..., None] - i + 1) / i, 1.0)
    out[valid] = np.rint(np.prod(terms, axis=-1)).astype(np.uint64)

    # return result if np.size(result) > 1 else result.item()
    return out


def binom_int_exact(
    n: ArrayLike, k: ArrayLike, out: np.ndarray | None = None
) -> np.ndarray:
    r"""Exact binomial coefficient using integer arithmetic.

    Computes ``C(n, k)`` elementwise with Python's exact ``math.comb`` and
    broadcasting. Invalid entries (``k < 0`` or ``k > n`` or ``n < 0``) map to 0.

    Returns ``np.uint64`` and raises ``OverflowError`` if any valid coefficient
    does not fit into ``uint64``.
    """
    n_arr, k_arr = np.broadcast_arrays(np.asarray(n), np.asarray(k))
    if out is None:
        out_arr = np.zeros(n_arr.shape, dtype=np.uint64)
    else:
        if out.shape != n_arr.shape:
            raise ValueError("out must have same broadcasted shape as n and k")
        out_arr = out
        out_arr.fill(0)

    valid = (n_arr >= 0) & (k_arr >= 0) & (k_arr <= n_arr)
    if not np.any(valid):
        return out_arr

    nv = n_arr[valid].astype(np.int64, copy=False)
    kv = k_arr[valid].astype(np.int64, copy=False)
    kv = np.minimum(kv, nv - kv)
    vals = [comb(int(nn), int(kk)) for nn, kk in zip(nv, kv)]
    max_u64 = np.iinfo(np.uint64).max
    if any(v > max_u64 for v in vals):
        raise OverflowError("binomial coefficient exceeds uint64 range")
    out_arr[valid] = np.asarray(vals, dtype=np.uint64)
    return out_arr
