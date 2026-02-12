"""Contains functions for enumerating, ranking, sampling, and checking combinations."""

import math
from math import ceil, comb, factorial, floor
from numbers import Integral
from typing import Iterable, Iterator, Literal, Union, Sequence, Self

import numpy as np
from more_itertools import collapse, first_true, spy
from numpy.typing import ArrayLike

from .utility import ensure
from .special import binom


# https://stackoverflow.com/questions/42138681/faster-numpy-solution-instead-of-itertools-combinations
def combs(n: int, k: int) -> np.ndarray:
    """Generates all C(n,k) combinations of the standard index set."""
    a = np.ones((k, n - k + 1), dtype=int)
    a[0] = np.arange(n - k + 1)
    for j in range(1, k):
        reps = (n - k + j) - a[j - 1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1 - reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
    return a.T


## Element-specific lexicographical ranking function
def _comb_unrank_lex(r: int, n: int, k: int) -> tuple[int]:
    result: list[int] = [0] * k
    x = 1
    for i in range(1, k + 1):
        while r >= comb(n - x, k - i):
            r -= comb(n - x, k - i)
            x += 1
        result[i - 1] = int(x - 1)
        x += 1
    return tuple(result)  # type: ignore


def comb_unrank_lex(ranks: ArrayLike, n: int, k: int) -> np.ndarray:
    """Unranks ranks representing into k-combinations from an n-element set in lexicographical order.

    Args:
                    ranks: array of integer ranks.
                    n: number of elements in the underlying set.
                    k: size of the combinations the ranks represent.

    Returns:
                    integer ranks for each combination.

    Notes:
                    For valid combinations, the resulting combinations are guaranteed to have entries in the range {0, ..., n-1} and
                    to be in bijection with their corresponding ranks, so long as $C(n,k) < 2^64$. The combinations for
                    invalid ranks are undefined. Uses O(n * k) memory.
    """
    ranks = np.atleast_1d(ranks).astype(np.int64).copy()
    m = np.size(ranks, axis=0)
    M = np.arange(m)
    out = np.empty((m, k), dtype=np.int64)
    x = np.ones(m, dtype=np.int64)
    xs_full = np.arange(1, n + 1, dtype=np.int64)
    for i in range(k):
        rem = k - i
        xs = xs_full[: n - rem]
        B = np.broadcast_to(binom(n - xs, rem), (m, xs.size))
        S = np.cumsum(B, axis=1)
        idx = (S > ranks[:, None]).argmax(axis=1)
        idx = np.where(S[M, idx] <= ranks, xs.size - 1, idx)
        prev = np.where(idx > 0, S[M, idx - 1], 0).astype(np.int64)
        ranks -= prev
        out[:, i] = x + idx - 1
        x += idx + 1
    return out if m > 1 else out.ravel()


def comb_rank_colex(combs: ArrayLike, is_sorted: bool = False):
    """Ranks k-combinations from an n-element set in colexicographical order.

    Args:
            combs: array-like of k-combinations.
            is_sorted: whether the combinations are given in sorted order already. Defaults to False.

    Returns:
            integer ranks for each combination.

    Notes:
            For valid combinations, the resulting ranks are guaranteed to be in the range {0, ..., C(n,k) - 1} and
            to be in bijection with their corresponding combinations, so long as $C(n,k) < 2^64$. The ranks for
            invalid combinations is undefined. Uses O(|combs| * k) memory.
    """
    C = np.atleast_2d(combs)
    k = C.shape[1]
    K = np.arange(k, dtype=np.int64) if is_sorted else np.argsort(C, axis=1) + 1
    return np.sum(binom(C, K), axis=1)


## Dummy version of the above; useful for testing
def _comb_rank_lex(c: Iterable, n: int) -> int:
    c = tuple(sorted(c))
    k = len(c)
    index = sum([comb(int(n - ci - 1), int(k - i)) for i, ci in enumerate(c)])
    return int(comb(n, k) - index - 1)


## Dummy version of the above; useful for testing
def _comb_rank_colex(c: Iterable) -> int:
    c = tuple(sorted(c))
    k = len(c)
    return sum([comb(ci, k - i) for i, ci in enumerate(reversed(c))])


def comb_rank_lex(
    combs: ArrayLike, n: int, N: int | None = None, is_sorted: bool = False
) -> np.ndarray:
    """Ranks k-combinations from an n-element set in lexicographical order.

    Args:
            combs: array-like of k-combinations.
            n: number of elements in the underlying set.
            N: Binomial coefficient C(n,k), if known.
            is_sorted: whether the combinations are given in sorted order already. Defaults to False.

    Returns:
            integer ranks for each combination.

    Notes:
            For valid combinations, the resulting ranks are guaranteed to be in the range {0, ..., C(n,k) - 1} and
            to be in bijection with their corresponding combinations, so long as $C(n,k) < 2^64$. The ranks for
            invalid combinations is undefined. Uses O(|combs| * k) memory.
    """
    C = np.atleast_2d(combs)
    n, k = int(n), C.shape[1]
    N = comb(n, k) if N is None else int(N)
    K = (
        (k - np.argsort(C, axis=1))
        if not is_sorted
        else np.arange(k, 0, -1, dtype=np.int64)
    )
    val = binom((n - 1) - C, K).sum(axis=1)
    return (N - 1) - val


def _comb_unrank_colex(r: int, k: int) -> tuple:
    r"""Unranks a k-combinations rank 'r' back into the original combination in colex order.

    This function uses a simple unranking process for testing purposes. For more efficient unranking, see `comb_to_rank`.

    When $k < n - k$, this function takes $O(k^2 (n - k))$ time, where n is
    the largest integer satisfying $0 <= r < C(n,k)$.

    Notes:
            Implements Algorithm 1 from [1].

    References:
            1. Kruchinin, Vladimir, et al. "Unranking Small Combinations of a Large Set in Co-Lexicographic Order." Algorithms 15.2 (2022): 36.
    """
    c = [0] * k
    for i in reversed(range(1, k + 1)):  # O(k)
        m = i
        ## O(n) as min comparisons == n − m + 1 (when i = m), max comparisons is n (when i = 1);
        while r >= comb(m, i):
            m += 1
        c[i - 1] = m - 1
        r -= comb(m - 1, i)  # comb is O(min(n-k, k)) ~ O(k)
    return tuple(c)


def comb_unrank_colex_findn(ranks: ArrayLike, k: int) -> np.ndarray:
    """Colex unranking using a find_n-based bound for the candidate table.

    This keeps vectorized updates, but shrinks the temporary B table width by
    using `find_n` to locate a tight upper limit on candidate m values.
    """
    R = np.array(ranks, dtype=np.int64, copy=True, ndmin=1)
    comb_out = np.empty((R.size, k), dtype=np.int64)
    if R.size == 0:
        return comb_out

    for i in reversed(range(1, k + 1)):
        r_max = int(np.max(R))
        m_hi = max(i, int(find_n(r_max, i)) + 1)
        while comb(m_hi, i) <= r_max:
            m_hi += 1

        ms = np.arange(i, m_hi + 1, dtype=np.int64)
        B = np.broadcast_to(binom(ms[None, :], i), (R.size, ms.size))
        mask = B > R[:, None]
        m_choice = ms[mask.argmax(axis=1)]
        comb_out[:, i - 1] = m_choice - 1
        R -= binom(m_choice - 1, i).astype(np.int64)

    return comb_out.ravel() if np.size(ranks) == 1 else comb_out


def comb_unrank_lex_findn(ranks: ArrayLike, n: int, k: int) -> np.ndarray:
    """Vectorized lex unranking that uses the find_n-bounded colex backend.

    Uses the lex/colex dual relation q = C(n,k) - 1 - r.
    """
    R = np.array(ranks, dtype=np.int64, copy=True, ndmin=1)
    N = comb(n, k)
    Q = (N - 1) - R
    B = np.atleast_2d(comb_unrank_colex_findn(Q, k))
    out = ((n - 1) - B[:, ::-1]).astype(np.int64, copy=False)
    return out if np.size(ranks) > 1 else out.ravel()


def comb_unrank_lex(ranks: ArrayLike, n: int, k: int) -> np.ndarray:
    """Working vectorized lex unranking kept separate for benchmarking.

    This implementation intentionally does not replace ``comb_unrank_lex`` so
    performance and correctness can be benchmarked side-by-side.
    """
    R = np.array(ranks, dtype=np.int64, copy=True, ndmin=1)
    N = comb(n, k)
    Q = (N - 1) - R

    # Vectorized colex unranking backend (no find_n bound).
    C = np.empty((Q.size, k), dtype=np.int64)
    Rc = Q.copy()
    if Rc.size > 0:
        max_rank = int(np.max(Rc))
        for i in reversed(range(1, k + 1)):
            ms = np.arange(i, max_rank + i + 1, dtype=np.int64)
            B = np.broadcast_to(binom(ms[None, :], i), (Rc.size, ms.size))
            mask = B > Rc[:, None]
            m_choice = ms[mask.argmax(axis=1)]
            C[:, i - 1] = m_choice - 1
            Rc -= binom(m_choice - 1, i).astype(np.int64)

    out = ((n - 1) - C[:, ::-1]).astype(np.int64, copy=False)
    return out if np.size(ranks) > 1 else out.ravel()


def comb_unrank_lex_vec(ranks: ArrayLike, n: int, k: int) -> np.ndarray:
    """Vectorized lex unranking variant for benchmarking."""
    return comb_unrank_lex(ranks=ranks, n=n, k=k)


def comb_unrank_colex(ranks: ArrayLike, k: int) -> np.ndarray:
    """Colex unranking.

    Args:
            ranks: scalar or array of ranks
            k: size of the combinations.

    Returns:
            array of integer ranks.
    """
    R = np.atleast_1d(ranks).copy()
    comb_out = np.empty((R.size, k), dtype=np.int64)

    max_rank = np.max(R)
    for i in reversed(range(1, k + 1)):
        # Candidate n values; upper bound safe over all ranks
        ms = np.arange(
            i, max_rank + i + 1, dtype=np.int64
        )  ## TODO: reduce space complexity using find_n

        ## Broadcast binomial coefficient table
        B = np.broadcast_to(binom(ms[None, :], i), (ranks.size, ms.size))
        mask = B > ranks[:, None]

        ## Take the first m where comb(m, i) > r
        m_choice = ms[mask.argmax(axis=1)]
        comb_out[:, i - 1] = m_choice - 1
        ranks -= binom(m_choice - 1, i).astype(np.int64)

    return comb_out.ravel() if ranks.size == 1 else comb_out


class CombinationIterator(Iterator):
    def __init__(self, seq: Sequence, k: int, batch_size: int = 1024):
        self.seq = np.atleast_1d(seq)
        self.n = np.size(self.seq, axis=0)
        self.k = k
        self._rank = 0
        self._N = comb(self.n, self.k)
        self._b = batch_size

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> np.ndarray:
        if self._rank >= self._N:
            raise IndexError()
        ranks = np.arange(self._rank, min(self._rank + self._b, self._N))
        indices = comb_unrank_colex(ranks, self.k)
        result = self.seq[indices]
        self._rank += len(ranks)
        return result

    def __repr__(self) -> str:
        msg = f"CombinationIterator over C({self.n},{self.k}) combinations"
        return msg


class Combinations(Iterable):
    def __init__(
        self, seq: Sequence, k: int, batch_size: int = 1024
    ):  # support __getitem__
        self.seq = np.array(seq)
        self.k = k
        self.batch_size = batch_size

    def __iter__(self) -> CombinationIterator:
        return CombinationIterator(self.seq, self.k, self.batch_size)

    def __getitem__(self, key: int | slice) -> np.ndarray:
        pass

    def batched(self, batch_size: int = 1024):
        pass
        # unrank_comb_lex
        # unrank_comb_lex()  # vectorized


def find_n(r: int, k: int) -> int:
    r"""Lower bound for determining binomial coefficients.

    Determines an approximate value $n \in \mathbb{N}$ satisfying:

    $$ C(n-1, k) \leq r < C(n, k) $$

    In $\approx O(1)$ time.

    Parameters:
            r: binomial coefficient
            m: combination tuple size

    Returns:
            The value $n$ with binomial coefficient $r < C(n,k)$.

    References:
            - Kruchinin, Vladimir, et al. "Unranking Small Combinations of a Large Set in Co-Lexicographic Order." Algorithms 15.2 (2022): 36.
    """
    # ensure(k > 0, "m must be greater than 0")
    if r == 0:
        ## If r = 0 we could have any n, but k <= n so just return k
        return k
    if k == 1:
        ## comb(n,1) = n for all n, so 0 <= r < n
        return r
    elif k == 2:
        ## Solving the quadratic: n(n-1)/2 = r
        n = math.ceil(
            (1.0 + math.sqrt(1.0 + 8.0 * r)) / 2.0
        )  # = 1 when r = 0 because ceil
        return n - 1
    elif k == 3:
        ## Approximation for n(n-1)(n-2)/6 = r
        n = math.ceil(math.pow(6.0 * r, 1 / 3)) + 1
        return n - 1
    else:
        ## Stirling approximation
        M = (
            math.log(r) / k
            + math.log(2 * math.pi * k) / (2 * k)
            + (1 / (12 * k**2))
            - 1 / (360 * k**4)
            - 1
        )
        n = math.ceil(k * math.exp(M) + (k - 1) / 2)
        return n - 1


def get_max_vertex(r: int, m: int, n: int, use_lb: bool = True, c_val: int = 0):
    """Finds the largest index $w$ satisfying $r >= choose(w, m)$."""

    # Predicate function: is choose(w, m) <= r?
    def pred(w):
        return binom(w, m) <= r

    # 1. Calculate Lower Bound
    k_lb = find_n(r, m) if use_lb else (m - 1)

    # 2. Check early exits (C parameter logic)
    # Check if the next few integers satisfy the condition to avoid binary search
    for i in range(1, c_val + 1):
        if not pred(k_lb + i):
            return k_lb + i

    # 3. Binary Search in range [k_lb, n]
    low = k_lb
    high = n
    ans = k_lb

    while low <= high:
        mid = (low + high) // 2
        if pred(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1

    return ans + 1


def inverse_choose(x: int, k: int, exact: bool = True):
    r"""Inverse binomial coefficient (approximately).

    This function attempts to find the integer _n_ such that binom(n,k) = x, where _binom_ is the binomial coefficient:

    $$ \mathrm{binom}(n,k) = n! / (k! \cdot (n-k)!) $$

    For k <= 2, a logartihmic numpy-based approach is used and the result is exact.
    For k > 2 and x <= 10e7, an linear-search is used based on tight bounds and the result is exact.
    For k > 2 and x > 10e7; an iterative approach is used based on loose bounds from the formula from this stack exchange post:

    https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
    """
    assert x >= 0, "x must be a non-negative integer"
    if k == 0:
        return 1
    if k == 1:
        return x
    if k == 2:
        rng = np.arange(
            np.floor(np.sqrt(2 * x)), np.ceil(np.sqrt(2 * x) + 2) + 1, dtype=np.uint64
        )
        final_n = rng[np.searchsorted((rng * (rng - 1) / 2), x)]
        if comb(final_n, 2) == x or not exact:
            return final_n
        raise ValueError(f"Failed to invert C(n,{k}) = {x}")
        # return int(rng[x == (rng * (rng - 1) / 2)])
    else:
        # From: https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
        if x < 10**7:
            lb = (factorial(k) * x) ** (1 / k)
            potential_n = np.arange(floor(lb), ceil(lb + k) + 1)
            comb_cand = np.array([comb(n, k) for n in potential_n])
            ub_ind = np.searchsorted(comb_cand, x)
            if exact and x != comb_cand[ub_ind]:
                raise ValueError(f"Unable to invert 'x' = {x}")
            elif exact and x == comb_cand[ub_ind]:
                return potential_n[ub_ind]
            else:  # not exact
                if ub_ind >= len(comb_cand):
                    raise ValueError(
                        f"Low/upper bounds calculations do not hold for 'x' = {x}"
                    )
                return potential_n[ub_ind]
        else:
            lb = np.floor((4**k) / (2 * k + 1))
            C, n = factorial(k) * x, 1
            while n**k < C:
                n = n * 2
            m = first_true((c**k for c in range(1, n + 1)), pred=lambda c: c**k >= C)
            potential_n = range(min([m, 2 * k]), m + k + 1)
            if len(potential_n) == 0:
                raise ValueError(f"Failed to invert C(n,{k}) = {x}")
            final_n = first_true(
                potential_n, default=-1, pred=lambda n: comb(n, k) == x
            )
            if final_n != -1:
                return final_n
            else:
                from scipy.optimize import minimize_scalar

                binom_loss = lambda n: np.abs(comb(int(n), k) - x)
                res = minimize_scalar(binom_loss, bounds=(comb(2 * k, k), x))
                n1, n2 = int(np.floor(res.x)), int(np.ceil(res.x))
                if comb(n1, k) == x:
                    return n1
                if comb(n2, k) == x:
                    return n2
                raise ValueError(f"Failed to invert C(n,{k}) = {x}")
