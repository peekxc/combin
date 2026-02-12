from math import comb

import numpy as np

from combin.special import binom, binom_int_exact


def test_binom_int_exact_matches_math_comb_grid():
    n = np.arange(0, 35, dtype=np.int64)
    k = np.arange(0, 35, dtype=np.int64)
    N, K = np.meshgrid(n, k, indexing="ij")
    got = binom_int_exact(N, K)
    expected = np.zeros_like(got, dtype=np.uint64)
    for i in range(N.shape[0]):
        for j in range(N.shape[1]):
            nn, kk = int(N[i, j]), int(K[i, j])
            expected[i, j] = comb(nn, kk) if 0 <= kk <= nn else 0
    assert np.array_equal(got, expected)


def test_binom_int_exact_broadcast_and_invalid():
    n = np.array([5, 6, 7, -1])
    k = 2
    got = binom_int_exact(n, k)
    assert np.array_equal(got, np.array([10, 15, 21, 0], dtype=np.uint64))

    got2 = binom_int_exact(np.array([4, 4, 4]), np.array([-1, 2, 8]))
    assert np.array_equal(got2, np.array([0, 6, 0], dtype=np.uint64))


def test_binom_int_exact_matches_current_binom_safe_range():
    rng = np.random.default_rng(0)
    n = rng.integers(0, 35, size=500, dtype=np.int64)
    k = rng.integers(0, 35, size=500, dtype=np.int64)
    a = binom_int_exact(n, k)
    b = binom(n, k)
    assert np.array_equal(a, b)
