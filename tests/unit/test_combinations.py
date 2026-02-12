from itertools import combinations

import numpy as np
from combin.combinatorial import (
    combs,
    find_n,
    comb_rank_lex,
    comb_rank_colex,
    comb_unrank_colex,
    comb_unrank_lex,
    comb_unrank_lex_findn,
)
from combin.special import binom


def test_combs():
    n, k = 10, 3
    C_truth = np.array(list(combinations(range(n), k)))
    C_test = combs(n, k)
    assert np.allclose(C_truth, C_test)


def test_comb_lex_rank():
    n, k = 10, 3
    N = binom(n, k).item()
    C = combs(n, k)
    R = comb_rank_lex(C, n=n)
    assert np.allclose(R, np.arange(N))
    assert np.allclose(comb_unrank_lex(R, n=n, k=k), C), (
        "unranking does not match original lexicographically ordered combinations"
    )


def test_comb_colex_rank():
    n, k = 10, 3
    N = binom(n, k).item()
    C = combs(n, k)
    assert (
        len(np.unique(comb_rank_colex(C))) == N
    )  # TODO: only test injectivity, not actual ranking


def test_comb_unrank_lex_findn():
    n, k = 10, 3
    N = binom(n, k).item()
    C = combs(n, k)
    R = np.arange(N, dtype=np.int64)
    assert np.array_equal(comb_unrank_lex_findn(R, n=n, k=k), C)


def test_find_n():
    from itertools import product
    from math import comb
    from combin.special import binom
    from combin.combinatorial import find_n

    for n, k in product(np.arange(1, 1_500, 5), range(8)):
        # R = np.arange(0, comb(n,k))
        N = comb(n, k)
        R = np.arange(0, N, step=max(1, N // 1_000))
        n_est = np.array([find_n(r, k) for r in R])
        ## "however, the obtained value of n must not be greater than
        ##  the true value of the solution of this inequality"

        ## Ensure find_n(r,k) <= n for all reasonable C(n,k)
        assert np.all(n_est <= n), f"estimated n ({n_est}) > {n}"

        # R_lb = binom(n_est-1, k)
        # R_ub = binom(n_est+0, k)

        # assert np.all((R_lb <= R) & (R <= R_ub))


# %%
# def test_bc_lower_bound():
# 	n = np.arange(10)
# 	k = 5
# 	r = binom(n, k)
# 	# np.array([find_n(rr, k) for rr in r]) < n
# 	# find_k(45,2)
