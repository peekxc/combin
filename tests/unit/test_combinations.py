import numpy as np
from itertools import combinations
from combin.special import binom
from combin.combinatorial import find_n, comb_lex_rank, combs


def test_combinations():
	n, k = 10, 3
	C_truth = np.array(list(combinations(range(n), k)))
	C_test = combs(n, k)
	assert np.allclose(C_truth, C_test)


def test_comb_lex_rank():
	n, k = 10, 3
	N = binom(n, k).item()
	C = combs(10, 3)
	assert np.allclose(comb_lex_rank(C, n=n), np.arange(N))`


def test_bc_lower_bound():
	n = np.arange(10)
	k = 5
	r = binom(n, k)
	# np.array([find_n(rr, k) for rr in r]) < n
	# find_k(45,2)
