import numpy as np
from combin.permutations import perm_check, perm_count, perm_lex_rank, perm_inv


def test_perm_check_valid():
	assert perm_check([1, 2, 3])
	assert perm_check(np.array([3, 1, 2]))
	assert not perm_check([1, 2, 3, 3])


def test_perm_check_invalid():
	assert not perm_check([])
	assert not perm_check([0, 1, 2])
	assert not perm_check([1, 2, 2])
	assert not perm_check([1, 2, 4])


def test_perm_count():
	assert perm_count(0) == 1
	assert perm_count(3) == 6


def test_perm_inv():
	p = np.array([3, 1, 2])
	pinv = perm_inv(p)
	assert np.array_equal(p[pinv - 1], np.arange(1, 4))


def test_perm_lex_rank():
	n = 3
	perms = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
	for r, perm in enumerate(perms):
		assert perm_lex_rank(np.array(perm)) == r
