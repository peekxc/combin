import numpy as np
from combin.partition import npart_table, npart_enum, npart_rsf_lex_unrank


def test_npart_base_cases():
	T = npart_table(5, 5)
	assert T[0, 0] == 1
	assert np.all(T[0, 1:] == 0)
	assert np.all(T[1:, 0] == 0)


def test_npart_small_values():
	T = npart_table(6, 3)
	expected = np.array(
		[[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 2, 1], [0, 1, 2, 2], [0, 1, 3, 3]]
	)
	assert np.array_equal(T, expected)


def test_npart_monotone():
	n, k = 20, 10
	T = npart_table(n, k)
	assert np.all(T[1:, :] >= 0)


def test_rsf_unrank_properties():
	n, k = 8, 3
	total = npart_enum(n, k)
	parts = np.vstack([npart_rsf_lex_unrank(r, n, k) for r in range(total + 1)])
	assert np.all(parts.sum(axis=1) == n)
	assert np.all(np.diff(parts, axis=1) >= 0)
	assert np.unique(parts, axis=0).shape[0] == parts.shape[0]
