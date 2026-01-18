import numpy as np
from combin.graycode import (
	ensure,
	gray_code_check,
	gray_code_enum,
	gray_code_random,
	gray_code_successor,
	gray_code_unrank,
)


def collect_by_successor(n: int) -> np.ndarray:
	t = np.zeros(n, int)
	rank = -1
	out = np.zeros((1 << n, n), int)
	for k in range(1 << n):
		t, rank = gray_code_successor(t, rank)
		out[k] = t
	return out


def collect_by_unrank(n: int) -> np.ndarray:
	ranks = np.arange(1 << n, dtype=int)
	return np.vstack([gray_code_unrank(int(r), n) for r in ranks])


def test_gray_code_unique_and_complete():
	n = 6
	G = collect_by_successor(n)
	assert G.shape == (1 << n, n)
	assert np.unique(G, axis=0).shape[0] == (1 << n)


def test_successor_matches_unrank():
	n = 7
	G1 = collect_by_successor(n)
	G2 = collect_by_unrank(n)
	assert np.array_equal(G1, G2)


def test_successor_has_hamming_1():
	n = 8
	G = collect_by_successor(n)
	diff = np.sum(G[1:] ^ G[:-1], axis=1)
	wrap = np.sum(G[0] ^ G[-1])
	assert np.all(diff == 1)
	assert wrap == 1


def test_gray_code_check():
	assert gray_code_check([0, 1, 0])
	assert not gray_code_check([])
	assert not gray_code_check([0, 2, 1])


def test_gray_code_enum():
	assert gray_code_enum(0) == 2**0
	assert gray_code_enum(3) == 2**3
	assert gray_code_enum(5) == 2**5


def test_gray_code_random_generates_valid():
	rng = np.random.default_rng(0)
	t = gray_code_random(4, rng)
	assert t.shape == (4,)
	assert set(t.tolist()) <= {0, 1}
