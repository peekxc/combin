import numpy as np
from combin.tree import (
	tree_enum,
	tree_check,
	pruefer_check,
	pruefer_enum,
	pruefer_rank,
	pruefer_successor,
	pruefer_to_tree,
	pruefer_unrank,
)


def test_tree_enum():
	assert tree_enum(0) == 0
	assert tree_enum(1) == 1
	assert tree_enum(2) == 1
	assert tree_enum(3) == 3**1
	assert tree_enum(4) == 4**2


def test_tree_check():
	edges = np.array([[1, 2], [2, 3]])
	assert tree_check(edges)
	# duplicate edge
	edges = np.array([[1, 2], [1, 2]])
	assert not tree_check(edges)
	# edge to nonexistent node
	edges = np.array([[1, 2], [2, 4]])
	assert not tree_check(edges)


def test_pruefer_check():
	assert pruefer_check(np.array([1, 2, 3]))
	assert not pruefer_check(np.array([0, 1, 2]))
	assert not pruefer_check(np.array([1, 2, 4]))  # n=4, 4 is valid; use n=3 for invalid
	assert not pruefer_check(np.array([]))  # n<3


def test_pruefer_enum():
	assert pruefer_enum(1) == 0
	assert pruefer_enum(2) == 1
	assert pruefer_enum(3) == 3**1
	assert pruefer_enum(4) == 4**2


def test_pruefer_rank_successor():
	n = 4
	p = np.ones(n - 2, int)
	rank = -1
	seen = []
	for _ in range(pruefer_enum(n)):
		p, rank = pruefer_successor(p, rank)
		seen.append(pruefer_rank(p))
	# should see all ranks 0..n^(n-2)-1 in order
	assert np.array_equal(np.sort(seen), np.arange(pruefer_enum(n)))


def test_pruefer_to_tree():
	# n=4, Prüfer sequence [1,1] corresponds to star 1-2,1-3,1-4
	p = np.array([1, 1])
	t = pruefer_to_tree(p)
	assert tree_check(t)
	assert t.shape == (3, 2)
	# edges include node 1
	assert np.any(t == 1)
	# sum of degrees should be 2*(n-1)
	deg = np.zeros(4, int)
	for e in t:
		deg[e[0] - 1] += 1
		deg[e[1] - 1] += 1
	assert deg.sum() == 6


def test_pruefer_unrank_rank_roundtrip():
	n = 6
	for rank in range(pruefer_enum(n)):
		p = pruefer_unrank(rank, n)
		assert p is not None
		assert pruefer_rank(p) == rank
