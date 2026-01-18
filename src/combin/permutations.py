"""Contains functions for enumerating, ranking, sampling, and checking permutations."""

import math

import numpy as np
from numpy.typing import ArrayLike

from .utility import ensure


def perm_check(p: ArrayLike) -> bool:
	"""Check whether a vector is a valid permutation of 1..n.

	Args:
	    p: Iterable of integers.

	Returns:
	    True if p contains each integer 1..n exactly once, False otherwise.
	"""
	a = np.asarray(p, int)
	n = a.size
	if n < 1:
		return False
	if not np.all((1 <= a) & (a <= n)):
		return False
	return np.unique(a).size == n


def perm_count(n: int) -> int:
	"""Return the number of permutations of n elements, n!.

	Args:
	    n: Number of elements.

	Returns:
	    n factorial.
	"""
	ensure(n >= 0, "n must be nonnegative")
	return int(math.factorial(n))


def perm_inv(p: ArrayLike) -> np.ndarray:
	"""Compute the inverse of a 1-based permutation.

	Args:
	    p: Array of integers of shape (n,), permutation of 1..n.

	Returns:
	    Array of shape (n,) representing the inverse permutation.
	"""
	p = np.asarray(p, int)
	ensure(perm_check(p), "illegal permutation")
	n = p.size
	pinv = np.empty(n, int)
	pinv[p - 1] = np.arange(1, n + 1)
	return pinv


def perm_lex_rank(p: np.ndarray) -> int:
	"""Compute the lexicographic rank of a 1-based permutation.

	Rank is 0-based.

	Args:
	    p: Array of integers of shape (n,), permutation of 1..n.

	Returns:
	    Integer rank in [0, ..., n!-1].
	"""
	p = np.asarray(p, int)
	ensure(perm_check(p), "illegal permutation")
	n = p.size
	rank = 0
	a = p.copy()
	for j in range(n):
		rank += (a[j] - 1) * math.factorial(n - 1 - j)
		a[j + 1 :] -= a[j] < a[j + 1 :]
	return rank
