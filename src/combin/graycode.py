"""Contains functions for enumerating, ranking, sampling, and checking Gray codes."""

from typing import Iterable, Optional

import numpy as np

from .utility import ensure


def gray_code_check(t: Iterable[int]) -> bool:
	"""Return whether the vector represents a valid Gray-code bit pattern.

	Args:
	    t: Iterable of integers to check.

	Returns:
	    True if len(t) ≥ 1 and every entry is either 0 or 1, otherwise False.
	"""
	a = np.asarray(t)
	return a.size >= 1 and np.logical_or(a == 0, a == 1).all()


def gray_code_enum(n: int) -> int:
	"""Return the number of n-bit Gray codes, i.e., 2^n.

	Args:
	    n: Number of bits.

	Returns:
	    The count 2**n.
	"""
	return 1 << n


def gray_code_unrank(rank: int, n: int) -> np.ndarray:
	"""Return the n-bit Gray code at rank r (0-based).

	Args:
	    rank: Integer with 0 <= rank < 2**n.
	    n: Number of bits.

	Returns:
	    Array of shape (n,) with entries in {0,1}.

	Raises:
	    ValueError: If inputs are illegal.
	"""
	ensure(n >= 1, "n must be >= 1")
	ensure(rank >= 0 and rank < 2**n, "Illegal rank")
	g = rank ^ (rank >> 1)
	return ((g >> np.arange(n - 1, -1, -1)) & 1).astype(int)


def gray_code_random(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
	"""Sample a random n-bit Gray code uniformly.

	Args:
	    n: Number of bits (must be >= 1).
	    rng: Optional NumPy random generator.

	Returns:
	    Array of shape (n,) containing a random Gray code.

	Raises:
	    ValueError: If n < 1.
	"""
	ensure(n >= 1, "n must be >= 1")
	if rng is None:
		rng = np.random.default_rng()
	rank = rng.integers(1, gray_code_enum(n) + 1)
	return gray_code_unrank(rank, n)


def gray_code_successor(t: np.ndarray, rank: int) -> tuple[np.ndarray, int]:
	"""Compute the next Gray code in sequence.

	Args:
	    t: Current n-bit Gray vector (values must be 0 or 1).
	    rank: Current rank, where -1 initializes to all zeros and rank 0.

	Returns:
	    (next_vector, next_rank).

	Raises:
	    ValueError: If t is invalid or n < 1.
	"""
	a = np.asarray(t, dtype=int)
	n = a.size
	ensure(n >= 1, "n must be >= 1")

	## Reset if sentinel value given
	if rank == -1:
		return np.zeros(n, int), 0
	ensure(all((a == 0) | (a == 1)), "illegal Gray vector")

	w = int(a.sum())
	if w % 2 == 0:
		a[-1] ^= 1
		return a, rank + 1

	for i in range(n - 1, 0, -1):
		if a[i] == 1:
			a[i - 1] ^= 1
			return a, rank + 1

	return np.zeros(n, int), 0
