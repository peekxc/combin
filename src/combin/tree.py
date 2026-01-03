import numpy as np


from .utility import ensure


def pruefer_check(p: np.ndarray) -> bool:
	"""Check whether a sequence is a valid Prüfer code for n >= 3.

	Args:
	    p: Array of shape (n-2,) with 1-based node labels.

	Returns:
	    True if valid, False otherwise.
	"""
	p = np.asarray(p, int)
	n = p.size + 2
	return False if n < 3 else all((1 <= p) & (p <= n))


def pruefer_enum(n: int) -> int:
	"""Return the number of labeled trees on n nodes via Prüfer sequences.

	Args:
	    n: Number of nodes.

	Returns:
	    Integer count of Prüfer sequences (n^(n-2) for n>=3).
	"""
	if n < 2:
		return 0
	if n == 2:
		return 1
	return n ** (n - 2)


def pruefer_rank(p: np.ndarray) -> int:
	"""Compute the 0-based lexicographic rank of a Prüfer sequence.

	Args:
	    p: Array of shape (n-2,) with 1-based node labels.

	Returns:
	    Integer rank in 0..n**(n-2)-1
	"""
	p = np.asarray(p, int)
	ensure(pruefer_check(p), "illegal Prüfer sequence")
	n = p.size + 2
	rank = 0
	k = 1
	for i in reversed(range(n - 2)):
		rank += k * (p[i] - 1)
		k *= n
	return rank


def pruefer_to_tree(p: np.ndarray) -> np.ndarray:
	"""Convert a Prüfer sequence to its corresponding tree as an edge list.

	Args:
	    p: Array of shape (n-2,) with 1-based node labels.

	Returns:
	    Array of shape (n-1, 2) with 1-based edges representing the tree.
	"""
	p = np.asarray(p, int)
	if not pruefer_check(p):
		raise ValueError("illegal Prüfer sequence")
	n = p.size + 2
	t = np.zeros((n - 1, 2), int)

	# degrees
	d = np.ones(n, int)
	np.add.at(d, p - 1, 1)

	for i in range(1, n):
		# find the smallest leaf
		leaves = np.where(d == 1)[0]
		x = leaves[-1]  # in C code, x starts from n downward
		if i == n - 1:
			y = 0  # node 1 in 0-based index
		else:
			y = p[i - 1] - 1
		t[i - 1, :] = [x + 1, y + 1]  # convert back to 1-based
		d[x] -= 1
		d[y] -= 1

	return t


def pruefer_successor(p: np.ndarray, rank: int) -> tuple[np.ndarray, int]:
	"""Return the next Prüfer sequence in lexicographic order and its rank.

	If rank==-1, initializes p to all ones.

	Args:
	    p: Array of shape (n-2,) with 1-based node labels.
	    rank: Current rank, or -1 to initialize.

	Returns:
	    Tuple of (next Prüfer sequence array, next rank)
	"""
	p = np.asarray(p, int)
	ensure(pruefer_check(p), "Illegal Prüfer sequence")

	n = p.size + 2
	p_next = p.copy()
	if rank == -1:
		p_next[:] = 1
		return p_next, 0

	j = n - 2
	while j > 0 and p_next[j - 1] == n:
		j -= 1

	if j != 0:
		p_next[j - 1] += 1
		p_next[j:] = 1
		rank += 1
	else:
		p_next[:] = 1
		rank = 0

	return p_next, rank


def pruefer_unrank(rank: int, n: int) -> np.ndarray | None:
	"""Return the Prüfer sequence corresponding to a 0-based lexicographic rank.

	Args:
	    rank: Integer rank, 0 <= rank < n**(n-2) for n>=3.
	    n: Number of nodes.

	Returns:
	    Array of shape (n-2,) with 1-based node labels, or None if n<3.
	"""
	if n < 1:
		raise ValueError("n must be >= 1")
	if n < 3:
		return None

	ncode = pruefer_enum(n)
	if rank < 0 or rank >= ncode:
		raise ValueError("illegal rank")

	p = np.empty(n - 2, int)
	r = rank
	for i in reversed(range(n - 2)):
		p[i] = (r % n) + 1
		r = (r - (p[i] - 1)) // n
	return p


def tree_check(t: np.ndarray) -> bool:
	"""Check whether an n-1 by 2 edge array represents a valid tree on $n$ nodes.

	Args:
	    t: Array of shape (n-1, 2) with 1-based node labels.

	Returns:
	    whether $t$ encodes a valid tree.
	"""
	t = np.asarray(t, int)
	if t.ndim != 2 or t.shape[1] != 2:
		return False
	n = t.shape[0] + 1
	if n < 1 or np.any((t < 1) | (t > n)):
		return False

	# Compute degrees
	d = np.zeros(n, int)
	for i in range(2):
		np.add.at(d, t[:, i] - 1, 1)

	t_copy = t.copy()
	for _ in range(n - 1):
		# find a leaf
		leaves = np.where(d == 1)[0]
		if leaves.size == 0:
			return False
		x = leaves[0]

		# find an edge containing x
		mask0 = t_copy[:, 0] == x + 1
		mask1 = t_copy[:, 1] == x + 1
		idxs = np.where(mask0 | mask1)[0]
		if idxs.size == 0:
			return False
		j = idxs[0]

		# y is the neighbor
		y = t_copy[j, 1] - 1 if t_copy[j, 0] - 1 == x else t_copy[j, 0] - 1

		# remove edge
		d[x] -= 1
		d[y] -= 1
		t_copy[j, :] = -t_copy[j, :]

	return True


def tree_enum(n: int) -> int:
	"""Return the number of labeled trees on n nodes.

	Args:
	    n: Number of nodes.

	Returns:
	    Integer count.
	"""
	if n < 1:
		return 0
	if n in {1, 2}:
		return 1
	return n ** (n - 2)
