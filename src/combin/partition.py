import numpy as np
from typing import Optional

from .utility import ensure


def npart_table(n: int, npart: int) -> np.ndarray:
	r"""Return table of restricted partition counts P(i,j).

	Returns a table $P$ when $P(i,j)$ equals the number of ways to partition $i$ objects into $j$ nonempty parts.
	The resulting table satisfies the recurrence: 
	$$
	P(i,0)=\mathbf{1}_{\{i=0\}}, \qquad
	P(i,j)=
	\begin{cases}
		0, & i < j, \\
		P(i-1,j-1), & j \le i < 2j, \\
		P(i-1,j-1) + P(i-j,j), & i \ge 2j .
	\end{cases}
  $$

	Args:
	    n: Maximum total size.
	    npart: Maximum number of parts.

	Returns:
	    Integer array T with shape (n+1, npart+1) where T[i,j] = P(i,j).

	Raises:
	    ValueError: If n < 0 or npart < 0.
	"""
	ensure(n >= 0 and npart >= 0, "n and npart must be nonnegative")

	T = np.zeros((n + 1, npart + 1), dtype=int)
	T[0, 0] = 1

	for i in range(1, n + 1):
		j = np.arange(1, npart + 1)

		mask1 = i < j
		mask2 = (~mask1) & (i < 2 * j)
		mask3 = ~mask1 & ~mask2

		T[i, j[mask1]] = 0
		T[i, j[mask2]] = T[i - 1, j[mask2] - 1]
		T[i, j[mask3]] = T[i - 1, j[mask3] - 1] + T[i - j[mask3], j[mask3]]

	return T


def npart_enum(n: int, npart: int) -> int:
	r"""Return the number of partitions of $n$ into $n_{\text{part}}$ nonempty parts.

	Returns the value $P(n,n_{\text{part}})$ from the recurrence: 
	$$ 
	P(i,0)=\mathbf{1}_{\{i=0\}}, \qquad
	P(i,j)=
	\begin{cases}
		0, & i < j, \\
		P(i-1,j-1), & j \le i < 2j, \\
		P(i-1,j-1)+P(i-j,j), & i \ge 2j .
	\end{cases}
	$$

	Args:
			n: Size of the set.
			npart: Number of parts.

	Returns:
			Integer count $P(n,n_{\text{part}})$.
	"""
	if n <= 0 or npart <= 0 or n < npart:
		return 0
	T = npart_table(n, npart)
	return int(T[n, npart])


import numpy as np


def npart_rsf_lex_unrank(rank: int, n: int, npart: int) -> np.ndarray:
	r"""Unrank the lexicographic restricted standard-form partition.

	Returns the unique vector $a \in \mathbb{N}^{n_{\text{part}}}$ such that  

	1) $\sum_i a_i = n$  
	2) $1 \le a_1 \le a_2 \le \cdots \le a_{n_{\text{part}}}$  
	3) $a$ is the partition with the given lexicographic rank (0-based).

	The ranking is consistent with the recurrence for $P(i,j)$ (partitions of $i$ into $j$ nonempty parts):
	$$
	P(i,0)=\mathbf{1}_{\{i=0\}}, \qquad
	P(i,j)=
	\begin{cases}
		0, & i < j, \\
		P(i-1,j-1), & j \le i < 2j, \\
		P(i-1,j-1)+P(i-j,j), & i \ge 2j .
	\end{cases}
	$$

	Args:
			rank: Integer with $0 \le \text{rank} \le P(n,n_{\text{part}})$.
			n: Total sum.
			npart: Number of parts.

	Returns:
			Array of shape $(n_{\text{part}},)$ containing the partition.

	Raises:
			ValueError: If inputs are illegal.
	"""
	ensure(n > 0, "n must be > 0")
	ensure(npart >= 1 and n >= npart, "illegal npart")
	total = npart_enum(n, npart)
	ensure(rank >= 0 and rank <= total, "illegal rank")

	T = npart_table(n, npart)
	a = np.zeros(npart, dtype=int)
	ncopy = n
	k = npart
	r = rank

	while ncopy > 0:
		cutoff = T[ncopy - 1, k - 1]
		if r < cutoff:
			a[npart - k] += 1
			ncopy -= 1
			k -= 1
		else:
			a[npart - k :] += 1
			r -= cutoff
			ncopy -= k

	return a  # convert 0-based counts to standard 1-based parts
