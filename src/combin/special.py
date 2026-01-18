import numpy as np
from numpy.typing import ArrayLike


def binom(n: ArrayLike, k: ArrayLike) -> np.ndarray:
	r"""Binomial coefficient.

	Computes the binomial coefficient $C(n,k)$ given by:
	$$ {n \choose 2} = \frac{n!}{k!(n-k)!} = { n - 1 \choose k - 1} { n - 1 \choose k} $$
	All parameters `n` and `k` are broadcasted appropriately. Uses $O(|n| \cdot |k|)$ memory.
	Uses the multiplicative formula to compute the coefficients.

	Parameters:
		n: numerator of the binomial coefficient.

	Examples:
		>>> binom(10, 2)
		45
		>>> binom(range(10), 2)

	Pyodide:
		<div class="language-python">
			<code id="code-snippet-1">print("hello")</code>
			<button class="run-py-btn" data-code-id="code-snippet-1">Run Code</button>
			<div id="code-snippet-1-output" class="py-output"></div>
		</div>
	"""
	n = np.asarray(n)
	k = np.asarray(k)
	k = np.minimum(k, n - k)
	i = np.arange(1, k.max() + 1)
	terms = np.where(i <= k[..., None], (n[..., None] - i + 1) / i, 1.0)
	result = np.rint(np.prod(terms, axis=-1)).astype(np.uint64)
	# return result if np.size(result) > 1 else result.item()
	return np.atleast_1d(result)
