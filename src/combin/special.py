import numpy as np


def binom(n: np.ndarray, k: np.ndarray) -> np.ndarray:
	r"""Binomial coefficient.

	Computes the binomial coefficient $C(n,k)$ given by:
	$$ {n \choose 2} = \frac{n!}{k!(n-k)!} = { n - 1 \choose k - 1} { n - 1 \choose k} $$
	All parameters `n` and `k` are broadcasted appropriately.
	Uses the multipliative formula to compute the coefficients.

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
	k = np.minimum(k, n - k)
	i = np.arange(1, k.max() + 1)
	terms = np.where(i <= k[..., None], (n[..., None] - i + 1) / i, 1.0)
	return np.rint(np.prod(terms, axis=-1)).astype(np.uint64)
