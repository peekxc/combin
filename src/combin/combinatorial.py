import numpy as np 
from typing import Iterable, Container, Union
from itertools import combinations
from numbers import Integral
from math import floor, ceil, comb, factorial
import _combinatorial
from more_itertools import collapse, spy, first_true

def _comb_unrank_lex(r: int, n: int, k: int):
  result = [0]*k
  x = 1
  for i in range(1, k+1):
    while(r >= comb(n-x, k-i)):
      r -= comb(n-x, k-i)
      x += 1
    result[i-1] = (x - 1)
    x += 1
  return tuple(result)

def _comb_rank_lex(c: Iterable, n: int) -> int:
  c = tuple(sorted(c))
  k = len(c)
  index = sum([comb(int(n-ci-1),int(k-i)) for i,ci in enumerate(c)])
  return int(comb(n, k) - index - 1)

def _comb_rank_colex(c: Iterable) -> int:
  c = tuple(sorted(c))
  k = len(c)
  return sum([comb(ci,k-i) for i,ci in enumerate(reversed(c))])

def _comb_unrank_colex(r: int, k: int) -> tuple:
  """
  Unranks a k-combinations rank 'r' back into the original combination in colex order.
  
  This function uses a simple unranking process for testing purposes. For more efficient unranking, see `comb_to_rank`. 
  
  Assuming comb takes O(min(n-k, k)) time, this function takes O(k^2 (n - k)) time, where n is 
  the largest integer satisfying 0 <= r < C(n,k).
  """
  c = [0]*k
  for i in reversed(range(1, k+1)): # O(k)
    m = i
    while r >= comb(m,i):           # O(n) as min comparisons == n − m + 1 (when i = m), max comparisons is n (when i = 1);
      m += 1
    c[i-1] = m-1
    r -= comb(m-1,i)                # comb is O(min(n-k, k)) ~ O(k)
  return tuple(c)

def comb_to_rank(
    C: Union[Iterable[tuple], np.ndarray], 
    k: int = None,
    n: int = None, 
    order: str = ["colex", "lex"]
  ) -> np.ndarray:
  """
  Ranks k-combinations to integer ranks in either lexicographic or colexicographical order.
  
  Parameters:
    C : combination, Iterable of combinations, or array of integers.
    k : size of each combination (broadcastable). If not supplied, 'C' must be 2-dimensional. 
    n : cardinality of the set (lex order only).
    order : the bijection to use.
  
  Returns:
    ndarray : integer ranks of the combinations.

  From: Unranking Small Combinations of a Large Set in Co-Lexicographic Order
  """
  n = int(n) if n is not None else None 
  assert isinstance(C, np.ndarray) or isinstance(C, Iterable), "Supply numpy array for vectorized version"
  colex_order = (order == ["colex", "lex"] or order == "colex")
  assert colex_order or n is not None, "Set cardinality 'n' must be supplied for lexicographical ranking." # note we need n for colex too!
  rank_comb_ = lambda c: _comb_rank_colex(c) if colex_order else _comb_rank_lex(c, n)
  
  if isinstance(C, np.ndarray) or isinstance(C, Container):
    if isinstance(C, np.ndarray):
      n = (np.max(C)+1) if n is None else n
      if C.ndim == 1: 
        assert k is None or isinstance(k, Integral), "array based ranking not supported yet"
        return rank_comb_(C)
      assert C.ndim == 2, "Can only handle array of dimensionality 2."
      C.sort(axis=1)
      C = np.fliplr(C) if colex_order else C
      C = np.array(C, order='C', copy=True) if not C.flags['OWNDATA'] else C # copy if a view was given
      assert C.flags['OWNDATA'] and C.flags['C_CONTIGUOUS'] and C.flags['ALIGNED'], "ndarray must not be a view"
      ranks = _combinatorial.rank_combs_sorted(C, n, colex_order)
      return ranks
    else:
      el, C = spy(C)
      if isinstance(el[0], Integral): return rank_comb_(C)
      assert isinstance(el[0], Container), "Elements must be containers."
      C = list(C)
      n = np.max([max(c) for c in C]) + 1 if n is None else n
      ranks = _combinatorial.rank_combs_unsorted(C, n, colex_order)
      return ranks
  elif isinstance(C, Iterable):
    # return comb_rank_colex(C) if colex_order else comb_rank_lex(C, n)
    el, C = spy(C)
    if colex_order:
      return _comb_rank_colex(C) if isinstance(el[0], Integral) else [_comb_rank_colex(c) for c in C]
    else:
      assert n is not None, "Cardinality of set must be supplied for lexicographical ranking"
      return _comb_rank_lex(C, n) if isinstance(el[0], Integral) else [_comb_rank_lex(c, n) for c in C]
  else: 
    raise ValueError(f"Invalid combination type '{type(C)}' supplied")

def rank_to_comb(
    R: Union[np.ndarray, Iterable, Integral], 
    k: Union[int, Iterable], 
    n: int = None, 
    order: str = ["colex", "lex"]
  ) -> np.ndarray:
  """
  Unranks integer ranks to  k-combinations in either lexicographic or colexicographical order.
  
  Parameters:
    R : Iterable of integer ranks 
    k : size of combination to unrank to, as either an integer or an array of integers.
    n : cardinality of the set (only required for lex order)
    order : the bijection to use
  
  Returns:
    list : k-combinations derived from R
  """
  n = int(n) if n is not None else None 
  colex_order = (order == ["colex", "lex"] or order == "colex")
  assert colex_order or n is not None, "Set cardinality 'n' must be supplied for lexicographical ranking."
  if isinstance(R, Integral):
    return _comb_unrank_colex(R, k=k) if colex_order else _comb_unrank_lex(R, k=k, n=n)
  elif isinstance(R, np.ndarray) and isinstance(k, Integral):
    assert R.ndim == 1, "Ranks must be one-dimensional array."
    R = R.astype(np.uint64) if (R.dtype != np.uint64) else R
    R = np.array(R, order='C', copy=True) if not R.flags['OWNDATA'] else R # copy if view was given
    n = inverse_choose(np.max(R), k, exact=False) if n is None else n
    n = max(n, k) # never let n be less than number of things we're choosing
    C = np.empty(shape=(len(R), k), dtype=np.uint16) ## TODO: change to np.min_scalar_type(n)
    _combinatorial.unrank_combs(R, n, k, colex_order, C)
    C.sort(axis=1) ## TODO: consider adding a flag 
    return C 
    # else: 
    #   assert isinstance(k, np.ndarray), "If R is given as an ndarray and k is a sequence, k must also be an array."
    #   K = np.array(k, order='C', copy=True) if not k.flags['OWNDATA'] else k # copy if view was given
    #   assert len(K) == len(R), "If K is given as a sequence, it must have the same length as the ranks sequence 'R'."
    #   n = inverse_choose(np.max(R[K == np.max(K)]), k, exact=False) if n is None else n
    #   C = np.empty(shape=np.sum(K), dtype=np.uint16)
    #   _combinatorial.unrank_combs(R, n, K, colex_order, C)
    #   C.sort(axis=1) ## TODO: consider adding a flag 
  elif isinstance(R, Iterable):
    R = np.fromiter(R, dtype=np.uint64)
    K = np.array([k]*len(R), dtype=np.uint16) if isinstance(k, Integral) else np.array(k).astype(np.uint16)
    n = inverse_choose(np.max(R), np.max(K), exact=False) if n is None else n
    n = max(n, np.max(K)) # never let n be less than number of things we're choosing
    assert len(K) == len(R), "If given as a sequence, k must match the length of the ranks sequence 'R'."
    C = np.zeros(K.sum(), dtype=np.uint16)
    if colex_order:
      _combinatorial.unrank_combs_k(R, n, K, K.max(), True, C)
      return np.array_split(C, np.cumsum(K)[:-1])
      # return [_comb_unrank_colex(r, k) for r,k in zip(R,K)]
    else:
      assert n is not None, "Cardinality of set must be supplied for lexicographical ranking"
      _combinatorial.unrank_combs_k(R, n, K, K.max(), False, C)
      return np.array_split(C, np.cumsum(K)[:-1])
      # return [_comb_unrank_lex(r, n, k) for r,k in zip(R,K)]
  else:
    raise ValueError(f"Unknown input type for ranks '{type(R)}'")
  
def inverse_choose(x: int, k: int, exact: bool = True):
  """Inverse binomial coefficient (approximately). 

  This function *attempts* to find the integer _n_ such that binom(n,k) = x, where _binom_ is the binomial coefficient: 

  binom(n,k) := n!/(k! * (n-k)!)

  For k <= 2, a logartihmic numpy-based approach is used and the result is exact. 
  For k > 2 and x <= 10e7, an linear-search is used based on tight bounds and the result is exact. 
  For k > 2 and x > 10e7; an iterative approach is used based on loose bounds from the formula from this stack exchange post: 

  https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
  """
  assert x >= 0, "x must be a non-negative integer" 
  if k == 0: return(1)
  if k == 1: return(x)
  if k == 2:
    rng = np.arange(np.floor(np.sqrt(2*x)), np.ceil(np.sqrt(2*x)+2) + 1, dtype=np.uint64)
    final_n = rng[np.searchsorted((rng * (rng - 1) / 2), x)]
    if comb(final_n, 2) == x or not exact:
      return final_n
    raise ValueError(f"Failed to invert C(n,{k}) = {x}")
    # return int(rng[x == (rng * (rng - 1) / 2)])
  else:
    # From: https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
    if x < 10**7:
      lb = (factorial(k)*x)**(1/k)
      potential_n = np.arange(floor(lb), ceil(lb+k)+1)
      comb_cand = np.array([comb(n, k) for n in potential_n])
      ub_ind = np.searchsorted(comb_cand, x)
      if exact and x != comb_cand[ub_ind]:
        raise ValueError(f"Unable to invert 'x' = {x}")
      elif exact and x == comb_cand[ub_ind]:
        return potential_n[ub_ind]
      else: # not exact
        if ub_ind >= len(comb_cand):
          raise ValueError(f"Low/upper bounds calculations do not hold for 'x' = {x}")
        return potential_n[ub_ind]
    else:
      lb = np.floor((4**k)/(2*k + 1))
      C, n = factorial(k)*x, 1
      while n**k < C: n = n*2
      m = first_true((c**k for c in range(1, n+1)), pred=lambda c: c**k >= C)
      potential_n = range(min([m, 2*k]), m+k+1)
      if len(potential_n) == 0: 
        raise ValueError(f"Failed to invert C(n,{k}) = {x}")
      final_n = first_true(potential_n, default = -1, pred = lambda n: comb(n,k) == x)
      if final_n != -1:
        return final_n
      else: 
        from scipy.optimize import minimize_scalar
        binom_loss = lambda n: np.abs(comb(int(n), k) - x)
        res = minimize_scalar(binom_loss, bounds=(comb(2*k, k), x))
        n1, n2 = int(np.floor(res.x)), int(np.ceil(res.x))
        if comb(n1,k) == x: return n1 
        if comb(n2,k) == x: return n2 
        raise ValueError(f"Failed to invert C(n,{k}) = {x}")