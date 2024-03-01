import numpy as np
from math import comb, ceil, floor, sqrt
from itertools import combinations, product 
from functools import partial
from typing import Callable
from combin import comb_to_rank, rank_to_comb, inverse_choose
from combin.combinatorial import _combinatorial, _comb_unrank_colex, _comb_rank_colex
# print(__file__)

def Comb(x: np.ndarray, k: int) -> np.ndarray:
  return np.array([comb(xi, k) for xi in x]).astype(np.int64)

def test_basic():
  n, k = 10,3
  c1, c2 = [0,1,2], [0,1,3]
  ranks = np.array([0,1], dtype=np.uint64) 
  combs_test_cpp = rank_to_comb(ranks, k=k, order="colex")
  assert np.all(np.array([c1,c2], dtype=np.uint64) == combs_test_cpp)

  n, k = 20, 5
  c = [2,4,13,15,19]
  r = _comb_rank_colex(c)
  ranks = np.array([r], dtype=np.uint64) 
  assert np.all(rank_to_comb(ranks, k=k, n=n,order="colex") == np.array([[c]], dtype=np.uint64))

  ## carefully crafted case for find_k
  n, k = 20, 4
  r = comb_to_rank(np.array([[0,1,2,6]], dtype=np.uint16), k=k, n=n)  # 15 
  c = rank_to_comb(r, k=k, n=n)
  assert np.all(c == np.array([[0,1,2,6]], dtype=np.uint16))

  ## Should be in reverse lex order
  n, k = 20, 3
  r = np.array([34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 0], dtype=np.uint64)
  c = rank_to_comb(r, k=k, n=n, order='lex')
  assert np.all(c[-1,:] == [0,1,2]) and np.all(c[0,:] == [0,2,19])
  
  ## Ensure same results across non-array types
  c = [[0,1,2], [0,3,4]]
  assert np.all(comb_to_rank(c, k=3, order='colex') == np.array([0,7]))
  assert np.all(comb_to_rank(np.array(c), k=3, order='colex') == np.array([0,7]))

  ## Test you can pass varying k 
  K = [3]*len(r)
  C_vary = rank_to_comb(r, K, n=n, order='lex')
  C_vect = rank_to_comb(r, k=3, n=n, order='lex')
  assert np.all(np.array(C_vary) == C_vect)
  assert list(map(tuple, rank_to_comb([0,3,4], [1,2,3]))) == [(0,), (0, 3), (0, 1, 4)]

  ## Test sorted colex 0-based shortcut ranking works
  C = np.array([[3, 2],[1, 0], [3, 1], [2, 0], [3, 0], [2, 1]], dtype=np.uint16)
  r_truth = (Comb(C[:,0], 2) + Comb(C[:,1], 1)).astype(np.uint16)
  r_test = comb_to_rank(C, n=4, order='colex')
  assert np.allclose(r_truth, r_test)

def test_combs():
  assert all(_combinatorial.comb([1,2,3],[1,2,3]) == np.array([1,1,1]))
  assert all(_combinatorial.comb([1,2,3],[0,0,0]) == np.array([1,1,1]))
  max_n, max_k = 100, 5
  all_n, all_k, res = [], [], []
  for n,k in product(range(max_n), range(max_k)):
    all_n.append(n)
    all_k.append(k)
    res.append(comb(n,k))
  assert np.allclose(_combinatorial.comb(all_n, all_k, max_n, max_k), np.array(res))

def test_numpy_ranking():
  n, k = 10, 3
  combs = np.array(list(combinations(range(n), k)), dtype=np.uint16)
  assert all(np.equal(comb_to_rank(combs, k, n, 'lex'), np.arange(120, dtype=np.uint64)))
  assert all(np.equal(comb_to_rank([tuple(c) for c in combs], n=n, order='lex'), np.arange(120, dtype=np.uint64)))
  assert all(comb_to_rank(np.fliplr(combs), order='colex') == comb_to_rank(combs, order='colex') )
  assert all(np.array(comb_to_rank(iter(combs))) == comb_to_rank(combs))
  assert isinstance(comb_to_rank(combs), np.ndarray)

def test_colex():
  n, k = 10, 3
  ranks = np.array([comb_to_rank(c) for c in combinations(range(n), k)])
  assert all(np.sort(ranks) == np.arange(comb(n,k))), "Colex ranking is not unique / does not form bijection"
  ranks2 = np.array([comb_to_rank(reversed(c)) for c in combinations(range(n), k)])
  assert all(ranks == ranks2), "Ranking is not order-invariant"
  combs_test = np.array([rank_to_comb(r, k) for r in ranks])
  combs_truth = np.array(list(combinations(range(n),k)))
  combs_test_cpp = rank_to_comb(ranks, k=k, order="colex")
  assert all(np.ravel(combs_test == combs_truth)), "Colex unranking invalid"
  assert all(np.ravel(combs_truth == combs_test_cpp)), "Colex unranking invalid"

def test_array_conversion():
  x = np.array(rank_to_comb([0,1,2], k=2))
  assert np.all(x == np.array([[0,1], [0,2], [1,2]], dtype=np.uint16))

def test_unranking_raw():
  n = 10
  K = np.random.choice([1,2,3], size=25, replace=True).astype(np.uint16)
  R = np.ravel(np.array([np.random.choice(np.arange(comb(n, k)), size=1) for k in K])).astype(np.uint64)
  
  out = np.zeros(K.sum(), dtype=np.uint16)
  _combinatorial.unrank_combs_k(R, n, K, K.max(), True, out)
  test = np.array_split(out, np.cumsum(K)[:-1])
  truth = [rank_to_comb(r, k=k, n=n, order='colex') for r,k in zip(R,K)]
  assert all([tuple(t1) == tuple(t2) for (t1,t2) in zip(test,truth)])

  truth_lex = [rank_to_comb(r, k=k, n=n, order='lex') for r,k in zip(R,K)]
  _combinatorial.unrank_combs_k(R, n, K, K.max(), False, out)  
  test = np.array_split(out, np.cumsum(K)[:-1])
  assert all([tuple(t1) == tuple(t2) for (t1,t2) in zip(test,truth_lex)])

def test_lex():
  n, k = 10, 3
  ranks = np.array([comb_to_rank(c, k, n, "lex") for c in combinations(range(n), k)])
  assert all(ranks == np.arange(comb(n,k))), "Lex ranking is not unique / does not form bijection"
  ranks2 = np.array([comb_to_rank(reversed(c), k, n, "lex") for c in combinations(range(n), k)])
  assert all(ranks == ranks2), "Ranking is not order-invariant"
  combs_test = np.array([rank_to_comb(r, k, n, "lex") for r in ranks])
  combs_truth = np.array(list(combinations(range(n),k)))
  combs_truth2 = rank_to_comb(ranks, k, n, "lex")
  assert all((combs_test == combs_truth).flatten()), "Lex unranking invalid"
  assert all((combs_truth == combs_truth2).flatten()), "Lex unranking invalid"

def test_api():
  assert np.all(np.array(rank_to_comb([0,1,2], k=3)) == np.array([[0,1,2],[0,1,3],[0,2,3]], dtype=np.uint16))
  assert all(comb_to_rank([(0,1,2), (0,1,3), (0,2,3)], n=4) == [0,1,2])
  n = 20
  for k in range(1, 5):
    combs = list(combinations(range(n), k))
    C = rank_to_comb(comb_to_rank(combs, k=k, n=n), k=k, n=n)
    assert all([tuple(s) == tuple(c) for s,c in zip(combs, C)])

def test_inverse():
  from math import comb
  assert inverse_choose(10, 2) == 5
  assert inverse_choose(45, 2) == 10
  comb2 = partial(lambda x: comb(x, 2))
  comb3 = partial(lambda x: comb(x, 3))
  N = [10, 12, 16, 35, 48, 78, 101, 240, 125070]
  for n, x in zip(N, map(comb2, N)):
    assert inverse_choose(x, 2) == n
  for n, x in zip(N, map(comb3, N)):
    assert inverse_choose(x, 3) == n

# def test_facet_enumeration():
#   from combin.combinatorial import _combinatorial
#   n, k = 10, 3
#   r = comb_to_rank([0,4,7], n = 10)
#   _combinatorial.facet_ranks(r, k-1, n)


N_EXP = 0
N_BIN = 0

## Binary / exponential search to find the largest index i \in [bottom, top] satisfying pred(i) == True, or bottom otherwise if all False
## Assumes predicate evaluates as True, True, ..., True, False, False, ..., False on the range [bottom, top]
def get_max(top: int, bottom: int, pred: Callable, exp: bool = False): 
  global N_EXP 
  global N_BIN
  if not pred(bottom): 
    return bottom
  size = (top - bottom)
  if exp: 
    inc = 1 # should be a increment
    while (bottom + inc) < top and pred(bottom + inc):
      N_EXP += 1
      inc *= 2
      # print(f"e: {bottom + inc}")
    bottom, top = np.max([bottom, (bottom + inc) // 2]), np.min([bottom + inc, top])
    size = (top - bottom)
  # if not pred(top):
  while size > 0:
    step = size >> 1
    mid = top - step
    if not pred(mid):
      N_BIN += 1
      top = mid - 1
      size -= step + 1
    else:
      size = step
    # print(f"b: {mid}")
  return top

## Finds the lower bound on an integer k <= m satisfying that comb(k, m) <= r
# "This function should calculate the approximate value of k for solving the inequality (3); 
# however, the obtained value of k must not be greater than the true value of the solution of this inequality"
def find_k(r: int, m: int):
  assert(m > 0)
  if r == 0: 
    return max(m - 1, 0) # note that m-1 is the lower bound here
  elif (m == 1):
    return r
  elif (m == 2):
    return max(ceil((1.0+sqrt(1+8*r))/2) - 1, 0)
  elif (m == 3):
    return max(ceil(pow(6*r, 1/3)) - 1, 0)
  else:
    return max(m - 1, 0)

## Should return the largest c satisfying comb(c-1, m) <= r < comb(c, m)
def get_max_vertex(r: np.ndarray, m: int, n: int, exp: bool = False, use_lb: bool = False, C: int = 0):
  k_lb = find_k(r,m) if use_lb else max(m - 1, 0)
  assert comb(k_lb, m) <= r
  if C == 0: 
    c = get_max(n, k_lb, lambda c: comb(c, m) <= r, exp=exp)
  elif C == 1: 
    if r < comb(k_lb+1, m):
      return k_lb + 1
    c = get_max(n, k_lb, lambda c: comb(c, m) <= r, exp=exp)
  elif C == 2:
    if r < comb(k_lb+1, m): return k_lb + 1
    if r < comb(k_lb+2, m): return k_lb + 2
    c = get_max(n, k_lb, lambda c: comb(c, m) <= r, exp=exp)
  else: 
    if r < comb(k_lb+1, m): return k_lb + 1
    if r < comb(k_lb+2, m): return k_lb + 2
    if r < comb(k_lb+2, m): return k_lb + 3
    c = get_max(n, k_lb, lambda c: comb(c, m) <= r, exp=exp)
  assert comb(c, m) <= r and r < comb(c + 1, m) 
  return c+1 

## Unranks each m-combination of a n-set into o
def unrank_colex(r: int, n: int, m: int, exp: bool = False, use_lb: bool = False, C: int = 0) -> np.ndarray:
  out = np.ones(m) * -1
  for ii, m_ in enumerate(np.flip(np.arange(1, m+1))):
    K = get_max_vertex(r, m_, n, exp, use_lb, C) # k satisfying comb(k-1,m) <= r < comb(k, m)
    out[ii] = K-1    # this differs from the paper because we want 0-based indices
    r -= comb(K-1, m_)
  return out

def find_k_naive(r: int, m: int):
  k = m 
  while r >= comb(k, m):
    k += 1
  assert comb(k-1, m) <= r and r < comb(k, m) # test condition
  return k

def test_get_max():
  assert get_max(100, 0, lambda x: x < 52) == 51
  assert get_max(10, 0, lambda x: x > 0) == 0

  # n, m = 1350, 3
  n,m = 100, 3
  R = np.arange(comb(n,m))
  ## "When k = m, we search the interval that contains the rank r among the following binomial coefficients"
  potential_K = np.arange(m, n+1)
  BR = np.array([comb(c, m) for c in potential_K]) # the range 

  ## Deduce maximum error 
  test_K = potential_K[np.searchsorted(BR, R, side='right')]
  cand_R = np.random.choice(R, size=1000, replace=False)
  print(f"Max find K error: {np.max(np.abs(test_K[cand_R] - np.array([find_k(r, m) for r in cand_R])))}")

  ## "If we know the rank r of an m-combination of an n-set and the solution for
  ## comb(k-1, m) <= r < comb(k, m)
  ## then the m-th selected element in the combination is equal to k
  true_K = potential_K[np.array([np.searchsorted(r < BR, True) for r in R])]
  test_K = np.array([find_k_naive(r, m) for r in R])
  assert np.all(potential_K[np.searchsorted(BR, R, side='right')] == true_K)
  assert np.all(test_K == true_K)

  ## Ensure find_K returns a lower bound
  assert np.all([find_k(r,m) <= true_K[r] for r in R])
  assert np.max([np.abs(find_k(r,m) - true_K[r]) for r in R]) <= 3

  ## Ensure the range is indeed k \in [m,n]
  assert np.all(np.logical_and(true_K >= m, true_K <= n))
  assert np.min(true_K) == m and np.max(true_K) == n
  
  ## Ensure get_max works as expected for both binary and exponential search
  status = np.zeros(n+1, dtype=bool)
  status[:m] = True
  for i in range(m, n):
    status[i] = True
    assert i == get_max(top=n, bottom=m, pred=lambda w: status[w], exp = False) 

  status = np.zeros(n+1, dtype=bool)
  status[:m] = True
  for i in range(m, n):
    status[i] = True
    assert i == get_max(top=n, bottom=m, pred=lambda w: status[w], exp = True) 

  ## Ensure get_max works on range [m-1, n] plainly
  for r in R:
    k1 = get_max(top=n, bottom=m-1, pred=lambda w: comb(w, m) <= r, exp = False) 
    k2 = get_max(top=n, bottom=m-1, pred=lambda w: comb(w, m) <= r, exp = True)
    assert comb(k1, m) <= r and r < comb(k1+1, m)
    assert comb(k2, m) <= r and r < comb(k2+1, m)
    assert k1 == k2

  ## Ensure get_max works after incorporating find_k
  for r in R: 
    k_lb = find_k(r, m)
    assert comb(k_lb, m) <= r
    k1 = get_max(top=n, bottom=k_lb, pred=lambda w: comb(w, m) <= r, exp = False) 
    k2 = get_max(top=n, bottom=k_lb, pred=lambda w: comb(w, m) <= r, exp = True)
    assert comb(k1, m) <= r and r < comb(k1+1, m)
    assert comb(k2, m) <= r and r < comb(k2+1, m)
    assert k1 == k2
    true_k = find_k_naive(r, m)
    assert true_k == (k1 + 1)

  ## Finally, test we match the ground truth
  for r in R:
    simplex_true = np.sort(rank_to_comb(r, m, order='colex', n=n))
    simplex_test = np.sort(unrank_colex(r, n, m))
    np.allclose(simplex_test, simplex_true)

  ## Check out performance 
  C = 3
  N_BIN, N_EXP = 0, 0
  simplices1 = np.array([unrank_colex(r, n, m, C=C) for r in R])
  # unrank_colex(r, n, m, exp=False)
  NS_BIN = N_BIN + N_EXP
  
  N_BIN, N_EXP = 0, 0
  simplices2 = np.array([unrank_colex(r, n, m, exp=True, C=C) for r in R])
  # unrank_colex(r, n, m, exp=True)
  NS_EXP = N_BIN + N_EXP
  
  N_BIN, N_EXP = 0, 0
  simplices3 = np.array([unrank_colex(r, n, m, exp=True, use_lb=True, C=C) for r in R])
  # unrank_colex(r, n, m, exp=True, use_lb=True)
  NS_EXP_LB = N_BIN + N_EXP
  
  N_BIN, N_EXP = 0, 0
  simplices4 = np.array([unrank_colex(r, n, m, exp=False, use_lb=True, C=C) for r in R])
  # unrank_colex(r, n, m, exp=False, use_lb=True)
  NS_BIN_LB = N_BIN + N_EXP
  print(f"(C={C}): \nBin only: {NS_BIN}\nExp only: {NS_EXP}\nExp w/ LB: {NS_EXP_LB}\nBin w/ LB: {NS_BIN_LB}")

  ## Somehow, binary w/ LB is worse than binary without! (binary+LB uses about 30% more!)
  ## Plain exponential search is abotu 2.5x worse than plain binary search! 
  ## On the other hand: exp search + LB is about 2x better binary search
  ## Setting C ~= 2 or 3 has a massive reduction
