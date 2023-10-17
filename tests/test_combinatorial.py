import numpy as np
from math import comb
from itertools import combinations, product 
from functools import partial
from combin import comb_to_rank, rank_to_comb, inverse_choose
from combin.combinatorial import _combinatorial, _comb_unrank_colex, _comb_rank_colex
# print(__file__)

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
  assert rank_to_comb([0,3,4], [1,2,3]) == [(0,), (0, 3), (0, 1, 4)]


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
  assert rank_to_comb([0,1,2], k=3) == [(0, 1, 2), (0, 1, 3), (0, 2, 3)]
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