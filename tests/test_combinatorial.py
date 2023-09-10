import numpy as np
from math import comb
from itertools import combinations
from functools import partial
from combin import comb_to_rank, rank_to_comb, inverse_choose
from combin.combinatorial import _combinatorial

def test_combs():
  assert all(_combinatorial.comb([1,2,3],[1,2,3]) == np.array([1,1,1]))
  assert all(_combinatorial.comb([1,2,3],[0,0,0]) == np.array([1,1,1]))
  assert rank_to_comb([0,1,2], k=3) == [(0, 1, 2), (0, 1, 3), (0, 2, 3)]
  assert all(comb_to_rank([(0,1,2), (0,1,3), (0,2,3)]) == [0,1,2])

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
  assert all(np.ravel(combs_test == combs_truth)), "Colex unranking invalid"
  assert all(np.ravel(combs_truth == rank_to_comb(ranks, k=k, order="colex"))), "Colex unranking invalid"

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

# def test_api():
#   n = 20
#   for d in range(1, 5):
#     combs = list(combinations(range(n), d))
#     C = unrank_combs(comb_to_rank(combs), k=d)
#     assert all([tuple(s) == tuple(c) for s,c in zip(combs, C)])

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
