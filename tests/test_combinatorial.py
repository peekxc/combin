import numpy as np
from math import comb
from itertools import combinations
from functools import partial
from combin import comb_to_rank, rank_to_comb, inverse_choose, comb_rank_colex
from combin.combinatorial import _combinatorial

def test_combs():
  _combinatorial.comb([1,2,3],[1,2,3])
  _combinatorial.comb([1,2,3],[0,0,0])
  _combinatorial.comb([1,2,3],[0,0,0])


def test_numpy_ranking():
  n, k = 10, 3
  combs = np.array(list(combinations(range(n), k)), dtype=np.uint16)
  assert all(np.equal(comb_to_rank(combs, n, 'lex'), np.arange(120, dtype=np.uint64)))
  assert all(np.equal(comb_to_rank([tuple(c) for c in combs], n, 'lex'), np.arange(120, dtype=np.uint64)))
  
  np.array([comb_rank_colex(c) for c in combs[:5]])
  np.array([comb_rank_colex(c) for c in np.fliplr(combs)])
  comb_to_rank(np.fliplr(combs), order='colex')- 1
  comb_to_rank(combs[:5], order='colex')
  comb_to_rank(np.fliplr(combs[:5]), order='colex')

  ## Gotta fliplr sorted array since C++ precondition requires it but otherwise it's correct 
  comb(2,3) + comb(1,2) + comb(0,1)
  _combinatorial.comb([2,1,0],[3,2,1])

  comb_to_rank(np.fliplr(combs), order='colex')




def test_colex():
  n, k = 10, 3
  ranks = np.array([comb_to_rank(c) for c in combinations(range(n), k)])
  assert all(np.sort(ranks) == np.arange(comb(n,k))), "Colex ranking is not unique / does not form bijection"
  ranks2 = np.array([rank_colex(reversed(c)) for c in combinations(range(n), k)])
  assert all(ranks == ranks2), "Ranking is not order-invariant"
  combs_test = np.array([unrank_colex(r, k) for r in ranks])
  combs_truth = np.array(list(combinations(range(n),k)))
  assert all((combs_test == combs_truth).flatten()), "Colex unranking invalid"

def test_array_conversion():
  x = np.array(unrank_combs([0,1,2], k=2))
  assert np.all(x == np.array([[0,1], [0,2], [1,2]], dtype=np.uint16))

def test_lex():
  n, k = 10, 3
  ranks = np.array([rank_lex(c, n) for c in combinations(range(n), k)])
  assert all(ranks == np.arange(comb(n,k))), "Lex ranking is not unique / does not form bijection"
  ranks2 = np.array([rank_lex(reversed(c), n) for c in combinations(range(n), k)])
  assert all(ranks == ranks2), "Ranking is not order-invariant"
  combs_test = np.array([unrank_lex(r, k, n) for r in ranks])
  combs_truth = np.array(list(combinations(range(n),k)))
  assert all((combs_test == combs_truth).flatten()), "Lex unranking invalid"

def test_api():
  n = 20
  for d in range(1, 5):
    combs = list(combinations(range(n), d))
    C = unrank_combs(rank_combs(combs), k=d)
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
