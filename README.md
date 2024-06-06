# combin
![coverage_badge](tests/coverage_badge.svg)
![macos workflow status](https://img.shields.io/github/actions/workflow/status/peekxc/combin/build_macos.yml?logo=apple&logoColor=white)
![windows workflow status](https://img.shields.io/github/actions/workflow/status/peekxc/combin/build_windows.yml?logo=windows&logoColor=white)
![linux workflow status](https://img.shields.io/github/actions/workflow/status/peekxc/combin/build_linux.yml?logo=linux&logoColor=white)

Lightweight implementation of various combinatorics routines (Under construction!).

Current routines offered: 

- Combinatorial number system ranking/unranking (`rank_to_comb` and `comb_to_rank`)
- Binomial coefficient inversion (`inverse_choose`)

## Installation 

`combin` is on PyPI and can be installed with the usual command: 

```bash
python -m pip install combin
```

If this fails for your system, please file an issue. 

## Usage

`combin` supports fast bijections to the [combinatorial number system](https://en.wikipedia.org/wiki/Combinatorial_number_system). 

```python
from combin import comb_to_rank, rank_to_comb

## Fix k-combinations of an n-set
n = 10 # universe size 
k = 3  # size of each combination

## Ranks each 3-tuple into its index in the combinatorial number system  
C = combinations(range(n), k) # from itertools 
R = comb_to_rank(C, order='lex', n = n)
print(R)
# [0, 1, 2, ..., 119]

## If a generator is given, the default return type is a list
assert R == list(range(120))
# True

## Alternatively, combin has native support for NumPy arrays
C = np.fromiter(combinations(range(n), k), dtype=(np.int16, k))
assert np.all(comb_to_rank(C, order='lex', n=n) == np.arange(comb(n,k)))
# True

## An alternative bijection can also be chosen, such as the colexicographical order
## Note the colex order doesn't require the universe size (n)
print(comb_to_rank(C, order='colex'))
# [0, 1, 4, ..., 119]

## Unranking is just as easy: just supply n, k, and the order
R = np.arange(comb(n,k))
C_lex = rank_to_comb(R, k=k, n=n, order='lex')
print(f"Equal? {np.all(C_lex == C)}, combs: {C_lex}")
# Equal? True, combs: [[0, 1, 2], [0, 1, 3], [0, 1, 4], ..., [6, 8, 9], [7, 8, 9]]

## Uniformly sampling from k-combinations becomes trivial
ind = np.random.choice(range(comb(n,k)), size=10, replace=False)
random_combs = rank_to_comb(ind, k=k, order='colex')
# [[2, 7, 8], [1, 4, 6], ...]
```