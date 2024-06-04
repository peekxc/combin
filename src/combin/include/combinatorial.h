// combinatorial.h 
// Contains routines for combinatorics-related tasks 
// The combinations and permutations generation code is copyright Howard Hinnant, taken from: https://github.com/HowardHinnant/combinations/blob/master/combinations.h
#ifndef COMBINATORIAL_H
#define COMBINATORIAL_H 

// #include <iostream>
#include <cstdint>		// uint_fast64_t
#include <array>
#include <cassert>
// #include <span> 		 	// span (C++20)
#include <cmath>	 	 	// round, sqrt
#include <numeric>   	// midpoint, accumulate
#include <vector> 	 	// vector  
#include <algorithm> 
#include <type_traits>
#include <vector>
#include <functional>
#include <iterator>
#include <cassert>
#include <cstdint>
#include <iostream>

using std::begin;
using std::end; 
using std::vector; 
using std::size_t;

namespace combinatorial {
	using index_t = int_fast64_t;

	template<typename T>
	using it_diff_t = typename std::iterator_traits<T>::difference_type;

	// Rotates two discontinuous ranges to put *first2 where *first1 is.
	//     If last1 == first2 this would be equivalent to rotate(first1, first2, last2),
	//     but instead the rotate "jumps" over the discontinuity [last1, first2) -
	//     which need not be a valid range.
	//     In order to make it faster, the length of [first1, last1) is passed in as d1,
	//     and d2 must be the length of [first2, last2).
	//  In a perfect world the d1 > d2 case would have used swap_ranges and
	//     reverse_iterator, but reverse_iterator is too inefficient.
	template <class It>
	void rotate_discontinuous(
		It first1, It last1, it_diff_t< It > d1,
		It first2, It last2, it_diff_t< It > d2)
	{
		using std::swap;
		if (d1 <= d2){ std::rotate(first2, std::swap_ranges(first1, last1, first2), last2); }
		else {
			It i1 = last1;
			while (first2 != last2)
				swap(*--i1, *--last2);
			std::rotate(first1, i1, last1);
		}
	}

	// Call f() for each combination of the elements [first1, last1) + [first2, last2)
	//    swapped/rotated into the range [first1, last1).  As long as f() returns
	//    false, continue for every combination and then return [first1, last1) and
	//    [first2, last2) to their original state.  If f() returns true, return
	//    immediately.
	//  Does the absolute mininum amount of swapping to accomplish its task.
	//  If f() always returns false it will be called (d1+d2)!/(d1!*d2!) times.
	template < typename It, typename Lambda >
	bool combine_discontinuous(
		It first1, It last1, it_diff_t< It > d1,  
		It first2, It last2, it_diff_t< It > d2,
		Lambda&& f, it_diff_t< It > d = 0)
	{
		using D = it_diff_t< It >;
		using std::swap;
		if (d1 == 0 || d2 == 0){ return f(); }
		if (d1 == 1) {
			for (It i2 = first2; i2 != last2; ++i2) {
				if (f()){ return true; }
				swap(*first1, *i2);
			}
		}
		else {
			It f1p = std::next(first1), i2 = first2;
			for (D d22 = d2; i2 != last2; ++i2, --d22){
				if (combine_discontinuous(f1p, last1, d1-1, i2, last2, d22, f, d+1))
					return true;
				swap(*first1, *i2);
			}
		}
		if (f()){ return true; }
		if (d != 0){ rotate_discontinuous(first1, last1, d1, std::next(first2), last2, d2-1); }
		else { rotate_discontinuous(first1, last1, d1, first2, last2, d2); }
		return false;
	}

	template < typename Lambda, typename It > 
	struct bound_range { 
		Lambda f_;
		It first_, last_;
		bound_range(Lambda& f, It first, It last) : f_(f), first_(first), last_(last) {}
		bool operator()(){ return f_(first_, last_); } 
		bool operator()(It, It) { return f_(first_, last_); }
	};

	template <class It, class Function>
	Function for_each_combination(It first, It mid, It last, Function&& f) {
		bound_range<Function&, It> wfunc(f, first, mid);
		combine_discontinuous(first, mid, std::distance(first, mid),
													mid, last, std::distance(mid, last),
													wfunc);
		return std::move(f);
	}

	template <std::size_t... Idx>
	constexpr auto make_index_dispatcher(std::index_sequence<Idx...>) {
		return [] (auto&& f) { (f(std::integral_constant<std::size_t,Idx>{}), ...); };
	};
	
	template <std::size_t N>
	constexpr auto make_index_dispatcher() {
		return make_index_dispatcher(std::make_index_sequence< N >{});
	};

	template <typename T, size_t I > 
	struct tuple_n{
		template< typename...Args> using type = typename tuple_n<T, I-1>::template type<T, Args...>;
	};

	// Modified from: https://stackoverflow.com/questions/38885406/produce-stdtuple-of-same-type-in-compile-time-given-its-length-by-a-template-a
	template <typename T> 
	struct tuple_n<T, 0 > {
		template<typename...Args> using type = std::tuple<Args...>;   
	};
	template < typename T, size_t I >  using tuple_of = typename tuple_n<T, I>::template type<>;
	
	// Constexpr binomial coefficient using recursive formulation
	template < size_t n, size_t k >
	constexpr auto bc_recursive() noexcept {
		if constexpr ( n == k || k == 0 ){ return(1); }
		else if constexpr (n == 0 || k > n){ return(0); }
		else {
		 return (n * bc_recursive< n - 1, k - 1>()) / k;
		}
	}
	
	// Baseline from: https://stackoverflow.com/questions/44718971/calculate-binomial-coffeficient-very-reliably
	// Requires O(min{k,n-k}), uses pascals triangle approach (+ degenerate cases)
	constexpr inline size_t binom(size_t n, size_t k) noexcept {
		return
			(k > n) ? 0 :                  // out of range
			(k == 0 || k == n) ? 1 :       // edge
			(k == 1 || k == n-1) ? n :     // first
			(k+k < n) ?                    // recursive:
			(binom(n-1,k-1) * n)/k :       //  path to k=1   is faster
			(binom(n-1,k) * n)/(n-k);      //  path to k=n-1 is faster
	}

	// Table to cache low values of the binomial coefficient
	template< size_t max_n, size_t max_k, typename value_t = index_t >
	struct BinomialCoefficientTable {
	  size_t pre_n = 0;
		size_t pre_k = 0; 
		value_t combinations[max_k][max_n+1];
	  vector< vector< value_t > > BT;

		constexpr BinomialCoefficientTable() : combinations() {
			auto n_dispatcher = make_index_dispatcher< max_n+1 >();
			auto k_dispatcher = make_index_dispatcher< max_k >();
			n_dispatcher([&](auto i) {
				k_dispatcher([&](auto j){
					combinations[j][i] = bc_recursive< i, j >();
				});
			});
	  }

		// Evaluate general binomial coefficient, using cached table if possible 
		value_t operator()(const index_t n, const index_t k) const {
			// std::cout << "INFO: " << pre_n << ", " << n <<  ":" << pre_k << ", " << k << " : " << BT.size() << std::endl; 
			// if (n < max_n && k < max_k){ return combinations[k][n]; } // compile-time computed table
			if (n <= pre_n && k <= pre_k){ return BT[k][n]; } 				// runtime computed extension table
			return static_cast< value_t >(binom(n,std::min(k,n-k)));
			// if (k == 0 || n == k){ return 1; }
			// if (n < k){ return 0; }
			// if (k == 2){ return static_cast< value_t >((n*(n-1))/2); }
			// if (k == 1){ return n; }
			// return binom(n, k);
			// return binomial_coeff_(n,std::min(k,n-k));
		}

		// Precompute a *larger* table of binomial coefficients
		void precompute(index_t n, index_t k){
			pre_n = n;
			pre_k = k;
			BT = std::vector< std::vector< value_t > >(k + 2, std::vector< value_t >(n + 2, 0));
			for (index_t i = 0; i <= n; ++i) {
				BT[0][i] = 1;
				for (index_t j = 1; j < std::min(i, k + 1); ++j){
					BT[j][i] = binom(i,j); // BT[j - 1][i - 1] + BT[j][i - 1];
				}
				if (i <= k) { BT[i][i] = 1; };
			}
		}

		// Fast but unsafe access to a precompute table
		[[nodiscard]]
		constexpr auto at(index_t n, index_t k) noexcept -> index_t {
			return BT[k][n];
		}

	}; // BinomialCoefficientTable

	// Build the cached table
	static auto BC = BinomialCoefficientTable< 64, 3 >();
	static bool keep_table_alive = false; 

	// Wrapper to choose between cached and non-cached version of the Binomial Coefficient
	template< bool safe = true >
	constexpr size_t BinomialCoefficient(const size_t n, const size_t k){
		if constexpr(safe){
			return BC(n,k);
		} else {
			return BC.at(n,k);
		}
	}
	
	#if __cplusplus >= 202002L
    // C++20 (and later) code
		// constexpr midpoint midpoint
		using std::midpoint; 
	#else
		template < class Integer > 
		constexpr Integer midpoint(Integer a, Integer b) noexcept {
			return (a+b)/2;
		}
	#endif

	// All inclusive range binary search 
	// Compare must return -1 for <(key, index), 0 for ==(key, index), and 1 for >(key, index)
	// Guaranteed to return an index in [0, n-1] representing the lower_bound
	template< typename T, typename Compare > [[nodiscard]]
	int binary_search(const T key, size_t n, Compare p) noexcept {
	  int low = 0, high = n - 1, best = 0; 
		while( low <= high ){
			int mid = int{ midpoint(low, high) };
			auto cmp = p(key, mid);
			if (cmp == 0){ 
				while(p(key, mid + 1) == 0){ ++mid; }
				return(mid);
			}
			if (cmp < 0){ high = mid - 1; } 
			else { 
				low = mid + 1; 
				best = mid; 
			}
		}
		return(best);
	}
	
	// ----- Combinatorial Number System functions -----
	template< std::integral I, typename Compare > 
	void sort_contiguous(vector< I >& S, const size_t modulus, Compare comp){
		for (size_t i = 0; i < S.size(); i += modulus){
			std::sort(S.begin()+i, S.begin()+i+modulus, comp);
		}
	}

	// Lexicographically rank 2-subsets
	[[nodiscard]]
	constexpr auto rank_lex_2(index_t i, index_t j, const index_t n) noexcept {
	  if (j < i){ std::swap(i,j); }
	  return index_t(n*i - i*(i+1)/2 + j - i - 1);
	}

	// #include <iostream> 
	// Lexicographically rank k-subsets
	template< bool safe = true, typename InputIter >
	[[nodiscard]]
	inline index_t rank_lex_k(InputIter s, const size_t n, const size_t k, const index_t N){
		index_t val = 0;
		for (int64_t i = k; i > 0; --i, ++s){
			val += BinomialCoefficient< safe >((n-1) - (*s), i);
		}
		// STL is apparently unreliable! 
	  // const index_t index = std::accumulate(s, s+k, 0, [n, &i](index_t val, index_t num){ 
		//   return val + BinomialCoefficient< safe >((n-1) - num, i--); 
		// });
	  const index_t combinadic = (N-1) - val; // Apply the dual index mapping
	  return combinadic;
	}

	// Rank a stream of integers (lexicographically)
	template< bool safe = true, typename InputIt, typename OutputIt >
	inline void rank_lex(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		switch (k){
			case 2:{
				for (; s != e; s += k){
					*out++ = rank_lex_2(*s, *(s+1), n);
				}
				break;
			}
			default: {
				const index_t N = BinomialCoefficient< safe >(n, k); 
				for (; s != e; s += k){
					*out++ = rank_lex_k< safe >(s, n, k, N);
				}
				break;
			}
		}
	}

	// should be in reverse colexicographical 
	template< bool safe = true >
	[[nodiscard]]
	constexpr auto rank_colex_2(index_t i, index_t j) noexcept {
		assert(i > j); // should be in colex order! 
		return BinomialCoefficient< safe >(j, 1) + BinomialCoefficient< safe >(i, 2);
	}
	

	// Colexicographically rank k-subsets
	// assumes each k tuple of s is in colex order! 
	template< bool safe = true, typename InputIter >
	[[nodiscard]]
	constexpr auto rank_colex_k(InputIter s, const size_t k) noexcept {
		int64_t val = 0; 
		for (int64_t ii = k; ii > 0; --ii, ++s){
			val += int64_t(BinomialCoefficient< safe >(*s, ii));
			// std::cout << int64_t(BinomialCoefficient< safe >(*(s+jj), ii)) << " -> " << val << ", ";
		}
		// STIL is just unreliable apparently 
		// const index_t index = std::accumulate(s, s+k, 0, [&ii](index_t val, index_t num){ 
		// 	int64_t out = int64_t(val) + int64_t(BinomialCoefficient< safe >(num, ii));
		// 	std::cout << int64_t(BinomialCoefficient< safe >(num, ii)) << " -> " << out << ", ";
		// 	ii--; 
		// 	return out;
		// 	// return val + BinomialCoefficient< safe >(num, i); 
		// });
		// std::cout << ": " << val << std::endl; 
		return val; 
	}

	template< bool safe = true, typename InputIt, typename OutputIt >
	inline void rank_colex(InputIt s, const InputIt e, [[maybe_unused]] const size_t n, const size_t k, OutputIt out) noexcept {
		switch (k){
			case 2:{
				for (; s != e; s += k){
					*out++ = rank_colex_2(*s, *(s+1));
				}
				break;
			}
			default: {
				for (; s != e; s += k){
					*out++ = rank_colex_k< safe >(s, k);
				}
				break;
			}
		}
	}

	// colex bijection from a lexicographical order
	// index_t i = 1; 
	// const index_t index = std::accumulate(s, s+k, 0, [&i](index_t val, index_t num){ 
	// 	return val + BinomialCoefficient< safe >(num, i++); 
	// });
	// return index; 

	template< bool colex = true, bool safe = true, typename InputIt > 
	inline auto rank_comb(InputIt s, const size_t n, const size_t k){
		if constexpr(colex){
			return rank_colex_k< safe >(s, k);
		} else {
			const index_t N = BinomialCoefficient< safe >(n, k); 
			return rank_lex_k< safe >(s, n, k, N);
		}
	}

	template< bool colex = true, bool safe = true, typename InputIt, typename OutputIt > 
	inline void rank_combs(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		if constexpr(colex){
			for (; s != e; s += k){
				*out++ = rank_colex_k< safe >(s, k);
			}
		} else {
			const index_t N = BinomialCoefficient< safe >(n, k); 
			for (; s != e; s += k){
				*out++ = rank_lex_k< safe >(s, n, k, N);
			}
		}
	}

	// Lexicographically unrank 2-subsets
	template< typename OutputIt  >
	inline auto unrank_lex_2(const index_t r, const index_t n, OutputIt out) noexcept  {
		auto i = static_cast< index_t >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
		auto j = static_cast< index_t >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
		*out++ = i; // equivalent to *out = i; ++i;
		*out++ = j; // equivalent to *out = j; ++j;
	}
	
	// Lexicographically unrank k-subsets [ O(log n) version ]
	// template< bool safe = true, typename OutputIterator > 
	// inline void unrank_lex_k(index_t r, const size_t n, const size_t k, OutputIterator out) noexcept {
	// 	const size_t N = combinatorial::BinomialCoefficient< safe >(n, k);
	// 	r = (N-1) - r; 
	// 	for (size_t ki = k; ki > 0; --ki){
	// 		int offset = binary_search(r, n, [ki](const auto& key, int index) -> int {
	// 			auto c = combinatorial::BinomialCoefficient< safe >(index, ki);
	// 			return(key == c ? 0 : (key < c ? -1 : 1));
	// 		});
	// 		r -= combinatorial::BinomialCoefficient< safe >(offset, ki); 
	// 		*out++ = (n-1) - offset;
	// 	}
	// }

	// Lexicographically unrank k-subsets [ O(n) version ]
	// template< bool safe = true, typename OutputIterator >
	// inline void unrank_lex_k(index_t r, const size_t n, const size_t k, OutputIterator out){
	// 	size_t x = 1; 
	// 	for (size_t i = 1; i <= k; ++i){
	// 		while(r >= BinomialCoefficient(n-x, k-i)){
	// 			r -= BinomialCoefficient(n-x, k-i);
	// 			x += 1;
	// 		}
	// 		*out++ = (x - 1);
	// 		x += 1;
	// 	}
	// }

	// Lexicographically unrank subsets wrapper
	// NOTE: don't re-pass output iterator to another function, as pass by value won't work here!
	template< bool safe = true, typename InputIt, typename OutputIt >
	inline void unrank_lex(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		switch(k){
			case 2: { 
				for (auto r = *s; s != e; ++s, r = *s){ 
					auto i = static_cast< index_t >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
					auto j = static_cast< index_t >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
					*out++ = i; // equivalent to *out = i; ++i;
					*out++ = j; // equivalent to *out = j; ++j;
				}
				break;
			}
			default: {
				// O(n) variant 
				// for (auto r = *s; s != e; ++s, r = *s){ 
				// 	size_t x = 1; 
				// 	for (size_t i = 1; i <= k; ++i){
				// 		while(r >= BinomialCoefficient(n-x, k-i)){
				// 			r -= BinomialCoefficient(n-x, k-i);
				// 			x += 1;
				// 		}
				// 		*out++ = (x - 1);
				// 		x += 1;
				// 	}
				// }
				// O(log(n)) variant 
				const size_t N = combinatorial::BinomialCoefficient< safe >(n, k);
				for (auto r = *s; s != e; ++s, r = *s){ 
					r = (N-1) - r; // apply dual mapping
					for (size_t ki = k; ki > 0; --ki){
						int offset = binary_search(r, n, [ki](const auto& key, int index) -> int {
							auto c = combinatorial::BinomialCoefficient< safe >(index, ki);
							return(key == c ? 0 : (key < c ? -1 : 1));
						});
						r -= combinatorial::BinomialCoefficient< safe >(offset, ki); 
						*out++ = (n-1) - offset; // undo dual mapping
					}
				}
				break;
			}
		}
	}

	// Finds the largest index i in the range [bottom, top] wherein pred(i) is true by binary search. 
	// For examle, assuming the sequence is sorted like: 
	// v = { true, true, true, false, false, false, false }
	// In this case, get_max(v.size()-1, v, v.operator[]) returns the index 2. 
	// template <class Predicate>
	// [[nodiscard]]
	// index_t get_max(index_t top, const index_t bottom, const Predicate pred) noexcept {
	// 	if (!pred(top)) {
	// 		index_t count = top - bottom;
	// 		while (count > 0) {
	// 			index_t step = count >> 1, mid = top - step;
	// 			if (!pred(mid)) {
	// 				top = mid - 1;
	// 				count -= step + 1;
	// 			} else {
	// 				count = step;
	// 			}
	// 		}
	// 	}
	// 	return top;
	// }

	// Finds the largest index i in the range [bottom, top] wherein pred(i) == true by exponential search or binary search. 
	// Ex: if v = { true, true, true, false, false, false, false }, then 
	// 				get_max(v.size()-1, 0, v.operator[]) returns 2. 
	template < bool ExpSearch = false, class Predicate >
	[[nodiscard]]
	index_t get_max(index_t top, index_t bottom, const Predicate pred) noexcept {
		if (!pred(bottom)) { return bottom; }
  	index_t size = (top - bottom);
		if constexpr (ExpSearch){
			index_t inc = 1;
			while ((bottom + inc) < top && pred(bottom + inc)){
				inc = inc << 1;
			}
			bottom = std::max(bottom, index_t((bottom + inc) / 2));
			top = std::min(bottom + inc, top);
			size = (top - bottom);
		}
		while (size > 0){
			index_t step = size >> 1;
			index_t mid = top - step;
			if (!pred(mid)){
				top = mid - 1;
				size -= step + 1;
			} else {
				size = step;
			}
		}
  	return top;
	}

	// ~ 4 KB of 
	// const std::array< int > R = {1,1,3,6,10,16,21,28,36,45,55,67,78,91,105,120,136,154,171,190,210,231,253,277,300,325,351,378,406,436,465,496,528,561,595,631,666,703,741,780,820,862,903,946,990,1035,1081,1129,1176,1225,1275,1326,1378,1432,1485,1540,1596,1653,1711,1771,1830,1891,1953,2016,2080,2146,2211,2278,2346,2415,2485,2557,2628,2701,2775,2850,2926,3004,3081,3160,3240,3321,3403,3487,3570,3655,3741,3828,3916,4006,4095,4186,4278,4371,4465,4561,4656,4753,4851,4950,5050,5152,5253,5356,5460,5565,5671,5779,5886,5995,6105,6216,6328,6442,6555,6670,6786,6903,7021,7141,7260,7381,7503,7626,7750,7876,8001,8128,8256,8385,8515,8647,8778,8911,9045,9180,9316,9454,9591,9730,9870,10011,10153,10297,10440,10585,10731,10878,11026,11176,11325,11476,11628,11781,11935,12091,12246,12403,12561,12720,12880,13042,13203,13366,13530,13695,13861,14029,14196,14365,14535,14706,14878,15052,15225,15400,15576,15753,15931,16111,16290,16471,16653,16836,17020,17206,17391,17578,17766,17955,18145,18337,18528,18721,18915,19110,19306,19504,19701,19900,20100,20301,20503,20707,20910,21115,21321,21528,21736,21946,22155,22366,22578,22791,23005,23221,23436,23653,23871,24090,24310,24532,24753,24976,25200,25425,25651,25879,26106,26335,26565,26796,27028,27262,27495,27730,27966,28203,28441,28681,28920,29161,29403,29646,29890,30136,30381,30628,30876,31125,31375,31627,31878,32131,32385,32640,32896,33154,33411,33670,33930,34191,34453,34717,34980,35245,35511,35778,36046,36316,36585,36856,37128,37401,37675,37951,38226,38503,38781,39060,39340,39622,39903,40186,40470,40755,41041,41329,41616,41905,42195,42486,42778,43072,43365,43660,43956,44253,44551,44851,45150,45451,45753,46056,46360,46666,46971,47278,47586,47895,48205,48517,48828,49141,49455,49770,50086,50404,50721,51040,51360,51681,52003,52327,52650,52975,53301,53628,53956,54286,54615,54946,55278,55611,55945,56281,56616,56953,57291,57630,57970,58312,58653,58996,59340,59685,60031,60379,60726,61075,61425,61776,62128,62482,62835,63190,63546,63903,64261,64621,64980,65341,65703,66066,66430,66796,67161,67528,67896,68265,68635,69007,69378,69751,70125,70500,70876,71254,71631,72010,72390,72771,73153,73537,73920,74305,74691,75078,75466,75856,76245,76636,77028,77421,77815,78211,78606,79003,79401,79800,80200,80602,81003,81406,81810,82215,82621,83029,83436,83845,84255,84666,85078,85492,85905,86320,86736,87153,87571,87991,88410,88831,89253,89676,90100,90526,90951,91378,91806,92235,92665,93097,93528,93961,94395,94830,95266,95704,96141,96580,97020,97461,97903,98347,98790,99235,99681,100128,100576,101026,101475,101926,102378,102831,103285,103741,104196,104653,105111,105570,106030,106492,106953,107416,107880,108345,108811,109279,109746,110215,110685,111156,111628,112102,112575,113050,113526,114003,114481,114961,115440,115921,116403,116886,117370,117856,118341,118828,119316,119805,120295,120787,121278,121771,122265,122760,123256,123754,124251,124750,125250,125751,126253,126757,127260,127765,128271,128778,129286,129796,130219}

	// count leading zeros of nonzero 64-bit unsigned integer
	// int clz64(uint64_t x);

	// isqrt64_tab[k] = isqrt(256*(k+65)-1) for 0 <= k < 192
	static const uint8_t isqrt64_tab[192] = {
			128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
			138, 139, 140, 141, 142, 143, 143, 144, 145, 146,
			147, 148, 149, 150, 150, 151, 152, 153, 154, 155,
			155, 156, 157, 158, 159, 159, 160, 161, 162, 163,
			163, 164, 165, 166, 167, 167, 168, 169, 170, 170,
			171, 172, 173, 173, 174, 175, 175, 176, 177, 178,
			178, 179, 180, 181, 181, 182, 183, 183, 184, 185,
			185, 186, 187, 187, 188, 189, 189, 190, 191, 191,
			192, 193, 193, 194, 195, 195, 196, 197, 197, 198,
			199, 199, 200, 201, 201, 202, 203, 203, 204, 204,
			205, 206, 206, 207, 207, 208, 209, 209, 210, 211,
			211, 212, 212, 213, 214, 214, 215, 215, 216, 217,
			217, 218, 218, 219, 219, 220, 221, 221, 222, 222,
			223, 223, 224, 225, 225, 226, 226, 227, 227, 228,
			229, 229, 230, 230, 231, 231, 232, 232, 233, 234,
			234, 235, 235, 236, 236, 237, 237, 238, 238, 239,
			239, 240, 241, 241, 242, 242, 243, 243, 244, 244,
			245, 245, 246, 246, 247, 247, 248, 248, 249, 249,
			250, 250, 251, 251, 252, 252, 253, 253, 254, 254,
			255, 255,
	};


	// integer square root of a 64-bit unsigned integer; cannot be 0 
	uint32_t isqrt64_nozero(uint64_t x){ 
		// if (x == 0) return 0;
		int lz = __builtin_clz(x) & 62;
		x <<= lz;
		uint32_t y = isqrt64_tab[(x >> 56) - 64];
		y = (y << 7) + (x >> 41) / y;
		y = (y << 15) + (x >> 17) / y;
		y -= x < (uint64_t)y * y;
		return y >> (lz >> 1);
	}

	uint32_t icbrt64(uint64_t x) {
		int s;
		uint32_t y;
		uint64_t b;

		y = 0;
		for (s = 63; s >= 0; s -= 3) {
			y += y;
			b = 3*y*((uint64_t) y + 1) + 1;
			if ((x >> s) >= b) {
				x -= b << s;
				y++;
			}
		}
		return y;
	}
	// Find a good lower bound to initiate the search for the value k satisfying choose(k-1, m) <= r < choose(k, m)
	// From: Kruchinin, Vladimir, et al. "Unranking Small Combinations of a Large Set in Co-Lexicographic Order." Algorithms 15.2 (2022): 36.
	// return std::ceil(m * exp(log(r)/m + log(2*pi*m)/2*m + 1/(12*m*m) - 1/(360*pow(m,4)) - 1) + (m-1)/2);
	// Unranking m-combinations of a n-set in co-lexicographic order
	[[nodiscard]]
	constexpr auto find_k(const index_t r, const index_t m) noexcept -> index_t {
		assert(m > 0); // m should never be zero probably
		if (r == 0){ return m - 1; } 
		switch(m){
			case 1:
				return r;
			case 2:
				// return std::max(std::ceil((1.0+std::sqrtf(1.f+8.f*r))/2.f) - 1.0, 0.0);
				return std::ceil((1.0+std::sqrt(1.f+8.f*r))/2.f) - 1.0;
				// return ((1+isqrt64_nozero(1+8*r)) >> 2) - 1;
				// return m - 1;
			case 3: 
				// return std::max(std::ceil(std::cbrtf(6.f*r)) - 1.0, 0.0);
				return std::ceil(std::cbrt(6.f*r)) - 1.0;
				// return icbrt64(6 * r); // this is WAY more expensive!
				// int b = ((64) - __builtin_clzll(r)) + 3; // adjust for multiplying by 6
    		// int k_lb = 1 << (int(b / 3) - 1);
				// return 1 << (int((((64) - __builtin_clzll(r)) + 3) / 3) - 1); // lower bound on the LB
				// return m - 1;
			default: 
				return m - 1;
		}
	}

// else if (m == 2){ return std::ceil((1.0+std::sqrt(1+8*static_cast<float>(r)))/2.f); }
// else if (m == 3){ return std::ceil(std::pow(6*static_cast<float>(r), 1.f/3.f)); }
// else { 
// 	return static_cast< index_t >(m); // the final bound they compute didn't make sense in my tests, so we return m - 1.
// }
// else if (m == 1){ return r; }
// else if (m == 2){ return std::ceil((1.0+std::sqrt(1+8*static_cast<float>(r)))/2.f); }
// else if (m == 3){ return std::ceil(std::pow(6*static_cast<float>(r), 1.f/3.f)); }
// else { 
// 	return static_cast< index_t >(m); // the final bound they compute didn't make sense in my tests, so we return m - 1.
// }


	// Binary searches for the value K satisfying choose(K-1, m) <= r < choose(K, m) 
	// This implements two algorithms: 
	// (1) First, a lower bound for K is computed in O(1) time using an approximation that tends to be tight
	// (2) Second, a constant number of comparisons are done to check the tightness of the approximation, returning early if successful
	// (3) If (1) and (2) fail, then the range [k-1, n] is searched for the largest index _w_ satisfying r >= choose(w,k) via binary search
	// k >= 1, N <= n - 1, 0 <= r < choose(n,k)
	// NOTE: For n < 500 (~50-150), regular binary search w/o LB calculation is the fastest. For larger n, Exp search + LB is worth it. 
	template< bool safe = true, bool use_lb = false, bool ExpSearch = false, size_t C = 0 > 
	[[nodiscard]]
	index_t get_max_vertex(const index_t r, const index_t m, const index_t n) noexcept {
		index_t k_lb;
		const auto pred = [r,m](index_t w) -> bool { return BinomialCoefficient< safe >(w, m) <= r; };
		if constexpr(use_lb){
			k_lb = find_k(r,m); // finds k such that comb(k-1, m) <= r
		} else {
			k_lb = m-1; 
		}
		assert(k_lb >= 0);  // It should be a non-negative integer!
		assert(pred(k_lb));	// it should be a lower bound!

		if constexpr (C == 0){
			return get_max< ExpSearch >(n, k_lb, pred) + 1;
		} else if constexpr (C == 1){
			if (!pred(k_lb + 1)){ return k_lb+1; }
			return get_max< ExpSearch >(n, k_lb, pred) + 1;
		} else if constexpr (C == 2){
			if (!pred(k_lb + 1)){ return k_lb+1; }
			if (!pred(k_lb + 2)){ return k_lb+2; }
			return get_max< ExpSearch >(n, k_lb, pred) + 1; 
		} else { // (K >= 3)
			if (!pred(k_lb + 1)){ return k_lb+1; }
			if (!pred(k_lb + 2)){ return k_lb+2; }
			if (!pred(k_lb + 3)){ return k_lb+3; }
			return get_max< ExpSearch >(n, k_lb, pred) + 1; 
		}
	}

	// Successively unranks each k-combination of a n-set into out
	template < bool safe = true, bool use_lb = false, bool ExpSearch = false, size_t C = 0, typename InputIt, typename OutputIt >
	void unrank_colex(InputIt s, const InputIt e, const index_t n, const index_t k, OutputIt out) noexcept {
		for (index_t K = n - 1; s != e; ++s){
			index_t r = static_cast< index_t >(*s); 
			for (index_t m = k; m > 1; --m) {
				K = get_max_vertex< safe, use_lb, ExpSearch, C >(r, m, n); // k satisfying comb(k-1,m) <= r < comb(k, m)
				*out++ = K-1;												 // this differs from the paper because we want 0-based indices
				r -= BinomialCoefficient< safe >(K-1, m); // TODO: how to fix this 
			}
			*out++ = r;
		}
	}

	// Unrank subsets wrapper
	template< bool colex = true, typename InputIt, typename OutputIt >
	inline void unrank_combs(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		if ((BC.pre_n < n) || (BC.pre_k < k)){ BC.precompute(n,k); } // precompute binomial coefficients
		if constexpr(colex){
			unrank_colex< false >(s, e, n, k, out);	// pass safe = false to use precomputed table
		} else {
			unrank_lex< false >(s, e, n, k, out); // pass safe = false to use precomputed table
		}
	}

}; // namespace combinatorial

//   // Lexicographically unrank subsets wrapper
// 	template< size_t k, typename InputIt, typename Lambda >
// 	inline void lex_unrank_f(InputIt s, const InputIt e, const size_t n, Lambda f){
//     if constexpr (k == 2){
//       std::array< I, 2 > edge;
//       for (; s != e; ++s){
//         lex_unrank_2(*s, n, edge.begin());
//         f(edge);
//       } 
//     } else if (k == 3){
//       std::array< I, 3 > triangle;
//       for (; s != e; ++s){
//         lex_unrank_k(*s, n, 3, triangle.begin());
// 				f(triangle);
//       }
// 		} else {
//       std::array< I, k > simplex;
//       for (; s != e; ++s){
//         lex_unrank_k(*s, n, k, simplex.begin());
// 				f(simplex);
//       }
//     }
// 	}

	
// 	[[nodiscard]]
// 	inline auto lex_unrank_2_array(const index_t r, const index_t n) noexcept -> std::array< I, 2 > {
// 		auto i = static_cast< index_t >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
// 		auto j = static_cast< index_t >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
// 		return(std::array< I, 2 >{ i, j });
// 	}
	
// 	[[nodiscard]]
// 	inline auto lex_unrank(const size_t rank, const size_t n, const size_t k) -> std::vector< index_t > {
// 		if (k == 2){
// 			auto a = lex_unrank_2_array(rank, n);
// 			std::vector< index_t > out(a.begin(), a.end());
// 			return(out);
// 		} else {
// 			std::vector< index_t > out; 
// 			out.reserve(k);
// 			lex_unrank_k(rank, n, k, std::back_inserter(out));
// 			return(out);
// 		}
// 	}
	

// 	template< typename Lambda >
// 	void apply_boundary(const size_t r, const size_t n, const size_t k, Lambda f){
// 		// Given a p-simplex's rank representing a tuple of size p+1, enumerates the ranks of its (p-1)-faces, calling Lambda(*) on its rank
// 		using combinatorial::I; 
// 		switch(k){
// 			case 0: { return; }
// 			case 1: {
// 				f(r);
// 				return;
// 			}
// 			case 2: {
// 				auto p_vertices = std::array< I, 2 >();
// 				lex_unrank_2(static_cast< index_t >(r), static_cast< index_t >(n), begin(p_vertices));
// 				f(p_vertices[0]);
// 				f(p_vertices[1]);
// 				return;
// 			}
// 			case 3: {
// 				auto p_vertices = std::array< I, 3 >();
// 				lex_unrank_k(r, n, k, begin(p_vertices));
// 				f(lex_rank_2(p_vertices[0], p_vertices[1], n));
// 				f(lex_rank_2(p_vertices[0], p_vertices[2], n));
// 				f(lex_rank_2(p_vertices[1], p_vertices[2], n));
// 				return; 
// 			} 
// 			default: {
// 				auto p_vertices = std::vector< index_t >(0, k);
// 				lex_unrank_k(r, n, k, p_vertices.begin());
// 				const index_t N = BinomialCoefficient(n, k); 
// 				combinatorial::for_each_combination(begin(p_vertices), begin(p_vertices)+2, end(p_vertices), [&](auto a, auto b){
// 					f(lex_rank_k(a, n, k, N));
// 					return false; 
// 				});
// 				return; 
// 			}
// 		}
// 	} // apply boundary 

// 	template< typename OutputIt >
// 	void boundary(const size_t p_rank, const size_t n, const size_t k, OutputIt out){
// 		apply_boundary(p_rank, n, k, [&out](auto face_rank){
// 			*out = face_rank;
// 			out++;
// 		});
// 	}
// } // namespace combinatorial


#endif 