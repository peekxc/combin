#include "combinatorial.h"
#include <cinttypes>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
// #include <pybind11/eigen.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <iterator>     // std::back_inserter
#include <vector>       // std::vector
#include <algorithm>    // std::copy
using std::vector; 

// This function assumes the incoming array has combinations sorted in descending order 
auto rank_combs_sorted(py::array_t< uint16_t, py::array::c_style | py::array::forcecast > combs, const size_t n, bool colex = true) noexcept -> py::array_t< int64_t > {
  py::buffer_info buffer = combs.request();
  uint16_t* p = static_cast< uint16_t* >(buffer.ptr);
  const size_t N = buffer.size;
  const size_t k = static_cast< size_t >(combs.shape(1));
  vector< int64_t > ranks; 
  ranks.reserve(static_cast< int64_t >(N/k));
  auto out = std::back_inserter(ranks);
  combinatorial::BC.precompute(n, k);
	if (colex) {
		combinatorial::rank_colex< false >(p, p+N, n, k, out);
	} else {
    combinatorial::rank_lex< false >(p, p+N, n, k, out);
	}
  return py::cast(ranks); 
}

// Ranks arbitrarily-sized integer combinations from a list into an array 
auto rank_combs_unsorted(py::list combs, const size_t n, bool colex = true) -> py::array_t< uint64_t >{
  std::vector< uint64_t > output_ranks;
  output_ranks.reserve(combs.size());
  auto out = std::back_inserter(output_ranks);
  if (colex) {
    for (py::handle obj: combs) {
      auto s = obj.cast< std::vector< uint16_t > >();
      std::sort(s.begin(), s.end(), std::greater<>());
      *out++ = combinatorial::rank_colex_k(s.begin(), s.size());
    }
  } else {
    for (py::handle obj: combs) {
      auto s = obj.cast< std::vector< uint16_t > >();
      std::sort(s.begin(), s.end(), std::less<>());
      const size_t k = s.size();
      const size_t N = combinatorial::BinomialCoefficient< true >(n, k);
      *out++ = combinatorial::rank_lex_k(s.begin(), n, k, N);
    }
	}
  return py::cast(output_ranks);
}

auto comb1(const py::array_t< uint64_t >& N, const py::array_t< uint64_t >& K) -> py::array_t< uint64_t > {
  if (N.size() != K.size()){ throw std::invalid_argument("N adn K must match."); }
  const size_t array_sz = N.size();  
  auto output_combs = std::vector< uint64_t >();\
  output_combs.reserve(array_sz);
  auto out = std::back_inserter(output_combs);
  auto NA = N.unchecked< 1 >(), KA = K.unchecked< 1 >();
  for (size_t i = 0; i < array_sz; ++i){
    *out++ = combinatorial::BinomialCoefficient< true >(NA(i), KA(i));
  }
  return py::cast(output_combs);
}

auto comb2(const py::array_t< uint64_t >& N, const uint16_t k) -> py::array_t< uint64_t > {
  const size_t array_sz = N.size();  
  auto output_combs = std::vector< uint64_t >();\
  output_combs.reserve(array_sz);
  auto out = std::back_inserter(output_combs);
  auto NA = N.unchecked< 1 >();
  for (size_t i = 0; i < array_sz; ++i){
    *out++ = combinatorial::BinomialCoefficient< true >(NA(i), k);
  }
  return py::cast(output_combs);
}

auto comb3(const uint16_t N, const py::array_t< uint64_t >& K) -> py::array_t< uint64_t > {
  const size_t array_sz = K.size();  
  auto output_combs = std::vector< uint64_t >();
  output_combs.reserve(array_sz);
  auto out = std::back_inserter(output_combs);
  auto KA = K.unchecked< 1 >();
  for (size_t i = 0; i < array_sz; ++i){
    *out++ = combinatorial::BinomialCoefficient< true >(N, KA(i));
  }
  return py::cast(output_combs);
}

auto comb4(const uint16_t N, const uint16_t K) -> uint64_t {
  return combinatorial::BinomialCoefficient< true >(N, K);
}

auto comb5(const py::array_t< uint64_t >& N, const py::array_t< uint64_t >& K, const size_t max_n, const size_t max_k) -> py::array_t< uint64_t > {
  if (N.size() != K.size()){ throw std::invalid_argument("N adn K must match."); }
  combinatorial::BC.precompute(max_n, max_k);
  
  const size_t array_sz = N.size();  
  auto output_combs = std::vector< uint64_t >();\
  output_combs.reserve(array_sz);
  auto out = std::back_inserter(output_combs);
  auto NA = N.unchecked< 1 >(), KA = K.unchecked< 1 >();
  for (size_t i = 0; i < array_sz; ++i){
    *out++ = combinatorial::BinomialCoefficient< false >(NA(i), KA(i));
  }
  return py::cast(output_combs);
}

auto unrank_combranks_array(
  const py::array_t< uint64_t >& ranks, 
  const size_t n,  
  const size_t k, 
  const bool colex,
  py::array_t< uint16_t, py::array::c_style | py::array::forcecast >& out
){
  const uint64_t* inp_ptr = static_cast< const uint64_t* >(ranks.data()); 
  // py::buffer_info rank_buffer = ranks.request();
  uint16_t* out_ptr = static_cast< uint16_t* >(out.mutable_data()); 
  // combinatorial::unrank_lex(inp_ptr, inp_ptr + ranks.size(), n, k, out_ptr);
  if (colex){
    combinatorial::unrank_combs< true >(inp_ptr, inp_ptr + ranks.size(), n, k, out_ptr);
  } else {
    combinatorial::unrank_combs< false >(inp_ptr, inp_ptr + ranks.size(), n, k, out_ptr);
  }
	// inline void unrank_lex(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
}

auto unrank_combranks_array_full_k(
  const py::array_t< uint64_t >& ranks, 
  const size_t n,  
  const py::array_t< uint16_t >& K, 
  const size_t max_k, 
  const bool colex,
  py::array_t< uint16_t, py::array::c_style | py::array::forcecast >& out
){
  // const uint64_t* inp_ptr = static_cast< const uint64_t* >(ranks.data()); 
  uint16_t* out_ptr = static_cast< uint16_t* >(out.mutable_data()); 
  // Moved to precondition
  // // Preallocate the array to store the integers
  // const size_t array_sz = std::accumulate(K.begin(), K.end(), 0);
  // auto output_combs = std::vector< uint64_t >();
  // output_combs.reserve(array_sz);
  // auto out = std::back_inserter(output_combs);
  
  // Do the unranking
  combinatorial::BC.precompute(n, max_k); // precompute all (n*k) binomial coefficients 
  const uint64_t* RA = static_cast< const uint64_t* >(ranks.data()); 
  const uint16_t* KA = static_cast< const uint16_t* >(K.data()); 
  const size_t M = static_cast< const size_t >(ranks.shape(0));
  if (colex){
    for (size_t i = 0; i < M; ++i){
      combinatorial::unrank_colex< true >(RA+i, RA+i+1, n, KA[i], out_ptr);
      std::reverse(out_ptr, out_ptr + KA[i]);
      out_ptr += KA[i];
    }
  } else {
    for (size_t i = 0; i < M; ++i){
      combinatorial::unrank_lex< true >(RA+i, RA+i+1, n, KA[i], out_ptr);
      out_ptr += KA[i];
    }
  }
}

// using combinatorial::index_t; 
// auto unrank_combs_unsorted(
//   const py::array_t< uint64_t >& ranks, 
//   const size_t n,  
//   const py::array_t< uint8_t >& K, 
//   const bool colex,
//   py::array_t< uint16_t, py::array::c_style | py::array::forcecast >& out
// ) -> py::array_t< uint64_t >{
//   std::vector< uint64_t > output_ranks;
//   output_ranks.reserve(combs.size());
//   auto out = std::back_inserter(output_ranks);
//   if (colex) {
//     for (py::handle obj: combs) {
//       auto s = obj.cast< std::vector< uint16_t > >();
//       std::sort(s.begin(), s.end(), std::greater<>());
//       *out++ = combinatorial::rank_colex_k(s.begin(), s.size());
//     }
//   } else {
//     for (py::handle obj: combs) {
//       auto s = obj.cast< std::vector< uint16_t > >();
//       std::sort(s.begin(), s.end(), std::less<>());
//       const size_t k = s.size();
//       const size_t N = combinatorial::BinomialCoefficient< true >(n, k);
//       *out++ = combinatorial::rank_lex_k(s.begin(), n, k, N);
//     }
// 	}
//   return py::cast(output_ranks);
// }

// 5,3,0
// auto enumerate_cofacets(index_t r, const size_t k, const size_t n) -> py::array_t< uint64_t > {
//   std::vector< uint64_t > facet_ranks;
//   index_t idx_below = r;
//   index_t idx_above = 0; 
//   index_t j = n - 1;
//   for (index_t l = 0; l < k - 1; ++l){

//   }
// }


// // Given these two functions, we should be able to bulk-process apparent pairs...
// auto enumerate_facets(index_t r, const size_t k, const size_t n) -> py::array_t< uint64_t > {
//   std::vector< uint64_t > facet_ranks;
//   index_t idx_below = r;
//   index_t idx_above = 0; 
//   index_t j = n - 1;
//   for (index_t l = 0; l < k - 1; ++l){

//   }
//   // for (index_t ki = 0; ki <= k; ++ki){
//   //   j = combinatorial::get_max_vertex< true >(idx_below, k + 1, j);
//   //   index_t c = combinatorial::BinomialCoefficient< true >(j, k + 1);
//   //   index_t face_index = idx_above - c + idx_below;
//   //   idx_below -= c;
//   //   idx_above += combinatorial::BinomialCoefficient< true >(j, k);
//   //   // --k;
//   //   facet_ranks.push_back(face_index);   
//   // }
//   return py::cast(facet_ranks);
// }




// auto unrank_combs(py::array_t< int > ranks, const int n, const int k) -> py::array_t< int > {
//   py::buffer_info buffer = ranks.request();
//   int* r = static_cast< int* >(buffer.ptr);
//   const size_t N = buffer.size;
//   vector< int > simplices; 
//   simplices.reserve(static_cast< int >(N*k));
//   auto out = std::back_inserter(simplices);
//   combinatorial::unrank_lex(r, r+N, size_t(n), size_t(k), out);
//   return py::cast(simplices);
// }

// auto boundary_ranks(const int p_rank, const int n, const int k) -> py::array_t< int > {
//   vector< int > face_ranks = vector< int >();
// 	combinatorial::apply_boundary(p_rank, n, k, [&face_ranks](size_t r){
//     face_ranks.push_back(r);
//   });
//   return py::cast(face_ranks);
// }

void unrank_colex_bench(
  const py::array_t< combinatorial::index_t >& ranks, 
  const size_t n,  
  const size_t m,
  const bool use_lb, 
  const bool use_exp, 
  const size_t C,
  py::array_t< uint16_t, py::array::c_style | py::array::forcecast >& out
){    
  if (n > 65535){ throw std::invalid_argument("n is too large; overflow detected!"); }
  uint16_t* out_ptr = static_cast< uint16_t* >(out.mutable_data()); 
  combinatorial::BC.precompute(n, m); // precompute binomial coefficients 
  auto rb = ranks.data(); 
  auto re = ranks.data() + ranks.size();
  if (use_lb && use_exp){
    // bool safe = true, bool use_lb = true, bool ExpSearch = false
    switch(C){
      case 0:
        combinatorial::unrank_colex< false, true, true, 0 >(rb, re, n, m, out_ptr); 
        break; 
      case 1: 
        combinatorial::unrank_colex< false, true, true, 1 >(rb, re, n, m, out_ptr);
        break; 
      case 2: 
        combinatorial::unrank_colex< false, true, true, 2 >(rb, re, n, m, out_ptr);
        break; 
      default: 
        combinatorial::unrank_colex< false, true, true, 3 >(rb, re, n, m, out_ptr);
        break; 
    }
  } else if (!use_lb && use_exp){
    switch(C){
      case 0:
        combinatorial::unrank_colex< false, false, true, 0 >(rb, re, n, m, out_ptr); 
        break; 
      case 1: 
        combinatorial::unrank_colex< false, false, true, 1 >(rb, re, n, m, out_ptr);
        break; 
      case 2: 
        combinatorial::unrank_colex< false, false, true, 2 >(rb, re, n, m, out_ptr);
        break; 
      default: 
        combinatorial::unrank_colex< false, false, true, 3 >(rb, re, n, m, out_ptr);
        break; 
    }
  }
  else if (use_lb && !use_exp){
    switch(C){
      case 0:
        combinatorial::unrank_colex< false, true, false, 0 >(rb, re, n, m, out_ptr); 
        break; 
      case 1: 
        combinatorial::unrank_colex< false, true, false, 1 >(rb, re, n, m, out_ptr);
        break; 
      case 2: 
        combinatorial::unrank_colex< false, true, false, 2 >(rb, re, n, m, out_ptr);
        break; 
      default: 
        combinatorial::unrank_colex< false, true, false, 3 >(rb, re, n, m, out_ptr);
        break; 
    }
  }
  else
    switch(C){
      case 0:
        combinatorial::unrank_colex< false, false, false, 0 >(rb, re, n, m, out_ptr); 
        break; 
      case 1: 
        combinatorial::unrank_colex< false, false, false, 1 >(rb, re, n, m, out_ptr);
        break; 
      case 2: 
        combinatorial::unrank_colex< false, false, false, 2 >(rb, re, n, m, out_ptr);
        break; 
      default: 
        combinatorial::unrank_colex< false, false, false, 3 >(rb, re, n, m, out_ptr);
        break; 
    }
  }
using combinatorial::index_t; 
// Package: pip install --no-deps --no-build-isolation --editable .
// Compile: clang -Wall -fPIC -c src/pbsig/combinatorial.cpp -std=c++20 -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 
PYBIND11_MODULE(_combinatorial, m) {
  m.doc() = "Combinatorial module";
  m.def("rank_combs_sorted", &rank_combs_sorted);
  m.def("rank_combs_unsorted", &rank_combs_unsorted);
  m.def("unrank_combs", &unrank_combranks_array);
  m.def("unrank_combs_k", &unrank_combranks_array_full_k);
  m.def("comb", &comb1);
  m.def("comb", &comb2);
  m.def("comb", &comb3);
  m.def("comb", &comb4);
  m.def("comb", &comb5);
  m.def("unrank_colex_bench", &unrank_colex_bench);
  m.def("get_max_vertex", [](const index_t r, const index_t m, const index_t n, bool use_lb = true, size_t C = 0 ){
    return use_lb ? combinatorial::get_max_vertex< true, true >(r, m, n) : combinatorial::get_max_vertex< true, false >(r, m, n);
  });
  m.def("find_k", [](const index_t r, const index_t m){
    return combinatorial::find_k(r, m);
  });
  m.def("get_max", [](const index_t r, const index_t m, const index_t n){
		const auto pred = [r,m](index_t w) -> bool { return combinatorial::BinomialCoefficient< true >(w, m) <= r; };
    return combinatorial::get_max(n, m-1, pred);
  });
  // m.def("facet_ranks", &enumerate_facets);
  // m.def("unrank_combs", &unrank_combs);
  // m.def("boundary_ranks", &boundary_ranks);
  // m.def("interval_cost", &pairwise_cost);
  // m.def("vectorized_func", py::vectorize(my_func));s
  //m.def("call_go", &call_go);
}