#pragma once
#ifndef CBRT_RE_HPP_
#define CBRT_RE_HPP_
//=================================================================================================
//  cbrt.hpp
//  #include "cbrt.hpp"
//  Created: 03.09.2019
//  License  MPL-2.0
//  Remotion (C) 2019 - All Rights Reserved
// [mm_cbrt_ps on godbolt](https://godbolt.org/z/AB5W52)
//=================================================================================================

#if defined _MSC_VER
#include <intrin.h>

#else // clang and gcc  [ -O3 -mavx2 -mfma -ffast-math]
#include <x86intrin.h>

#ifndef __forceinline
#define __forceinline inline __attribute__((always_inline))
#endif 

#endif

namespace re {

/// Reinterprets the four single precision floating point values in a as four 32-bit integers
__forceinline __m128i as_int(__m128 f)  { return _mm_castps_si128(f); } // the same as as_uint
/// Reinterprets the two double precision floating point values in a as two 64-bit integers
__forceinline __m128i as_int(__m128d f) { return _mm_castpd_si128(f); }

/// Reinterprets the four four 32-bit integers values in as single precision floats
__forceinline __m128 as_float(__m128i i) { return _mm_castsi128_ps(i); }
/// Reinterprets the two 64-bit integers values in as doubles
__forceinline __m128d as_double(__m128i i) { return _mm_castsi128_pd(i); }


/// Reinterprets the eight single precision floating point values in a as four 32-bit integers
__forceinline __m256i as_int(__m256 f) { return _mm256_castps_si256(f); } // the same as as_uint
/// Reinterprets the four double precision floating point values in a as four 64-bit integers
__forceinline __m256i as_int(__m256d f) { return _mm256_castpd_si256(f); }

/// Reinterprets the eight 32-bit integers values in as single precision floats
__forceinline __m256 as_float(__m256i i) { return _mm256_castsi256_ps(i); }
/// Reinterprets the four 64-bit integers values in as doubles
__forceinline __m256d as_double(__m256i i) { return _mm256_castsi256_pd(i); }


/// approximate cube root 
inline __m128 mm_cbrt_ps(__m128 x0) {
	const auto sign_mask = _mm_set1_epi32(0x7fffffff);
	const auto two = _mm_set1_ps(2);
	const auto third = _mm_set1_ps(1.0 / 3.0);

	auto ix = as_int(x0);
	const auto ix0 = ix; // value with sign 
	ix = _mm_and_si128(ix, sign_mask); // abs
	const auto abs_x = x0; 

	// ix/3 is approximated as ix/4 + ix/16 + ix/64 + ix/256 + ... + 1/65536.
	ix = _mm_add_epi32(_mm_srli_epi32(ix, 2), _mm_srli_epi32(ix, 4));  // ix = ix/4 + ix/16;
	ix = _mm_add_epi32(ix, _mm_srli_epi32(ix, 4)); // ix = ix + ix/16; 
	ix = _mm_add_epi32(ix, _mm_srli_epi32(ix, 8)); // ix = ix + ix/256;
	ix = _mm_add_epi32(_mm_set1_epi32(0x2a5137a0), ix); // ix = 0x2a5137a0 + ix;        // Initial guess.

	auto x = as_float(ix);
	x = _mm_mul_ps(third, _mm_fmadd_ps(two, x, _mm_div_ps(abs_x, _mm_mul_ps(x, x))));   // Newton step.
	x = _mm_mul_ps(third, _mm_fmadd_ps(two, x, _mm_div_ps(abs_x, _mm_mul_ps(x, x))));   // Newton step.
	x = _mm_mul_ps(third, _mm_fmadd_ps(two, x, _mm_div_ps(abs_x, _mm_mul_ps(x, x))));   // Newton step.

	ix = as_int(x);
	ix = _mm_or_si128(ix, _mm_and_si128(ix0, _mm_set1_epi32(0x80000000u))); // copy sign:  ix = ix|ix0&-0x80000000
	return as_float(ix);
}


/// approximate cube root 
inline __m256 mm256_cbrt_ps(__m256 x0) {
	const auto sign_mask = _mm256_set1_epi32(0x7fffffff);
	const auto two = _mm256_set1_ps(2);
	const auto third = _mm256_set1_ps(1.0 / 3.0);

	auto ix = as_int(x0);
	const auto ix0 = ix; // value with sign 
	ix = _mm256_and_si256(ix, sign_mask); // abs
	const auto abs_x = x0;

	// ix/3 is approximated as ix/4 + ix/16 + ix/64 + ix/256 + ... + 1/65536.
	ix = _mm256_add_epi32(_mm256_srli_epi32(ix, 2), _mm256_srli_epi32(ix, 4));  // ix = ix/4 + ix/16;
	ix = _mm256_add_epi32(ix, _mm256_srli_epi32(ix, 4)); // ix = ix + ix/16; 
	ix = _mm256_add_epi32(ix, _mm256_srli_epi32(ix, 8)); // ix = ix + ix/256;
	ix = _mm256_add_epi32(_mm256_set1_epi32(0x2a5137a0), ix); // ix = 0x2a5137a0 + ix;        // Initial guess.

	auto x = as_float(ix);
	x = _mm256_mul_ps(third, _mm256_fmadd_ps(two, x, _mm256_div_ps(abs_x, _mm256_mul_ps(x, x))));   // Newton step.
	x = _mm256_mul_ps(third, _mm256_fmadd_ps(two, x, _mm256_div_ps(abs_x, _mm256_mul_ps(x, x))));   // Newton step.
	x = _mm256_mul_ps(third, _mm256_fmadd_ps(two, x, _mm256_div_ps(abs_x, _mm256_mul_ps(x, x))));   // Newton step.

	ix = as_int(x);
	ix = _mm256_or_si256(ix, _mm256_and_si256(ix0, _mm256_set1_epi32(0x80000000u))); // copy sign:  ix = ix|ix0&-0x80000000
	return as_float(ix);
}

} // namespace re 

#endif // CBRT_RE_HPP_