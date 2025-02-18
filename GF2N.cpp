#include "GF2N.h"
#include<x86intrin.h>

// This should work as long as there are less than 32 qubits. For N > 32, the getters (_mm_cvtsi128_si64) must take into account that some part of the numbers is also stored in the other 64 bit half. The setters in first and second are also wrong for this
unsigned long long GF2Nmult(const unsigned int &a, const unsigned int &b) {
    __m128i prod;
    // unsigned int pol = 0b1011;
    unsigned int N = 8;
    unsigned int mask = (1<<N) - 1;
    // __mmask32 mmask = _cvtu32_mask32(mask);
    unsigned long long res1, res2;
    __m128i qplus = _mm_set_epi32(0,0,0,0b1011110); // x^16 mod(generator polynomial) = x^6 + x^4 + x^3 + x^2 + x. This needs to be precalculated
    __m128i pol = _mm_set_epi32(0,0,0,0b11011); // The t least significant terms of the polynomial

    __m128i first = _mm_set_epi32(0,0,0,a);
    __m128i second = _mm_set_epi32(0,0,0,b);
    prod = _mm_clmulepi64_si128(first, second, 0);
    unsigned long long product = _mm_cvtsi128_si64(prod);
    res1 = product & mask;
    res2 = product >> N; // Technically, it should only take the next N bits, but the remaining digits are zero always. It may be possible to improve this with _bextr_u32?
    first = _mm_set_epi64x(0, res2);
    prod = _mm_clmulepi64_si128(first, qplus, 0);
    product = _mm_cvtsi128_si64(prod); // Select the first s most significant terms of a 2s-1 polynomial (it has 2s elements, remove the last s elements). The result is a s-1 degree polynomial
    first = _mm_set_epi64x(0, product);
    res2 =_mm_cvtsi128_si64(_mm_clmulepi64_si128(first, pol, 0)) & mask; // Selects the t least significant terms of the product g*(x) M^s( c q^+ )
    return res1 ^ res2;
}