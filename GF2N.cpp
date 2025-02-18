#include "GF2N.h"
#include<x86intrin.h>
#include<fstream>
#include<string>
#include<sstream>
#include<iostream>

// This should work as long as there are less than 32 qubits. For N > 32, the getters (_mm_cvtsi128_si64) must take into account that some part of the numbers is also stored in the other 64 bit half. The setters in first and second are also wrong for this
unsigned long long GF2N_pol_mult(const unsigned int &a, const unsigned int &b) {
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

// Returns in base the coefficients in the generator basis of a self-dual basis. Can only hold 32 qubits bases and the N = 1 qubits is trivialized, so that this returns an error in that case. Assumes base points to N*sizeof(unsigned int) allocated space in memory
void read_basis_from_generator(const std::string &basis_filename, unsigned int &N, unsigned int* basis) {
    std::ifstream file(basis_filename);
    std::string line;
    size_t currentRow = 0;
    if (N < 2) {
        std::cerr << "Error: The basis for the N = 1 qubit case is trivial" << std::endl;
    }
    unsigned int basisRow = N - 2;
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << basis_filename << std::endl;
        return;
    }
    
    while (std::getline(file, line)) {
        if (currentRow == basisRow) {
            std::stringstream ss(line);
            std::string binaryStr;
            
            unsigned int j = 0;
            while (std::getline(ss, binaryStr, ',')) {
                basis[j] = std::stoul(binaryStr, nullptr, 2);
                j++;
            }
            break;
        }
        currentRow++;
    }
    
    file.close();
}

// Changes element to another basis, given the expansion of the current basis in terms of the new basis. This expansion is assumed to be contained in basis, with N elements.
void change_basis(const unsigned int &element, const unsigned int* basis, const unsigned int &N, unsigned int &transformed_element) {
    transformed_element = 0;
    for (unsigned int j = 0; j < N; j++) {
        transformed_element ^= ((element >> j) & 1) * basis[j];
    }
}

// Calculates the inverse of matrix, where the binary decomposition of each element is taken as a row. Assumes both matrix and inverse_matrix hold space for N elements. Only works for up to 32 qubits. This can be seen in that it takes only unsigned ints, so that no error can happen. Augmented matrix may be changed to two separate matrices, for which more qubits can be added
void GF2N_invert_matrix(const unsigned int* matrix, const unsigned int &N, unsigned int* inverse_matrix) {
    unsigned long long* augmented_matrix = static_cast<unsigned long long*>( _malloca(N * sizeof(unsigned long long)) );
    for (unsigned int j = 0; j < N; j++) {
        augmented_matrix[j] = (static_cast<unsigned long long>(matrix[j]) << N) | (1 << (N - j));
    }
    unsigned long long pivot = 1 << (2*N - 1);
    for (unsigned int j = 0; j < N; j++) {
        // Find pivot elements and order them accordingly 
        for (unsigned int k = j; k < N; k++) {
            if (augmented_matrix[k] & pivot) {
                std::swap(augmented_matrix[j], augmented_matrix[k]);
                break;
            }
        }
        // Eliminate the remaining ones in that pivot
        for (unsigned int k = 0; k < N; k++) {
            if (augmented_matrix[k] & pivot && k != j) {
                augmented_matrix[k] ^= augmented_matrix[j];
            }
        }
        pivot >>= 1;
    }
    unsigned long long mask = (1 << N) - 1;
    for (unsigned int j = 0; j < N; j++) {
        inverse_matrix[j] = static_cast<unsigned int>(augmented_matrix[j] & mask);
    }
}