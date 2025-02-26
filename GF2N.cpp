#include "GF2N.h"
#include<x86intrin.h>
#include<fstream>
#include<string>
#include<sstream>
#include<iostream>
#include<bitset> // Easy binary printing for debugging
#include<algorithm> // Reverse raw pointer array

// Returns the number on the (N-2)th row of polynomial filename. Can be used to parse the full, reduced and reducing polynomials. Can read polynomials of up to 64 qubits
__m128i read_polynomial(const std::string &polynomial_filename, const unsigned int &N) {
    std::ifstream file(polynomial_filename);
    std::string line;
    unsigned int currentRow = 0;
    unsigned long long polynomial;
    if (N < 2) {
        std::cerr << "Error: The basis for the N = 1 qubit case is trivial" << std::endl;
    }
    unsigned int basisRow = N - 2;
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << polynomial_filename << std::endl;
        return _mm_set_epi64x(0, 0);
    }
    
    while (std::getline(file, line)) {
        if (currentRow == basisRow) {
            std::stringstream ss(line);
            std::string binaryStr;
            
            std::getline(ss, binaryStr, ',');
            polynomial = std::stoull(binaryStr, nullptr, 2);
            break;
        }
        currentRow++;
    }
    
    file.close();
    return _mm_set_epi64x(0, polynomial);
}

// This should work as long as there are less than 32 qubits. For N > 32, the getters (_mm_cvtsi128_si64) must take into account that some part of the numbers is also stored in the other 64 bit half. The setters in first and second are also wrong for this
unsigned int GF2N_pol_mult(const unsigned int &a, const unsigned int &b, const unsigned int &N) {
    __m128i prod;
    const static unsigned int mask = (1<<N) - 1;
    const static __m128i pol = read_polynomial("C:\\dev\\Campos\\Data\\red_gen_pols.txt", N);
    const static __m128i qplus = read_polynomial("C:\\dev\\Campos\\Data\\reducing_pols.txt", N); // The quotient of x^2N by the generating polynomial in GF(2)
    
    unsigned long long res1, res2;
    __m128i first = _mm_set_epi32(0,0,0,a);
    __m128i second = _mm_set_epi32(0,0,0,b);
    prod = _mm_clmulepi64_si128(first, second, 0);
    unsigned long long product = _mm_cvtsi128_si64(prod);
    res1 = product & mask;
    res2 = product >> N; // Technically, it should only take the next N bits, but the remaining digits are zero always. It may be possible to improve this with _bextr_u32?
    first = _mm_set_epi64x(0, res2);
    prod = _mm_clmulepi64_si128(first, qplus, 0);
    product = _mm_cvtsi128_si64(_mm_srli_epi64(prod, N)); // Select the first s most significant terms of a 2s-1 polynomial (it has 2s elements, remove the last s elements). The result is a s-1 degree polynomial
    first = _mm_set_epi64x(0, product);
    res2 =_mm_cvtsi128_si64(_mm_clmulepi64_si128(first, pol, 0)) & mask; // Selects the t least significant terms of the product g*(x) M^s( c q^+ )
    return res1 ^ res2;
}

// Returns in base the coefficients in the generator basis of a self-dual basis. Can only hold 32 qubits bases and the N = 1 qubits is trivialized, so that this returns an error in that case. Assumes base points to N*sizeof(unsigned int) allocated space in memory
void read_basis_from_generator(const std::string &basis_filename, unsigned int* basis, const unsigned int &N) {
    std::ifstream file(basis_filename);
    std::string line;
    unsigned int currentRow = 0;
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
void change_basis(const unsigned int &element, const unsigned int* basis, unsigned int &transformed_element, const unsigned int &N) {
    transformed_element = 0;
    for (unsigned int j = 0; j < N; j++) {
        transformed_element ^= ((element >> j) & 1) * basis[j];
    }
}

unsigned int change_basis(const unsigned int &element, const unsigned int* basis, const unsigned int &N) {
    unsigned int transformed_element = 0;
    for (unsigned int j = 0; j < N; j++) {
        transformed_element ^= ((element >> j) & 1) * basis[j];
    }
    return transformed_element;
}

// Prints the augmented matrix for debugging purposes
void print_augmented_matrix(const unsigned long long* augmented_matrix, const unsigned int &N) {
    for (unsigned int j = 0; j < N; j++) {
        std::cout << std::bitset<8>(augmented_matrix[j]) << std::endl;
    }
    std::cout << "_____________" << std::endl;
}

// Calculates the inverse of matrix, where the binary decomposition of each element is taken as a row. Assumes both matrix and inverse_matrix hold space for N elements. Only works for up to 32 qubits. This can be seen in that it takes only unsigned ints, so that no error can happen. Augmented matrix may be changed to two separate matrices, for which more qubits can be added
void GF2N_invert_matrix(const unsigned int* matrix, unsigned int* inverse_matrix, const unsigned int &N) {
    unsigned long long* augmented_matrix = static_cast<unsigned long long*>( _malloca(N * sizeof(unsigned long long)) );
    for (unsigned int j = 0; j < N; j++) {
        augmented_matrix[j] = (static_cast<unsigned long long>(matrix[j]) << N) | (1 << j);
    }
    unsigned long long pivot = 1 << (2*N - 1);
    for (unsigned int j = 0; j < N; j++) {
        // Find pivot elements and order them accordingly 
        for (unsigned int k = j; k < N; k++) {
            if (augmented_matrix[k] & pivot) {
                std::swap(augmented_matrix[j], augmented_matrix[k]);
                // print_augmented_matrix(augmented_matrix, N);
                break;
            }
        }
        // Eliminate the remaining ones in that pivot
        for (unsigned int k = 0; k < N; k++) {
            if (augmented_matrix[k] & pivot && k != j) {
                augmented_matrix[k] ^= augmented_matrix[j];
                // print_augmented_matrix(augmented_matrix, N);
            }
        }
        pivot >>= 1;
    }
    unsigned long long mask = (1 << N) - 1;
    for (unsigned int j = 0; j < N; j++) {
        inverse_matrix[j] = static_cast<unsigned int>(augmented_matrix[j] & mask);
    }
}

void initialize_self_dual_basis(unsigned int* self_dual_basis, unsigned int* generator_basis, const unsigned int &N) {
    read_basis_from_generator("C:\\dev\\Campos\\Data\\self_adj_bases.txt", self_dual_basis, N);
    GF2N_invert_matrix(self_dual_basis, generator_basis, N);
    std::reverse(generator_basis, generator_basis + N);
}