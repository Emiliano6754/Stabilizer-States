#ifndef GF2NH
#define GF2NH
#include<x86intrin.h>
#include<string>

__m128i read_polynomial(const std::string &polynomial_filename, const unsigned int &N);
unsigned int GF2N_pol_mult(const unsigned int &a, const unsigned int &b, const unsigned int &N);
void read_basis_from_generator(const std::string &basis_filename, unsigned int* basis, const unsigned int &N);
void change_basis(const unsigned int &element, const unsigned int* basis, unsigned int &transformed_element, const unsigned int &N);
unsigned int change_basis(const unsigned int &element, const unsigned int* basis, const unsigned int &N);
void GF2N_invert_matrix(const unsigned int* matrix, unsigned int* inverse_matrix, const unsigned int &N);
void initialize_self_dual_basis(unsigned int* self_dual_basis, unsigned int* generator_basis, const unsigned int &N);

#endif