#ifndef GF2NH
#define GF2NH

unsigned long long GF2N_pol_mult(const unsigned int &a, const unsigned int &b);
void read_basis_from_generator(const std::string &basis_filename, unsigned int &N, unsigned int* basis);
void change_basis(const unsigned int &element, const unsigned int* basis, const unsigned int &N, unsigned int &transformed_element);
void GF2N_invert_matrix(const unsigned int* matrix, const unsigned int &N, unsigned int* inverse_matrix);

#endif