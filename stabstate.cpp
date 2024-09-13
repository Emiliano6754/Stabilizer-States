#include<iostream>
#include<fstream>
#include<complex>
#include<vector>
#include<algorithm>
#include<utility> // std::pair
#include<bitset> // Print numbers as binary easily
#include<filesystem> // Current directory
#include<chrono> // Timing
#include<Eigen/Dense>
#include<unsupported/Eigen/CXX11/Tensor>

// Calculates the field-wise trace of alpha by calculating its hamming weight and returning the last bit (modulo 2)
inline int trace(const unsigned int &alpha) {
    return std::popcount(alpha) & 1;
}

// Calculates the trace of the product by doing bitwise and. Equivalent to calling trace(alpha&beta)
inline int trace(const unsigned int &alpha, const unsigned int &beta) {
    return std::popcount(alpha & beta) & 1;
}

// Returns (-1)^(a+b)
inline double sign(const unsigned int &a, const unsigned int &b) {
    return 1.0 - 2.0 * ( (a + b) & 1);
}

inline double sign(const unsigned int &a) {
    return 1.0 - 2.0 * (a & 1);
}

// Equivalent to generate_xi_buffer, but for a xi_buffer allocated on the stack
void generate_xi_buffers(double* norm_buffer, double* sum_buffer, std::complex<double>* subs_buffer, const unsigned int &n_qubits, const std::complex<double> &xi) {
    double norm_coeff = (1- std::norm(xi))/(1+std::norm(xi));
    double sum_coeff = (sqrt(3)-1)/(1+std::norm(xi));
    std::complex<double> subs_coeff = (std::conj(xi) - xi)/(1+std::norm(xi));
    norm_buffer[0] = 1;
    sum_buffer[0] = 1;
    subs_buffer[0] = 1;
    for (unsigned int n = 1; n <= n_qubits; n++) {
        norm_buffer[n] = norm_buffer[n-1] * norm_coeff;
        sum_buffer[n] = sum_buffer[n-1] * sum_coeff;
        subs_buffer[n] = subs_buffer[n-1] * subs_coeff;
    }
}

void generate_sum_buffers(std::vector<std::complex<double>> &xi_product, std::vector<unsigned int> &mu_sums, std::vector<unsigned int> &nu_sums, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const std::complex<double> &xi, const unsigned int* mu, const unsigned int* nu) {
    double* norm_buffer = static_cast<double*>(alloca((n_qubits+1) * sizeof(double)));
    double* sum_buffer = static_cast<double*>(alloca((n_qubits+1) * sizeof(double)));
    std::complex<double>* subs_buffer = static_cast<std::complex<double>*>(alloca((n_qubits+1) * sizeof(std::complex<double>)));
    generate_xi_buffers(norm_buffer,sum_buffer,subs_buffer,n_qubits,xi);
    #pragma omp parallel
    {
        unsigned int mu_sum = 0;
        unsigned int nu_sum = 0;
        unsigned int sign_sum = 0;
        unsigned int eta_n = 0;
        unsigned int hB = 0;
        unsigned int hC = 0;
        unsigned int hBpC = 0;
        #pragma omp for
        for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
            sign_sum = 0;
            mu_sum = ( (eta & 1) * mu[0]);
            nu_sum = ( (eta & 1) * nu[0]);
            for (unsigned int n = 1; n < n_qubits; n++) {
                eta_n = (eta >> n) & 1;
                sign_sum += eta_n * std::popcount( mu[n] & nu_sum ); // Needs checking. Seems to output correctly
                mu_sum = mu_sum ^ ( eta_n * mu[n]);
                nu_sum = nu_sum ^ ( eta_n * nu[n]);
            }
            mu_sums[eta] = mu_sum;
            nu_sums[eta] = nu_sum;
            // std::cout << std::bitset<8>(eta) << std::endl;
            // std::cout << "adj_sums(" << std::bitset<8>(eta) << ") = " << std::bitset<8>(adj_sum) << std::endl;
            // std::cout << "sign_sum(" << std::bitset<8>(eta) << ") = " << sign_sum << std::endl;
            hB = std::popcount(mu_sum);
            hC = std::popcount(nu_sum);
            hBpC = std::popcount(mu_sum ^ nu_sum);
            xi_product[eta] = sign(trace(mu_sum,nu_sum) + sign_sum) * norm_buffer[(hB - hC + hBpC)/2] * sum_buffer[(hC - hB + hBpC)/2] * subs_buffer[(hB + hC - hBpC)/2];
            // std::cout << "h(eta) = " << std::to_string(std::popcount(eta)) << " xi_prod(eta) = " << xi_product[eta] << std::endl;
        }
    }
}

void graphQ(Eigen::MatrixXd &Qfunc, Eigen::Tensor<double,3> &sym_Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int* mu, const unsigned int* nu) {
    const std::complex<double> xi = 0.5 * (sqrt(3)-1) * std::complex<double>(1.0,1.0);
    double denom = 1.0 / qubitstate_size;
    std::vector<std::complex<double>> xi_product(qubitstate_size); // Can be optimized to only store real value, as the imaginary part after summation should be zero
    std::vector<unsigned int> mu_sums(qubitstate_size);
    std::vector<unsigned int> nu_sums(qubitstate_size);
    generate_sum_buffers(xi_product,mu_sums,nu_sums,n_qubits,qubitstate_size,xi,mu,nu);
    #pragma omp parallel
    {
        std::complex<double> coeff = 0;
        #pragma omp for
        for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
            for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                coeff = 0;
                for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
                    coeff += sign(trace(alpha,nu_sums[eta]),trace(beta,mu_sums[eta])) * xi_product[eta];
                }
                Qfunc(alpha,beta) = coeff.real() * denom;
                // if (coeff.imag() > 0.1) {
                //     std::cout << coeff.imag() << std::endl;
                // }
                sym_Qfunc(std::popcount(alpha),std::popcount(beta),std::popcount(alpha^beta)) += Qfunc(alpha,beta);
            }
        }
    }
}

void symonly_graphQ(Eigen::Tensor<double,3> &sym_Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int* mu, const unsigned int* nu) {
    const std::complex<double> xi = 0.5 * (sqrt(3)-1) * std::complex<double>(1.0,1.0);
    double denom = 1.0 / qubitstate_size;
    std::vector<std::complex<double>> xi_product(qubitstate_size); // Can be optimized to only store real value, as the imaginary part after summation should be zero
    std::vector<unsigned int> mu_sums(qubitstate_size);
    std::vector<unsigned int> nu_sums(qubitstate_size);
    generate_sum_buffers(xi_product,mu_sums,nu_sums,n_qubits,qubitstate_size,xi,mu,nu);
    #pragma omp parallel
    {
        std::complex<double> coeff = 0;
        #pragma omp for
        for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
            for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                coeff = 0;
                for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
                    coeff += sign(trace(alpha,nu_sums[eta]),trace(beta,mu_sums[eta])) * xi_product[eta];
                }
                sym_Qfunc(std::popcount(alpha),std::popcount(beta),std::popcount(alpha^beta)) += coeff.real() * denom;
            }
        }
    }
}

// Calculates the expansion in the normal self-dual basis of the generators of a line with slope m. Warning: A normal self-dual basis doesn't exist if the number of qubits is a multiple of 4.
void calc_line_gens(unsigned int* mu, unsigned int* nu, const unsigned int &m, const unsigned int &n_qubits) {
    for (unsigned int i = 0; i < n_qubits; i++) {
        mu[i] = 1 << i;
        nu[i] = (m << i + (m >> (n_qubits - i))) & ((1 << n_qubits) - 1);
    }
}

void save_Qfunc(const Eigen::MatrixXd &Qfunc, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/Qfuncs/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    Eigen::IOFormat FullPrecision(Eigen::FullPrecision,0,"\n");
    if (output_file.is_open()) {
        output_file << Qfunc.format(FullPrecision) << std::endl;
    } else {
        std::cout << "Could not save Qfunc" << std::endl;
    }
}

void save_symQfunc(const Eigen::Tensor<double,3> &Qfunc, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/symQfuncs/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (output_file.is_open()) {
        for (unsigned int i = 0; i < Qfunc.dimension(0); i++) {
            for (unsigned int j = 0; j < Qfunc.dimension(1); j++) {
                for (unsigned int k = 0; k < Qfunc.dimension(2); k++) {
                    output_file << Qfunc(i,j,k) << "\n";
                }
            }
        }
    } else {
        std::cout << "Could not save Qfunc" << std::endl;
    }
}

// Calculates the symmetric Q function of a stabilizer state with a given set of generators and saves it to filename. Prints time taken to perform the calculations
void calc_save_symQ(const unsigned int &n_qubits, unsigned int* mu, unsigned int* nu, const std::string &filename) {
    const unsigned int qubitstate_size = 1 << n_qubits;
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    auto start = std::chrono::high_resolution_clock::now();
    graphQ(Qfunc,sym_Qfunc.setZero(), n_qubits, qubitstate_size, mu, nu);
    // symonly_graphQ(sym_Qfunc.setZero(), n_qubits, qubitstate_size, mu, nu);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating took " << duration.count() << "s" << std::endl;
    save_symQfunc(sym_Qfunc,filename);
    save_Qfunc(Qfunc,filename);
}

int main() {
    const unsigned int m = 0;
    const unsigned int n_qubits = 10;
    unsigned int* mu = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    unsigned int* nu = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    calc_line_gens(mu,nu,m,n_qubits);
    calc_save_symQ(n_qubits,mu,nu,"test.txt");

    return 0;
}