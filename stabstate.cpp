#include<iostream>
#include<fstream>
#include<complex>
#include<vector>
#include<algorithm>
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

// Equivalent to generate_xi_buffer, but for a xi_buffer allocated on the stack
void generate_stack_xi_buffer(double* xi_buffer, const unsigned int &max_weight, const std::complex<double> &xi) {
    double coeff = 1.0 / (1 + std::norm(xi));
    double ratio = (1 - std::norm(xi)) / (1 + std::norm(xi));
    for (unsigned int n = 0; n <= max_weight; n++) {
        xi_buffer[n] = coeff;
        coeff *= ratio;
    }
}

unsigned int get_max_degree(const unsigned int* Adj, const unsigned int &n_qubits) {
    unsigned int max = 0;
    for (unsigned int n = 0; n < n_qubits; n++) {
        if (max < std::popcount(Adj[n])) {
            max = std::popcount(Adj[n]);
        }
    }
    return max;
}

void graphQ(Eigen::MatrixXd &Qfunc, Eigen::Tensor<double,3> &sym_Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int* Adj) {
    const std::complex<double> xi = 0.5 * (sqrt(3)-1) * std::complex<double>(1.0,1.0);
    const double xi_sum = 2 * xi.real();
    const unsigned int max_weight = get_max_degree(Adj,n_qubits);
    double* xi_buffer = static_cast<double*>(alloca((max_weight+1) * sizeof(double)));
    generate_stack_xi_buffer(xi_buffer,max_weight,xi);

    #pragma omp parallel
    {
        double coeff = 0;
        #pragma omp for
        for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
            for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                coeff = 1;
                for (unsigned int j = 0; j < n_qubits; j++) {
                    coeff *= 1 + int(1 - 2 * ((alpha>>j) & 1)) * (1 - 2 * trace(Adj[j],beta)) * xi_buffer[std::popcount(Adj[j])] * xi_sum;
                }
                Qfunc(alpha,beta) = coeff / qubitstate_size;
                sym_Qfunc(std::popcount(alpha),std::popcount(beta),std::popcount(alpha^beta)) += Qfunc(alpha,beta);
            }
        }
    }
    
}

void symonly_graphQ(Eigen::Tensor<double,3> &sym_Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int* Adj) {
    const std::complex<double> xi = 0.5 * (sqrt(3)-1) * std::complex<double>(1.0,1.0);
    const double xi_sum = 2 * xi.real();
    const unsigned int max_weight = get_max_degree(Adj,n_qubits);
    double* xi_buffer = static_cast<double*>(alloca((max_weight+1) * sizeof(double)));
    generate_stack_xi_buffer(xi_buffer,max_weight,xi);

    #pragma omp parallel
    {
        double coeff = 0;
        #pragma omp for
        for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
            for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                coeff = 1;
                for (unsigned int j = 0; j < n_qubits; j++) {
                    coeff *= 1 + int(1 - 2 * ((alpha>>j) & 1)) * (1 - 2 * trace(Adj[j],beta)) * xi_buffer[std::popcount(Adj[j])] * xi_sum;
                }
                sym_Qfunc(std::popcount(alpha),std::popcount(beta),std::popcount(alpha^beta)) += coeff / qubitstate_size;
            }
        }
    }
    
}

// Initializes adjacency matrix, def controls whether all edges are connected or disconnected, with disconnected as default
inline void initAdj(unsigned int* Adj, const unsigned int &size, const unsigned int def=0) {
    if (size > 8*sizeof(unsigned int)) {
        std::cout << "Too many qubits, change unsigned int in adjacency matrices to use more" << std::endl;
    } else {
        unsigned int connection = (1 << (def * size)) - 1;
        for (unsigned int i = 0; i < size; i++) {
            Adj[i] = connection;
        }
    }
}

// Adds (or removes if already present) edge (a,b) from the adjacency matrix. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
inline void addEdge(unsigned int* Adj, const unsigned int &size, const unsigned int &a, const unsigned int &b) {
    if (size < a+1 || size < b+1) {
        std::cout << "Edges outside bounds" << std::endl;
    } else {
        Adj[a] = Adj[a] ^ (1 << b);
        Adj[b] = Adj[b] ^ (1 << a);
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

void generate_maxcon_symQ(const unsigned int &n_qubits) {
    const unsigned int qubitstate_size = 1 << n_qubits;
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    initAdj(Adj,n_qubits,1);
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits,n_qubits,n_qubits);
    auto start = std::chrono::high_resolution_clock::now();
    symonly_graphQ(sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating took " << duration.count() << "s" << std::endl;
    save_symQfunc(sym_Qfunc,"mc_q"+std::to_string(n_qubits)+".txt");
}

unsigned int parse_unsignedint(const std::string &input) {
    try {
        unsigned long u = std::stoul(input);
        if (u > std::numeric_limits<unsigned int>::max())
            throw std::out_of_range(input);

        return u;
    } catch (const std::invalid_argument& e) {
        std::cout << "Input could not be parsed: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Input out of range: " << e.what() << std::endl;
    }
    return 0;
}

int main() {
    std::string input;
    std::cout << "Enter the number of qubits" << std::endl;
    std::cin >> input;
    unsigned int n_qubits = parse_unsignedint(input);
    generate_maxcon_symQ(n_qubits);
    return 0;
}