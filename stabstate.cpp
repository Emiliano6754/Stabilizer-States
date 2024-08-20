#include<iostream>
#include<fstream>
#include<complex>
#include<vector>
#include<algorithm>
#include<bitset> // Print numbers as binary easily
#include<filesystem> // Current directory
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

void graphQ(Eigen::MatrixXd &Qfunc, Eigen::Tensor<double,3> &sym_Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const std::vector<unsigned int> &Adj) {
    const std::complex<double> xi = 0.5 * (sqrt(3)-1) * std::complex<double>(1.0,1.0);
    const double xi_sum = 2 * xi.real();
    const unsigned int max_weight = std::popcount(*std::max_element(Adj.begin(), Adj.end(), [](const unsigned int &a, const unsigned int &b)
    {
        return std::popcount(a) < std::popcount(b);
    }));
    double* xi_buffer = static_cast<double*>(alloca((max_weight+1) * sizeof(double)));
    generate_stack_xi_buffer(xi_buffer,max_weight,xi);

    #pragma omp parallel
    {
        double coeff = 0;
        double hola = 0;
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

// inline void addEdge(std::vector<std::vector<unsigned int>>& Adj, const unsigned int& a, const unsigned int& b) {
//     if (Adj.size() < a+1 || Adj.size() < b+1 || Adj[b].size() < a+1 || Adj[a].size() < b+1) {
//         std::cout << "Edges outside bounds" << std::endl;
//     } else {
//         Adj[a][b] = 1;
//         Adj[b][a] = 1;
//     }
// }

// std::vector<std::vector<unsigned int>> initAdj(const unsigned int& size, const unsigned int def=0) {
//     std::vector<std::vector<unsigned int>> Adj(size);
//     Adj.assign(size, std::vector<unsigned int>(size));
//     return Adj;
// }

// Initializes adjacency matrix, def controls whether all edges are connected or disconnected, with disconnected as default
inline void initAdj(std::vector<unsigned int> &Adj, const unsigned int &size, const unsigned int def=0) {
    if (size > 8*sizeof(unsigned int)) {
        std::cout << "Too many qubits, change unsigned int in adjacency matrices to use more" << std::endl;
    } else {
        unsigned int connection = (1 << (def * size)) - 1;
        Adj.assign(size,connection);
    }
}

// Adds (or removes if already present) edge (a,b) from the adjacency matrix. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
inline void addEdge(std::vector<unsigned int> &Adj, const unsigned int &a, const unsigned int &b) {
    if (Adj.size() < a+1 || Adj.size() < b+1) {
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

int main() {
    const unsigned int n_qubits = 5;
    const unsigned int qubitstate_size = 1 << n_qubits;
    std::vector<unsigned int> Adj(n_qubits);
    initAdj(Adj,n_qubits,0);
    addEdge(Adj, 0, 2);
    // addEdge(Adj, 1, 3);
    // addEdge(Adj, 1, 4);
    // addEdge(Adj, 1, 5);
    // addEdge(Adj, 2, 6);
    // addEdge(Adj, 2, 3);
    // addEdge(Adj, 4, 3);
    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits,n_qubits,n_qubits);
    graphQ(Qfunc, sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
    save_Qfunc(Qfunc, "test.txt");
    save_symQfunc(sym_Qfunc,"symtest.txt");

    return 0;
}