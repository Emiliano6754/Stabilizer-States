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

unsigned int get_max_degree(const unsigned int* Adj, const unsigned int &n_qubits) {
    unsigned int max = 0;
    for (unsigned int n = 0; n < n_qubits; n++) {
        if (max < std::popcount(Adj[n])) {
            max = std::popcount(Adj[n]);
        }
    }
    return max;
}

void generate_sum_buffers(std::vector<std::complex<double>> &xi_product, std::vector<unsigned int> &adj_sums, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const std::complex<double> &xi, const unsigned int* Adj) {
    double* norm_buffer = static_cast<double*>(alloca((n_qubits+1) * sizeof(double)));
    double* sum_buffer = static_cast<double*>(alloca((n_qubits+1) * sizeof(double)));
    std::complex<double>* subs_buffer = static_cast<std::complex<double>*>(alloca((n_qubits+1) * sizeof(std::complex<double>)));
    generate_xi_buffers(norm_buffer,sum_buffer,subs_buffer,n_qubits,xi);
    #pragma omp parallel
    {
        unsigned int adj_sum = 0;
        unsigned int sign_sum = 0;
        unsigned int hB = 0;
        unsigned int hC = 0;
        unsigned int hBpC = 0;
        #pragma omp for
        for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
            sign_sum = 0;
            adj_sum = ( (eta & 1) * Adj[0]);
            for (unsigned int n = 1; n < n_qubits; n++) {
                adj_sum = adj_sum ^ ( ((eta >> n) & 1) * Adj[n]);
                sign_sum += ((eta >> n) & 1) * trace(Adj[n],eta & ((1 << n) - 1)); // Needs checking
            }
            adj_sums[eta] = adj_sum;
            hB = std::popcount(adj_sum);
            hC = std::popcount(eta);
            hBpC = std::popcount(adj_sum ^ eta);
            xi_product[eta] = (1.0 - 2.0 * (sign_sum & 1)) * norm_buffer[(hB - hC + hBpC)/2] * sum_buffer[(hC - hB + hBpC)/2] * subs_buffer[(hB + hC - hBpC)/2];
            // std::cout << "h(eta) = " << std::to_string(std::popcount(eta)) << " xi_prod(eta) = " << xi_product[eta] << std::endl;
        }
    }
}

void graphQ(Eigen::MatrixXd &Qfunc, Eigen::Tensor<double,3> &sym_Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int* Adj) {
    const std::complex<double> xi = 0.5 * (sqrt(3)-1) * std::complex<double>(1.0,1.0);
    double denom = 1.0 / qubitstate_size;
    std::vector<std::complex<double>> xi_product(qubitstate_size); // Can be optimized to only store real value, as the imaginary part after summation should be zero
    std::vector<unsigned int> adj_sums(qubitstate_size);
    generate_sum_buffers(xi_product,adj_sums,n_qubits,qubitstate_size,xi,Adj);
    #pragma omp parallel
    {
        std::complex<double> coeff = 0;
        #pragma omp for
        for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
            for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                coeff = 0;
                for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
                    coeff += sign(trace(alpha,eta),trace(adj_sums[eta],beta^eta)) * xi_product[eta];
                }
                Qfunc(alpha,beta) = coeff.real() * denom;
                sym_Qfunc(std::popcount(alpha),std::popcount(beta),std::popcount(alpha^beta)) += Qfunc(alpha,beta);
            }
        }
    }
}

void symonly_graphQ(Eigen::Tensor<double,3> &sym_Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int* Adj) {
    const std::complex<double> xi = 0.5 * (sqrt(3)-1) * std::complex<double>(1.0,1.0);
    double denom = 1.0 / qubitstate_size;
    std::vector<std::complex<double>> xi_product(qubitstate_size); // Can be optimized to only store real value, as the imaginary part after summation should be zero
    std::vector<unsigned int> adj_sums(qubitstate_size);
    generate_sum_buffers(xi_product,adj_sums,n_qubits,qubitstate_size,xi,Adj);
    #pragma omp parallel
    {
        std::complex<double> coeff = 0;
        #pragma omp for
        for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
            for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                coeff = 0;
                for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
                    coeff += sign(trace(alpha,beta),trace(adj_sums[eta],beta^eta)) * xi_product[eta];
                }
                sym_Qfunc(std::popcount(alpha),std::popcount(beta),std::popcount(alpha^beta)) += coeff.real() * denom;
            }
        }
    }
}

// Initializes adjacency matrix, def controls whether all edges are connected or disconnected, with disconnected as default
void init_Adj(unsigned int* Adj, const unsigned int &size, const unsigned int def=0) {
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
void add_edge(unsigned int* Adj, const unsigned int &size, const unsigned int &a, const unsigned int &b) {
    if (size < a+1 || size < b+1) {
        std::cout << "Edges outside bounds" << std::endl;
    } else {
        Adj[a] = Adj[a] ^ (1 << b);
        Adj[b] = Adj[b] ^ (1 << a);
    }
}

// Adds (or removes if already present) edge (a,b) from the adjacency matrix. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
void add_edge(unsigned int* Adj, const unsigned int &size, const std::pair<unsigned int, unsigned int> &edge) {
    if (size < edge.first+1 || size < edge.second+1) {
        std::cout << "Edges outside bounds" << std::endl;
    } else {
        Adj[edge.first] = Adj[edge.first] ^ (1 << edge.second);
        Adj[edge.second] = Adj[edge.second] ^ (1 << edge.first);
    }
}

void add_cyclic_edges(const unsigned int &n_qubits, unsigned int* Adj) {
    for (unsigned int n = 0; n < n_qubits-1; n++) {
        add_edge(Adj,n_qubits,n,n+1);
    }
    add_edge(Adj,n_qubits,n_qubits-1,0);
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

// Calculates the symmetric Q function of a given adjacency matrix and saves it to filename. Prints time taken to perform the calculations
void calc_save_symQ(const unsigned int &n_qubits, unsigned int* Adj, const std::string &filename) {
    const unsigned int qubitstate_size = 1 << n_qubits;
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits,n_qubits,n_qubits);
    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    auto start = std::chrono::high_resolution_clock::now();
    graphQ(Qfunc,sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating took " << duration.count() << "s" << std::endl;
    save_symQfunc(sym_Qfunc,filename);
    save_Qfunc(Qfunc,filename);
}

// Calculates the symmetric Q function of a maximally connected graph state with removed cyclic edges
void generate_acyclic_symQ(const unsigned int &n_qubits) {
    const std::string filename = "ac_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,1);
    add_cyclic_edges(n_qubits,Adj);
    calc_save_symQ(n_qubits,Adj,filename);
}

// Calculates the symmetric Q function of a cyclically connected graph state
void generate_cyclic_symQ(const unsigned int &n_qubits) {
    const std::string filename = "cc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,0);
    add_cyclic_edges(n_qubits,Adj);
    calc_save_symQ(n_qubits,Adj,filename);
}

// Calculates the symmetric Q function of a maximally connected graph state
void generate_maxcon_symQ(const unsigned int &n_qubits) {
    const std::string filename = "mc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,1);
    calc_save_symQ(n_qubits,Adj,filename);
}

void generate_discon_symQ(const unsigned int &n_qubits) {
    const std::string filename = "dc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,0);
    calc_save_symQ(n_qubits,Adj,filename);
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

void sel_calc_state(const unsigned int &n_qubits) {
    bool selected = false;
    while (!selected) {
        std::cout << "Select the graph type [m(aximmally connected),c(yclically connected),a(cyclically connected),d(isconnected)]" << std::endl;
        std::string input;
        std::cin >> input;
        if (input == "mc" || input == "m") {
            generate_maxcon_symQ(n_qubits);
            selected = true;
        } else if (input == "cc" || input == "c") {
            generate_cyclic_symQ(n_qubits);
            selected = true;
        } else if (input == "ac" || input == "a") {
            generate_acyclic_symQ(n_qubits);
            selected = true;
        } else if (input == "dc" || input == "d") {
            generate_discon_symQ(n_qubits);
            selected = true;
        }
    }
}

int main() {
    std::string input;
    std::cout << "Enter the number of qubits" << std::endl;
    std::cin >> input;
    unsigned int n_qubits = parse_unsignedint(input);
    sel_calc_state(n_qubits);
    // std::string filename = "testing.txt";
    // unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    // init_Adj(Adj,n_qubits,0);
    // add_edge(Adj,n_qubits,0,1);
    // calc_save_symQ(n_qubits,Adj,filename);
    return 0;
}