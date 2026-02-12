#include<iostream>
#include<fstream>
#include<complex>
#include<vector>
#include<algorithm>
#include<utility> // std::pair
#include<tuple>
#include<bitset> // Print numbers as binary easily
#include<filesystem> // Current directory
#include<chrono> // Timing
#include<Eigen/Dense>
#include<unsupported/Eigen/CXX11/Tensor>
#include<sstream>
#include<memory>
#include "omp.h"
#include "displaced_Qfunc.h"
#include "graph_generator.h"
#include "sym_space.h"
#include "Qfunc.h"
#include "graph.h"
#include "GF2N.h"

#define SQRT3 1.73205080756

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
                sign_sum += ((eta >> n) & 1) * std::popcount( Adj[n] & (eta & ((1 << n) - 1)) );
            }
            adj_sums[eta] = adj_sum;
            hB = std::popcount(adj_sum);
            hC = std::popcount(eta);
            hBpC = std::popcount(adj_sum ^ eta);
            xi_product[eta] = (1.0 - 2.0 * (sign_sum & 1)) * norm_buffer[(hB - hC + hBpC)/2] * sum_buffer[(hC - hB + hBpC)/2] * subs_buffer[(hB + hC - hBpC)/2];
        }
    }
}

void graphQ(Eigen::MatrixXd &Qfunc, Eigen::Tensor<double,3> &sym_Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int* Adj) {
    static const std::complex<double> xi = 0.5 * (sqrt(3)-1) * std::complex<double>(1.0,1.0);
    static double denom = 1.0 / qubitstate_size;
    std::vector<std::complex<double>> xi_product(qubitstate_size); // Can be optimized to only store real value, as the imaginary part after summation should be zero
    std::vector<unsigned int> adj_sums(qubitstate_size);
    generate_sum_buffers(xi_product,adj_sums,n_qubits,qubitstate_size,xi,Adj);
    #pragma omp parallel
    {
        std::complex<double> coeff = 0;
        #pragma omp for collapse(2)
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
        #pragma omp for collapse(2)
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

// Generates a buffer of all powers of 1/sqrt(3) from 0 to n_qubits
std::unique_ptr<double[]> generate_sqrt3_buffer(unsigned int const &n_qubits) {
    std::unique_ptr<double[]> sqrt3_buffer = std::make_unique<double[]>(n_qubits + 1);
    double sqrt3_power = 1;
    for (int j = 0; j <= n_qubits; j++) {
        sqrt3_buffer[j] = sqrt3_power;
        sqrt3_power /= SQRT3;
    }
    return std::move(sqrt3_buffer);
}

// Returns the symmetric Q function of Adj from its characteristic function
Eigen::Tensor<double, 3> opt_graph_only_symQ(unsigned int const &n_qubits, unsigned int const &qubitstate_size, unsigned int* const Adj) {
    static std::vector<polynomial3> gmnk = get_all_gmnk(n_qubits);
    std::cout << "Calculating graph characteristic" << std::endl;
    Eigen::Tensor<int, 3> C_A = graph_characteristic(n_qubits, qubitstate_size, Adj);
    static std::unique_ptr<double[]> sqrt3_buffer = generate_sqrt3_buffer(n_qubits);
    double norm = 1.0 / (1 << n_qubits);
    polynomial3 pol_symQ(n_qubits, n_qubits, n_qubits);
    std::cout << "Summing polynomials" << std::endl;
    // p is the last index so that it runs faster, for cache efficiency
    sym_space_loop(n_qubits, [&](int const &r, int const &q, int const &p) {
        pol_symQ += gmnk[p + (q + r * (n_qubits + 1)) * (n_qubits + 1)].mult(norm * sqrt3_buffer[(p+q+r)/2] * C_A(p, q, r));
    });
    std::cout << "Finished summing polynomials" << std::endl;
    return pol_symQ.as_binom_tensor(n_qubits);
}

// Calculates the symmetric Q function of a given adjacency matrix and saves it to filename. Prints time taken to perform the calculations
void calc_save_symQ(const unsigned int &n_qubits, unsigned int* Adj, const std::string &filename) {
    const unsigned int qubitstate_size = 1 << n_qubits;
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    auto start = std::chrono::high_resolution_clock::now();
    symonly_graphQ(sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating took " << duration.count() << "s" << std::endl;
    save_symQfunc(sym_Qfunc,filename);
}

// Calculates the Q function of a given adjacency matrix and saves it to filename. Prints time taken to perform the calculations
void calc_save_graph_symQ(const unsigned int &n_qubits, unsigned int* Adj, const std::string &filename) {
    const unsigned int qubitstate_size = 1 << n_qubits;
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    auto start = std::chrono::high_resolution_clock::now();
    graphQ(Qfunc, sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating took " << duration.count() << "s" << std::endl;
    save_symQfunc(sym_Qfunc,filename);
    save_Qfunc(Qfunc,filename);
}

// Calculates the symmetric Q function of a maximally connected graph state with removed cyclic edges
void generate_acyclic_symQ(const unsigned int &n_qubits) {
    const std::string filename = "ac_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,1);
    add_cyclic_edges(n_qubits,Adj);
    calc_save_symQ(n_qubits,Adj,filename);
}

// Calculates the symmetric Q function of a cyclically connected graph state
void generate_cyclic_symQ(const unsigned int &n_qubits) {
    const std::string filename = "cc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,0);
    add_cyclic_edges(n_qubits,Adj);
    calc_save_symQ(n_qubits,Adj,filename);
}

// Calculates the symmetric Q function of a maximally connected graph state
void generate_maxcon_symQ(const unsigned int &n_qubits) {
    const std::string filename = "mc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,1);
    calc_save_symQ(n_qubits,Adj,filename);
}

void generate_discon_symQ(const unsigned int &n_qubits) {
    const std::string filename = "dc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,0);
    calc_save_symQ(n_qubits,Adj,filename);
}

// Calculates the Q function of a maximally connected graph state with removed cyclic edges
void generate_acyclic_graphQ(const unsigned int &n_qubits) {
    const std::string filename = "ac_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,1);
    add_cyclic_edges(n_qubits,Adj);
    calc_save_graph_symQ(n_qubits,Adj,filename);
}

// Calculates the Q function of a cyclically connected graph state
void generate_cyclic_graphQ(const unsigned int &n_qubits) {
    const std::string filename = "cc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,0);
    add_cyclic_edges(n_qubits,Adj);
    calc_save_graph_symQ(n_qubits,Adj,filename);
}

// Calculates the Q function of a maximally connected graph state
void generate_maxcon_graphQ(const unsigned int &n_qubits) {
    const std::string filename = "mc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,1);
    calc_save_graph_symQ(n_qubits,Adj,filename);
}

void generate_discon_graphQ(const unsigned int &n_qubits) {
    const std::string filename = "dc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,0);
    calc_save_graph_symQ(n_qubits,Adj,filename);
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

void sel_calc_state_symQ(const unsigned int &n_qubits) {
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

void sel_calc_state_graph_symQ(const unsigned int &n_qubits) {
    bool selected = false;
    while (!selected) {
        std::cout << "Select the graph type [m(aximmally connected),c(yclically connected),a(cyclically connected),d(isconnected)]" << std::endl;
        std::string input;
        std::cin >> input;
        if (input == "mc" || input == "m") {
            generate_maxcon_graphQ(n_qubits);
            selected = true;
        } else if (input == "cc" || input == "c") {
            generate_cyclic_graphQ(n_qubits);
            selected = true;
        } else if (input == "ac" || input == "a") {
            generate_acyclic_graphQ(n_qubits);
            selected = true;
        } else if (input == "dc" || input == "d") {
            generate_discon_graphQ(n_qubits);
            selected = true;
        }
    }
}

static void get_unsignedint(unsigned int &parsed_input) {
    std::string input = "";
    std::cin >> input;
    try {
        unsigned long u = std::stoul(input);
        if (u > std::numeric_limits<unsigned int>::max())
            throw std::out_of_range(input);

        parsed_input = u;
    } catch (const std::invalid_argument& e) {
        std::cout << "Input could not be parsed: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Input out of range: " << e.what() << std::endl;
    }
}

void calc_manual_graph() {
    unsigned int n_qubits = 0;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    unsigned int* Adj = static_cast<unsigned int*>(_malloca(n_qubits * n_qubits * sizeof(unsigned int)));
    std::string filename = "";
    parse_manual_graph(n_qubits, Adj, filename);
    calc_save_graph_symQ(n_qubits, Adj, filename);
}

void calc_gen_graph_symQ() {
    unsigned int N = 0;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(N);
    sel_calc_state_symQ(N);
}

void calc_gen_graph_graph_symQ() {
    unsigned int N = 0;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(N);
    sel_calc_state_graph_symQ(N);
}

void graphQ_from_file(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &graph_num, unsigned int* Adj, Eigen::MatrixXd &Qfunc) {
    parse_graph_from_edge_list(n_qubits, graph_num, Adj);
    
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    auto start = std::chrono::high_resolution_clock::now();
    graphQ(Qfunc, sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating Q took " << duration.count() << "s" << std::endl;
}

// Calculates the full displaced entropies of a particular graph state, specified by the number of qubits and graph_num
void calc_full_displaced_graph_entropy(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &graph_num) {
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    std::string filename = "entropies/" + std::to_string(n_qubits) + "q_" + std::to_string(graph_num) + ".txt"; // Use folder inside Qfuncs as this is secondary
    
    graphQ_from_file(n_qubits, qubitstate_size, graph_num, Adj, Qfunc);

    Eigen::MatrixXd entropies(qubitstate_size, qubitstate_size);
    std::tuple<unsigned int, unsigned int> max_displacement = {0, 0};
    std::tuple<unsigned int, unsigned int> min_displacement = {0, 0};
    auto start = std::chrono::high_resolution_clock::now();
    calc_full_displaced_maxmin_entropy(Qfunc, n_qubits, qubitstate_size, entropies, max_displacement, min_displacement);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    std::cout << "Calculating displacements took " << duration.count() << "s" << std::endl;
    save_Qfunc(entropies, filename);
}

void calc_all_displaced_graph_symQ(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &graph_num) {
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    std::string filepath = std::to_string(n_qubits) + "q_" + std::to_string(graph_num);

    graphQ_from_file(n_qubits, qubitstate_size, graph_num, Adj, Qfunc);

    calc_all_displaced_symQ(Qfunc, n_qubits, qubitstate_size, filepath);
}

void manual_minmax_displaced_graph_distance() {
    unsigned int n_qubits;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    const unsigned int qubitstate_size = 1 << n_qubits;
    unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));;
    std::string filename = "";
    generate_selected_graph(n_qubits, Adj, filename);

    double max_distance = 0;
    double min_distance = 1.0e10;
    std::tuple<unsigned int, unsigned int> max_displacement = {0, 0};
    std::tuple<unsigned int, unsigned int> min_displacement = {0, 0};

    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    
    auto start = std::chrono::high_resolution_clock::now();
    graphQ(Qfunc, sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating Q took " << duration.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    max_displaced_distances(n_qubits, qubitstate_size, Qfunc, sym_Qfunc, min_distance, max_distance, min_displacement, max_displacement);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Calculating displacements took " << duration.count() << "s" << std::endl;
    std::cout << "Max distance: " << max_distance << std::endl;
    std::cout << "Min distance: " << min_distance << std::endl;
    std::cout << "Max displacement: " << std::get<0>(max_displacement) << ", " << std::get<1>(max_displacement) << std::endl;
    std::cout << "Min displacement: " << std::get<0>(min_displacement) << ", " << std::get<1>(min_displacement) << std::endl;
}

void manual_minmax_lClifford_graph_distance() {
    unsigned int n_qubits;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    const unsigned int qubitstate_size = 1 << n_qubits;
    unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));;
    std::string filename = "";
    generate_selected_graph(n_qubits, Adj, filename);

    double max_distance = 0;
    double min_distance = 1.0e10;
    std::tuple<unsigned int, unsigned int, unsigned int> max_displacement = {0, 0, 0};
    std::tuple<unsigned int, unsigned int, unsigned int> min_displacement = {0, 0, 0};

    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    
    auto start = std::chrono::high_resolution_clock::now();
    graphQ(Qfunc, sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating Q took " << duration.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    max_lClifford_distances(n_qubits, qubitstate_size, Qfunc, sym_Qfunc, min_distance, max_distance, min_displacement, max_displacement);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Calculating displacements took " << duration.count() << "s" << std::endl;
    std::cout << "Max distance: " << max_distance << std::endl;
    std::cout << "Min distance: " << min_distance << std::endl;
    std::cout << "Max displacement: " << std::get<0>(max_displacement) << ", " << std::get<1>(max_displacement) << std::endl;
    std::cout << "Min displacement: " << std::get<0>(min_displacement) << ", " << std::get<1>(min_displacement) << std::endl;
}

// Loops over all graphs with n_qubits, calculating both their Q and symmetrized Q functions and executes a particular function acting on them and the graph number
template<typename LoopFunc> 
void for_all_graphs_Qfuncs(const unsigned int &n_qubits, LoopFunc operate_graph) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string graphs_suffix = std::to_string(n_qubits) + ".txt";
    std::ifstream input_file(cwd.string()+"/data/graphs/"+graphs_suffix,std::ifstream::in);
    std::string line;
    unsigned int graph_num = 1;
    
    unsigned int qubitstate_size = 1 << n_qubits;
    unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    Eigen::MatrixXd graph_Qfunc(qubitstate_size, qubitstate_size);
    Eigen::Tensor<double, 3> graph_symQ(n_qubits + 1, n_qubits + 1, n_qubits + 1);

    if (input_file.is_open()) {
        while (std::getline(input_file, line)) {
            parse_graph_line(n_qubits, line, Adj);
            graphQ(graph_Qfunc, graph_symQ.setZero(), n_qubits, qubitstate_size, Adj);
            
            operate_graph(graph_Qfunc, graph_symQ, graph_num);

            graph_num++;
        }
    } else {
        std::cout << "Could not parse graphs" << std::endl;
    }
}


void max_all_displaced_graphs_distances() {
    unsigned int n_qubits;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    const unsigned int qubitstate_size = 1 << n_qubits;

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string save_folder = cwd.string()+"/data/disp_dist/";
    std::string max_G_distances_filename = "max_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_R_distances_filename = "max_R_q" + std::to_string(n_qubits) + ".txt";
    std::string max_disp_G_filename = "max_disp_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_disp_R_filename = "max_disp_R_q" + std::to_string(n_qubits) + ".txt";
    std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_disp_G_file(save_folder + max_disp_G_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_disp_R_file(save_folder + max_disp_R_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    double max_G_distance, max_R_distance;
    std::tuple<unsigned int, unsigned int> max_G_parameters, max_R_parameters;
    
    if (max_G_distances_file.is_open() && max_R_distances_file.is_open() && max_disp_G_file.is_open() && max_disp_R_file.is_open()) {
        for_all_graphs_Qfuncs(
            n_qubits,
            [&](const Eigen::MatrixXd &graph_Qfunc, const Eigen::Tensor<double, 3> &graph_symQ, const unsigned int &graph_num) {
                max_displaced_distances(n_qubits, qubitstate_size, graph_Qfunc, graph_symQ, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
                max_G_distances_file << max_G_distance << "\n";
                max_R_distances_file << max_R_distance << "\n";
                max_disp_G_file << std::get<0>(max_G_parameters) << ", " << std::get<1>(max_G_parameters) << "\n";
                max_disp_R_file << std::get<0>(max_R_parameters) << ", " << std::get<1>(max_R_parameters) << "\n";
                std::cout << graph_num << std::endl;
            }
        );
        max_G_distances_file.close();
        max_R_distances_file.close();
        max_disp_G_file.close();
        max_disp_R_file.close();

    } else {
        std::cout << "Could not save minmax results" << std::endl;
    }
}

void max_all_lClifford_graphs_distances() {
    unsigned int n_qubits;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    const unsigned int qubitstate_size = 1 << n_qubits;

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string save_folder = cwd.string()+"/data/clif_dist/";
    std::string max_G_distances_filename = "max_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_R_distances_filename = "max_R_q" + std::to_string(n_qubits) + ".txt";
    std::string max_lClifford_G_filename = "max_cliff_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_lClifford_R_filename = "max_cliff_R_q" + std::to_string(n_qubits) + ".txt";
    std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_lClifford_G_file(save_folder + max_lClifford_G_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_lClifford_R_file(save_folder + max_lClifford_R_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    double max_G_distance, max_R_distance;
    std::tuple<unsigned int, unsigned int, unsigned int> max_G_parameters, max_R_parameters;
    
    if (max_G_distances_file.is_open() && max_R_distances_file.is_open() && max_lClifford_G_file.is_open() && max_lClifford_R_file.is_open()) {
        for_all_graphs_Qfuncs(
            n_qubits,
            [&](const Eigen::MatrixXd &graph_Qfunc, const Eigen::Tensor<double, 3> &graph_symQ, const unsigned int &graph_num) {
                max_lClifford_distances(n_qubits, qubitstate_size, graph_Qfunc, graph_symQ, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
                max_G_distances_file << max_G_distance << "\n";
                max_R_distances_file << max_R_distance << "\n";
                max_lClifford_G_file << std::get<0>(max_G_parameters) << ", " << std::get<1>(max_G_parameters) << ", " << std::get<2>(max_G_parameters) << "\n";
                max_lClifford_R_file << std::get<0>(max_R_parameters) << ", " << std::get<1>(max_R_parameters) << ", " << std::get<2>(max_R_parameters) << "\n";
                std::cout << graph_num << std::endl;
            }
        );
        max_G_distances_file.close();
        max_R_distances_file.close();
        max_lClifford_G_file.close();
        max_lClifford_R_file.close();

    } else {
        std::cout << "Could not save minmax results" << std::endl;
    }
}

std::vector<unsigned int> ask_integers(const std::string &prompt) {
    std::vector<unsigned int> numbers;
    std::string line;

    std::cout << prompt << std::endl;
    std::cout << "Leave empty or send f to exit" << std::endl;

    if (std::cin.peek() == '\n') {
        std::cin.ignore();
    }

    while (true) {
        std::getline(std::cin, line);

        if (line.empty() || line == "f")
            break;

        std::istringstream iss(line);
        unsigned int num;
        if (iss >> num) {
            numbers.push_back(num);
        } else {
            std::cout << "Invalid input. Please enter an integer, 'f', or an empty line to finish.\n";
        }
    }

    return numbers;
}

void max_random_displaced_graphs_distances() {
    unsigned int n_qubits, n_graphs;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    const unsigned int qubitstate_size = 1 << n_qubits;
    std::cout << "Enter the number of distintict graphs to be generated" << std::endl;
    get_unsignedint(n_graphs);

    std::vector<unsigned int> seeds = ask_integers("Enter random engine seeds");
    std::vector<Edge_list> graphs(n_graphs);
    set_engine_seed(seeds);
    generate_random_edge_connected_graph_set(n_qubits, n_graphs, graphs);

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string save_folder = cwd.string()+"/data/disp_dist/";
    std::string max_G_distances_filename = "rand_max_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_R_distances_filename = "rand_max_R_q" + std::to_string(n_qubits) + ".txt";
    std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    double max_G_distance, max_R_distance;
    std::tuple<unsigned int, unsigned int> max_G_parameters, max_R_parameters;
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
    Eigen::MatrixXd graph_Qfunc(qubitstate_size,qubitstate_size);
    Eigen::Tensor<double,3> graph_symQ(n_qubits+1,n_qubits+1,n_qubits+1);
    
    unsigned int count = 1;
    if (max_G_distances_file.is_open() && max_R_distances_file.is_open()) {
        for (Edge_list edge_list : graphs) {
            init_Adj(Adj, n_qubits, 0);
            add_edge_list(n_qubits, edge_list, Adj);
            graphQ(graph_Qfunc, graph_symQ.setZero(), n_qubits, qubitstate_size, Adj);
            max_displaced_distances(n_qubits, qubitstate_size, graph_Qfunc, graph_symQ, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
            max_G_distances_file << max_G_distance << "\n";
            max_R_distances_file << max_R_distance << "\n";
            std::cout << count << std::endl;
            count++;
        }
        max_G_distances_file.close();
        max_R_distances_file.close();

    } else {
        std::cout << "Could not save minmax results" << std::endl;
    }
}

void field_sym_sums_comparison() {
    unsigned int n_qubits, n_graphs;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    const unsigned int qubitstate_size = 1 << n_qubits;

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string save_folder = cwd.string()+"/data/test/";
    std::string sums_difference_filename = "diff_" + std::to_string(n_qubits) + ".txt";
    std::ofstream sums_difference_file(save_folder + sums_difference_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    Eigen::Tensor<double, 3> Rmnk = get_Rmnk(n_qubits);
    // As it will only divide, but it causes problems wherever mnk is not a valid pair, hence Rmnk = 0, set those values to one. This won't matter, as the Q function is null there
    std::function<double(double)> remove_negatives = [](double x) {
        return (std::abs(x) < 1e-10) ? 0.0 : x;
    };
    std::function<double(double)> zeros_to_one = [](double x) {
    return (x == 0) ? 1 : x;
    };
    Rmnk = Rmnk.unaryExpr(zeros_to_one);

    if (sums_difference_file.is_open()) {
        if (n_qubits < 9) {
            for_all_graphs_Qfuncs(
                n_qubits, 
                [&] (Eigen::MatrixXd &graphQ, Eigen::Tensor<double, 3> &graph_symQ, const unsigned int &graph_num) {
                    Eigen::Tensor<double, 3> graphG = get_Gfunc(n_qubits, qubitstate_size, graph_symQ);
                    // GraphQ might be -1e^-17. Could this cause problems anywhere else?
                    graphG = graphG.unaryExpr(remove_negatives);
                    graphQ = graphQ.unaryExpr(remove_negatives);
                    graph_symQ = graph_symQ.unaryExpr(remove_negatives);
                    Eigen::Tensor<double, 3> graphG_tilde = graphG / Rmnk;
                    
                    double field_sum = 0;
                    double sym_sum = 0;
                    for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
                        for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                            field_sum += std::sqrt(graphG_tilde(std::popcount(alpha), std::popcount(beta), std::popcount(alpha ^ beta)) * graphQ(alpha, beta));
                        }
                    }
                    Eigen::Tensor<double, 0> sym_sum_result = (graphG * graph_symQ).sqrt().sum();
                    sym_sum = sym_sum_result(0);
                    
                    sums_difference_file << (field_sum) / sym_sum << "\n";
                }
            );
        } else {
            std::cout << "Enter the number of distintict graphs to be generated" << std::endl;
            get_unsignedint(n_graphs);

            std::vector<unsigned int> seeds = ask_integers("Enter random engine seeds");
            std::vector<Edge_list> graphs(n_graphs);
            set_engine_seed(seeds);
            generate_random_edge_connected_graph_set(n_qubits, n_graphs, graphs);

            unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
            Eigen::MatrixXd graph_Qfunc(qubitstate_size,qubitstate_size);
            Eigen::Tensor<double,3> graph_symQ(n_qubits+1,n_qubits+1,n_qubits+1);
            
            unsigned int count = 1;
            for (Edge_list edge_list : graphs) {
                init_Adj(Adj, n_qubits, 0);
                add_edge_list(n_qubits, edge_list, Adj);
                graphQ(graph_Qfunc, graph_symQ.setZero(), n_qubits, qubitstate_size, Adj);
                Eigen::Tensor<double, 3> graphG = get_Gfunc(n_qubits, qubitstate_size, graph_symQ);
                graphG = graphG.unaryExpr(remove_negatives);
                graph_Qfunc = graph_Qfunc.unaryExpr(remove_negatives);
                graph_symQ = graph_symQ.unaryExpr(remove_negatives);
                Eigen::Tensor<double, 3> graphG_tilde = graphG / Rmnk;
                
                double field_sum = 0;
                double sym_sum = 0;
                unsigned int halpha, hbeta, halphabeta;
                for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
                    halpha = std::popcount(alpha);
                    for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                        // GraphQ might be -e^-17. Could this cause problems anywhere else?
                        if (graph_Qfunc(alpha, beta) > 0) {
                            hbeta = std::popcount(beta);
                            halphabeta = std::popcount(alpha ^ beta);
                            field_sum += std::sqrt( graphG_tilde(halpha, hbeta, halphabeta) * graph_Qfunc(alpha, beta));
                        }
                    }
                }

                Eigen::Tensor<double, 0> sym_sum_result = (graphG * graph_symQ).sqrt().sum();
                sym_sum = sym_sum_result(0);
                
                sums_difference_file << (field_sum) / sym_sum << "\n";
            }
        }
    } else {
        std::cout << "Could not save comparison results" << std::endl;
    }


}

inline unsigned int get_jth_bit(const unsigned int &number, const unsigned int &j) {
    return (number >> j) & 1;
}

// Returns the expected value of Sx^r for a graph state given by its adjacency matrix Adj
int exp_val_Sx_r(const unsigned int &n_qubits, const unsigned int &r, unsigned int const* const Adj) {
    int exp_val = 0;
    unsigned int adj_sums = 0, eta = 0, pairs = 0;
    nested_basis_loop(n_qubits, r, 
    [&] (unsigned int const* const j_vector) {
        adj_sums = eta = pairs = 0;
        for (int l = 0; l < r; l++) {
            adj_sums ^= Adj[j_vector[l] - 1]; // Sum of the j_vector rows of Adj (j starts at 1, but the first element in Adj is 0)
            eta ^= 1 << (j_vector[l] - 1); // Sum of the j_vector basis vectors (j starts at 1, but the first basis vector should be 0x1)
        }
        if (adj_sums == 0) {
            pairs = 0;
            for (int m = 1; m <= n_qubits; m++) {
                for (int k = 1; k <= m; k++) {
                    pairs += get_jth_bit(Adj[m], k) * get_jth_bit(eta, m) * get_jth_bit(eta, k);
                }
            }
            exp_val += 1 - 2 * (pairs % 2);
        }
    }
    );
    return exp_val;
}

// Loops over all graphs with n_qubits, calculating their adjacency matrix and executes a particular function acting on it and the graph number
template<typename LoopFunc> 
void for_all_graphs_Adj(const unsigned int &n_qubits, LoopFunc operate_graph) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string graphs_suffix = std::to_string(n_qubits) + ".txt";
    std::ifstream input_file(cwd.string()+"/data/graphs/"+graphs_suffix,std::ifstream::in);
    std::string line;
    unsigned int graph_num = 1;
    
    unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));

    if (input_file.is_open()) {
        while (std::getline(input_file, line)) {
            parse_graph_line(n_qubits, line, Adj);
            
            operate_graph(Adj, graph_num);

            graph_num++;
        }
    } else {
        std::cout << "Could not parse graphs" << std::endl;
    }
}

// Calculates the exact expected values of Sx^r for r = 1, ..., 4 for all graph states with a prompted number of qubits and saves them in the same order as the database
void classify_graphs() {
    unsigned int n_qubits;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream save_file(cwd.string()+"/data/graphs/Sx_exp_q"+std::to_string(n_qubits)+".txt",std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    if (save_file.is_open()) {
        for_all_graphs_Adj(n_qubits, 
        [&] (unsigned int* const Adj, const unsigned int &graph_num) {
            for (int j = 1; j <= 4; j++) {
                save_file << exp_val_Sx_r(n_qubits, j, Adj) << ", ";
            }
            save_file << "\n";
            std::cout << graph_num << "\n";
        }
        );
    } else {
        std::cout << "Could not save expected values" << std::endl;
    }
}

void check_opt_symQ() {
    std::function<double(double)> remove_negatives = [](double x) {
        return (std::abs(x) < 1e-10) ? 0.0 : x;
    };
    unsigned int n_qubits;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    const unsigned int qubitstate_size = 1 << n_qubits;
    double norm = 1.0 / qubitstate_size;
    Eigen::Tensor<double, 3> opt_symQ(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    Eigen::Tensor<double, 3> full_symQ(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    for_all_graphs_Adj(n_qubits, [&] (unsigned int* const Adj, const unsigned int &graph_num) {
        full_symQ.setZero();
        opt_symQ.setZero();
        std::cout << graph_num << std::endl;
        std::cout << "Calculating opt symQ" << std::endl;
        opt_symQ = opt_graph_only_symQ(n_qubits, qubitstate_size, Adj);
        // opt_symQ = opt_symQ.unaryExpr(remove_negatives);
        std::cout << "Calculating symQ" << std::endl;
        symonly_graphQ(full_symQ, n_qubits, qubitstate_size, Adj);
        // full_symQ = full_symQ.unaryExpr(remove_negatives);
        Eigen::Tensor<double, 0> b = opt_symQ.sum();
        Eigen::Tensor<double, 0> c = full_symQ.sum();
        std::cout << " Difference is " << b(0) << "\n";
        std::cout << " Difference is " << c(0) << "\n";
    });
}

int main() {
    check_opt_symQ();
    

    return 0;
}