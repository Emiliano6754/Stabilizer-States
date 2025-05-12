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
#include "displaced_Qfunc.h"

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

// Initializes adjacency matrix, def controls whether all edges are connected or disconnected, with disconnected as default. def should only be 0 or 1, undefined behavior otherwise
void init_Adj(unsigned int* Adj, const unsigned int &size, const unsigned int def=0) {
    if (size > 8*sizeof(unsigned int)) {
        std::cout << "Too many qubits, change unsigned int in adjacency matrices to use more" << std::endl;
    } else {
        unsigned int connection = def * ((1 << size) - 1);
        for (unsigned int i = 0; i < size; i++) {
            Adj[i] = connection ^ (def << i);
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

// Prints the Qfunc to console for debugging purposes
void print_Qfunc(const Eigen::MatrixXd &Qfunc) {
    const unsigned int n_qubits = 4;
    const unsigned int qubitstate_size = 1<<n_qubits;
    for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
            for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                std::cout << "Q(" << alpha << "," << beta << ") = " <<Qfunc(alpha,beta) << std::endl;
            }
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

// Calculates the Q function of a maximally connected graph state with removed cyclic edges
void generate_acyclic_graphQ(const unsigned int &n_qubits) {
    const std::string filename = "ac_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,1);
    add_cyclic_edges(n_qubits,Adj);
    calc_save_graph_symQ(n_qubits,Adj,filename);
}

// Calculates the Q function of a cyclically connected graph state
void generate_cyclic_graphQ(const unsigned int &n_qubits) {
    const std::string filename = "cc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,0);
    add_cyclic_edges(n_qubits,Adj);
    calc_save_graph_symQ(n_qubits,Adj,filename);
}

// Calculates the Q function of a maximally connected graph state
void generate_maxcon_graphQ(const unsigned int &n_qubits) {
    const std::string filename = "mc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    init_Adj(Adj,n_qubits,1);
    calc_save_graph_symQ(n_qubits,Adj,filename);
}

void generate_discon_graphQ(const unsigned int &n_qubits) {
    const std::string filename = "dc_q" + std::to_string(n_qubits)+".txt";
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
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

void get_unsignedint(unsigned int &parsed_input) {
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

void parse_manual_graph(const unsigned int &n_qubits, unsigned int* Adj, std::string &filename) {
    init_Adj(Adj, n_qubits, 0);
    unsigned int q1, q2;
    bool finished = false;
    std::string input;
    while (!finished) {
        std::cout << "Enter a qubit connection" << std::endl;
        get_unsignedint(q1);
        get_unsignedint(q2);
        if (q1 < n_qubits || q2 < n_qubits) {
            add_edge(Adj, n_qubits, q1, q2);
            std::cout << "Enter n to exit" << std::endl;
            std::getline(std::cin, input);
            if (std::cin.peek() != '\n') {
                std::cin >> input;
                if (input == "n") {
                    break;
                }
            }
            
        } else {
            std::cout << "Connection outside bounds" << std::endl;
        }
    }
    std::cout << "Enter the file prefix" << std::endl;
    input = "";
    std::cin >> input;
    filename = input+"_q" + std::to_string(n_qubits)+".txt";
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

// Parses the graph_num graph (as numerated in the graph list) with n unlabeled nodes as the adjacency matrix in Adj from the edge list'. Assumes the file is named as n_qubits.txt
void parse_graph_from_edge_list(const unsigned int &n_qubits, const unsigned int &graph_num, unsigned int* Adj) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string filename = std::to_string(n_qubits) + ".txt";
    std::ifstream input_file(cwd.string()+"/data/graphs/"+filename,std::ifstream::in);
    std::string line, first, second;
    // unsigned int start = 0;
    unsigned int pos = 1;
    unsigned int next_sc = 0;
    unsigned int comma_pos = 0;
    unsigned int last_sc = 0;
    bool finished = false;

    if (input_file.is_open()) {
        while (std::getline(input_file, line)) {
            if (pos == graph_num) {
                last_sc = line.find(':');
                while(!finished) {
                    next_sc = line.find(';', last_sc+1);
                    comma_pos = line.find(',', last_sc);
                    if (next_sc == std::string::npos || next_sc >= line.size()) {
                        finished = true;
                        next_sc = line.back();
                    }
                    first = line.substr(last_sc+1, comma_pos-last_sc-1);
                    second = line.substr(comma_pos+1, next_sc-comma_pos-1);
                    add_edge(Adj, n_qubits, std::stoul(first) - 1, std::stoul(second) - 1);
                    
                    last_sc = next_sc;
                }
                break;
            }
            pos++;
        }
    } else {
        std::cout << "Could not parse graph" << std::endl;
    }
}

void graphQ_from_file(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &graph_num, unsigned int* Adj, Eigen::MatrixXd &Qfunc) {
    parse_graph_from_edge_list(n_qubits, graph_num, Adj);
    
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    auto start = std::chrono::high_resolution_clock::now();
    graphQ(Qfunc,sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating Q took " << duration.count() << "s" << std::endl;
}

// Calculates the full displaced entropies of a particular graph state, specified by the number of qubits and graph_num
void calc_full_displaced_graph_entropy(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &graph_num) {
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
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

void ask_manual_graph_params(unsigned int &n_qubits, unsigned int &graph_num) {
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    std::cout << "Enter the graph number" << std::endl;
    get_unsignedint(graph_num);
}

void calc_all_displaced_graph_symQ(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &graph_num) {
    unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));
    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    std::string filepath = std::to_string(n_qubits) + "q_" + std::to_string(graph_num);

    graphQ_from_file(n_qubits, qubitstate_size, graph_num, Adj, Qfunc);

    calc_all_displaced_symQ(Qfunc, n_qubits, qubitstate_size, filepath);
}

// Prompts for a type of graph, the number of qubits and returns its adjacency matrix and filename
void generate_selected_graph(unsigned int &n_qubits, unsigned int *Adj, std::string &filename) {
    bool selected = false;
    while (!selected) {
        std::cout << "Select the graph type [m(aximmally connected),c(yclically connected),a(cyclically connected),d(isconnected),g(raph num),e(dge list)]" << std::endl;
        std::string input;
        std::cin >> input;
        if (input == "mc" || input == "m") {
            init_Adj(Adj, n_qubits, 1);
            filename = "mc_q" + std::to_string(n_qubits)+".txt";
            selected = true;
        } else if (input == "cc" || input == "c") {
            init_Adj(Adj, n_qubits, 0);
            add_cyclic_edges(n_qubits, Adj);
            filename = "cc_q" + std::to_string(n_qubits)+".txt";
            selected = true;
        } else if (input == "ac" || input == "a") {
            init_Adj(Adj, n_qubits, 1);
            add_cyclic_edges(n_qubits, Adj);
            filename = "ac_q" + std::to_string(n_qubits)+".txt";
            selected = true;
        } else if (input == "dc" || input == "d") {
            init_Adj(Adj, n_qubits, 0);
            filename = "dc_q" + std::to_string(n_qubits)+".txt";
            selected = true;
        } else if (input == "gn" || input == "g") {
            unsigned int graph_num = 0;
            std::cout << "Enter the graph number" << std::endl;
            get_unsignedint(graph_num);
            filename = "q" + std::to_string(n_qubits) + "_" + std::to_string(graph_num) + ".txt";
            parse_graph_from_edge_list(n_qubits, graph_num, Adj);
            selected = true;
        } else if (input == "ed" || input == "e") {
            parse_manual_graph(n_qubits, Adj, filename);
            selected = true;
        }
    }
}

void manual_minmax_displaced_graph_distance() {
    unsigned int n_qubits;
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    const unsigned int qubitstate_size = 1 << n_qubits;
    unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));;
    std::string filename = "";
    generate_selected_graph(n_qubits, Adj, filename);

    double max_distance = 0;
    double min_distance = 1.0e10;
    std::tuple<unsigned int, unsigned int> max_displacement = {0, 0};
    std::tuple<unsigned int, unsigned int> min_displacement = {0, 0};

    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    
    auto start = std::chrono::high_resolution_clock::now();
    graphQ(Qfunc,sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
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
    unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));;
    std::string filename = "";
    generate_selected_graph(n_qubits, Adj, filename);

    double max_distance = 0;
    double min_distance = 1.0e10;
    std::tuple<unsigned int, unsigned int, unsigned int> max_displacement = {0, 0, 0};
    std::tuple<unsigned int, unsigned int, unsigned int> min_displacement = {0, 0, 0};

    Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
    Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    
    auto start = std::chrono::high_resolution_clock::now();
    graphQ(Qfunc,sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
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

void parse_graph_line(const unsigned int &n_qubits, std::string &line, unsigned int* Adj) {
    std::string first, second;
    bool finished = false;
    unsigned int next_sc = 0;
    unsigned int comma_pos = 0;
    unsigned int last_sc = 0;

    init_Adj(Adj, n_qubits, 0);
    last_sc = line.find(':');
    while(!finished) {
        next_sc = line.find(';', last_sc+1);
        comma_pos = line.find(',', last_sc);
        if (next_sc == std::string::npos || next_sc >= line.size()) {
            finished = true;
            next_sc = line.back();
        }
        first = line.substr(last_sc+1, comma_pos-last_sc-1);
        second = line.substr(comma_pos+1, next_sc-comma_pos-1);
        add_edge(Adj, n_qubits, std::stoul(first) - 1, std::stoul(second) - 1);
        
        last_sc = next_sc;
    }
}

// Loops over all graphs with n_qubits, calculating both their Q and symmetrized Q functions and executes a particular function acting on them and the graph number
template<typename LoopFunc> 
void for_all_graphs(const unsigned int &n_qubits, LoopFunc operate_graph) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string graphs_suffix = std::to_string(n_qubits) + ".txt";
    std::ifstream input_file(cwd.string()+"/data/graphs/"+graphs_suffix,std::ifstream::in);
    std::string line;
    unsigned int graph_num = 1;
    
    unsigned int qubitstate_size = 1 << n_qubits;
    unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * n_qubits * sizeof(unsigned int)));;
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

    double max_G_distance;
    double max_R_distance;
    std::tuple<unsigned int, unsigned int> max_G_parameters;
    std::tuple<unsigned int, unsigned int> max_R_parameters;
    
    if (max_G_distances_file.is_open() && max_R_distances_file.is_open() && max_disp_G_file.is_open() && max_disp_R_file.is_open()) {
        for_all_graphs(
            n_qubits,
            [&](const Eigen::MatrixXd &graph_Qfunc, const Eigen::Tensor<double, 3> &graph_symQ, const unsigned int &graph_num) {
                max_displaced_distances(n_qubits, qubitstate_size, graph_Qfunc, graph_symQ, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
                max_G_distances_file << max_G_distance << "\n";
                max_R_distances_file << max_R_distance << "\n";
                max_disp_G_file << std::get<0>(max_G_parameters) << ", " << std::get<1>(max_G_parameters) << "\n";
                max_disp_R_file << std::get<0>(max_R_parameters) << ", " << std::get<1>(max_R_parameters) << "\n";
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

    double max_G_distance;
    double max_R_distance;
    std::tuple<unsigned int, unsigned int, unsigned int> max_G_parameters;
    std::tuple<unsigned int, unsigned int, unsigned int> max_R_parameters;
    
    if (max_G_distances_file.is_open() && max_R_distances_file.is_open() && max_lClifford_G_file.is_open() && max_lClifford_R_file.is_open()) {
        for_all_graphs(
            n_qubits,
            [&](const Eigen::MatrixXd &graph_Qfunc, const Eigen::Tensor<double, 3> &graph_symQ, const unsigned int &graph_num) {
                max_lClifford_distances(n_qubits, qubitstate_size, graph_Qfunc, graph_symQ, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
                max_G_distances_file << max_G_distance << "\n";
                max_R_distances_file << max_R_distance << "\n";
                max_lClifford_G_file << std::get<0>(max_G_parameters) << ", " << std::get<1>(max_G_parameters) << ", " << std::get<2>(max_G_parameters) << "\n";
                max_lClifford_R_file << std::get<0>(max_R_parameters) << ", " << std::get<1>(max_R_parameters) << ", " << std::get<2>(max_R_parameters) << "\n";
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

int main() {    
    // calc_gen_graph_graph_symQ();
    // calc_full_displaced_graph_entropy(n_qubits, qubitstate_size, graph_num);
    // calc_all_displaced_graph_symQ(n_qubits, qubitstate_size, graph_num);
    // max_all_lClifford_graphs_distances();
    max_all_displaced_graphs_distances();
    

    return 0;
}