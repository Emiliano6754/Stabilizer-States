#include "graph.h"
#include<algorithm>
#include<iostream>
#include<fstream>
#include<filesystem>

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

// Initializes adjacency matrix with n_vertices vertices, def controls whether all edges are connected or disconnected, with disconnected as default. def should only be 0 or 1, undefined behavior otherwise
void init_Adj(unsigned int* Adj, const unsigned int &n_vertices, const unsigned int def) {
    if (n_vertices > 8*sizeof(unsigned int)) {
        std::cout << "Too many qubits, change unsigned int in adjacency matrices to use more" << std::endl;
    } else {
        unsigned int connection = def * ((1 << n_vertices) - 1);
        for (unsigned int i = 0; i < n_vertices; i++) {
            Adj[i] = connection ^ (def << i);
        }
    }
}

// Adds (or removes if already present) edge (a,b) to the adjacency matrix. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
void add_edge(unsigned int* Adj, const unsigned int &size, const unsigned int &a, const unsigned int &b) {
    if (size < a+1 || size < b+1) {
        std::cout << "Edges outside bounds" << std::endl;
    } else {
        Adj[a] = Adj[a] ^ (1 << b);
        Adj[b] = Adj[b] ^ (1 << a);
    }
}

// Adds (or removes if already present) edge (a,b) to the adjacency matrix. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
void add_edge(unsigned int* Adj, const unsigned int &size, const std::pair<unsigned int, unsigned int> &edge) {
    if (size < edge.first+1 || size < edge.second+1) {
        std::cout << "Edges outside bounds" << std::endl;
    } else {
        Adj[edge.first] = Adj[edge.first] ^ (1 << edge.second);
        Adj[edge.second] = Adj[edge.second] ^ (1 << edge.first);
    }
}

// Adds (or removes if already present) all edges from Edge_list to the adjacency matrix. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
void add_edge_list(const unsigned int &n_qubits, const Edge_list &edge_list, unsigned int* Adj) {
    for (std::pair<unsigned int, unsigned int> edge : edge_list) {
        add_edge(Adj, n_qubits, edge);
    }
}

// Adds (or removes if already present) all cyclic edges to the adjacency matrix
void add_cyclic_edges(const unsigned int &n_qubits, unsigned int* Adj) {
    for (unsigned int n = 0; n < n_qubits-1; n++) {
        add_edge(Adj,n_qubits,n,n+1);
    }
    add_edge(Adj,n_qubits,n_qubits-1,0);
}

// Returns the maximum degree of a vertex in Adj
unsigned int get_max_degree(const unsigned int* Adj, const unsigned int &n_qubits) {
    unsigned int max = 0;
    for (unsigned int n = 0; n < n_qubits; n++) {
        if (max < std::popcount(Adj[n])) {
            max = std::popcount(Adj[n]);
        }
    }
    return max;
}

// Prompts user to create a graph manually via introducing the vertex connections
void parse_manual_graph(const unsigned int &n_qubits, unsigned int* Adj, std::string &filename) {
    init_Adj(Adj, n_qubits, 0);
    unsigned int q1, q2;
    bool finished = false;
    std::string input;
    while (!finished) {
        std::cout << "Enter a vertex connection" << std::endl;
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

// Parses a graph line in edge_list format from http://combos.org/nauty
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
            next_sc = line.size();
        }
        first = line.substr(last_sc+1, comma_pos-last_sc-1);
        second = line.substr(comma_pos+1, next_sc-comma_pos-1);
        add_edge(Adj, n_qubits, std::stoul(first) - 1, std::stoul(second) - 1);
        
        last_sc = next_sc;
    }
}

// Parses the graph_num graph (as numerated in the graph list) with n unlabeled nodes as the adjacency matrix in Adj from the edge list. Assumes the file is named as n_qubits.txt
void parse_graph_from_edge_list(const unsigned int &n_qubits, const unsigned int &graph_num, unsigned int* Adj) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string filename = std::to_string(n_qubits) + ".txt";
    std::ifstream input_file(cwd.string()+"/data/graphs/"+filename,std::ifstream::in);
    std::string line;
    unsigned int pos = 1;

    if (input_file.is_open()) {
        while (std::getline(input_file, line)) {
            if (pos == graph_num) {
                parse_graph_line(n_qubits, line, Adj);
                break;
            }
            pos++;
        }
    } else {
        std::cout << "Could not parse graph" << std::endl;
    }
}

// Prompts the user to select a graph from the database
void ask_manual_graph_params(unsigned int &n_qubits, unsigned int &graph_num) {
    std::cout << "Enter the number of qubits" << std::endl;
    get_unsignedint(n_qubits);
    std::cout << "Enter the graph number" << std::endl;
    get_unsignedint(graph_num);
}

// Prompts for a type of graph, the number of qubits and returns its adjacency matrix and a suffix for filenames to indicate its type
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

// Returns Adj * eta, where Adj is treated as a matrix and eta a vector, in GF(2^N). Assumes Adj is symmetric, so that no transpose is needed
static unsigned int adj_mult(unsigned int const &n_qubits, unsigned int *Adj, unsigned int const &eta) {
    unsigned int res = 0;
    for (unsigned int j = 0; j < n_qubits; j++) {
        res ^= Adj[j] * ((eta >> j) & 1);
    }
    return res;
}

// Returns the graph characteristic function C_A of Adj
Eigen::Tensor<int, 3> graph_characteristic(unsigned int const &n_qubits, unsigned int const &qubitstate_size, unsigned int *Adj) {
    Eigen::Tensor<int, 3> C_A(n_qubits+1, n_qubits+1, n_qubits+1);
    C_A.setZero();
    unsigned int mult = 0;
    for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
        mult = adj_mult(n_qubits, Adj, eta);
        C_A(std::popcount(eta), std::popcount(mult), std::popcount(mult^eta)) += 1;
    }
    return C_A;
}
