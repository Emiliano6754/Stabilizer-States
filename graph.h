#ifndef GRAPH_H
#define GRAPH_H

#include<vector>
#include<string>
#include<utility> // std::pair
#include<unsupported/Eigen/CXX11/Tensor>
#include<iostream>
#include<filesystem>
#include<fstream>
#include "GF2N_matrix.h"

typedef std::vector<std::pair<unsigned int, unsigned int>> Edge_list;

// Initializes adjacency matrix with n_vertices vertices, def controls whether all edges are connected or disconnected, with disconnected as default. def should only be 0 or 1, undefined behavior otherwise
void init_Adj(unsigned int* Adj, const unsigned int &n_vertices, const unsigned int def=0);

// Adds (or removes if already present) edge (a,b) to the adjacency matrix. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
void add_edge(unsigned int* Adj, const unsigned int &size, const unsigned int &a, const unsigned int &b);

// Adds (or removes if already present) edge (a,b) to the adjacency matrix. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
void add_edge(unsigned int* Adj, const unsigned int &size, const std::pair<unsigned int, unsigned int> &edge);

// Adds (or removes if already present) all edges from Edge_list to the adjacency matrix. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
void add_edge_list(const unsigned int &n_qubits, const Edge_list &edge_list, unsigned int* Adj);

// Adds (or removes if already present) all cyclic edges to the adjacency matrix
void add_cyclic_edges(const unsigned int &n_qubits, unsigned int* Adj);

// Returns the maximum degree of a vertex in Adj
unsigned int get_max_degree(const unsigned int* Adj, const unsigned int &n_qubits);

// Prompts user to create a graph manually via introducing the vertex connections
void parse_manual_graph(const unsigned int &n_qubits, unsigned int* Adj, std::string &filename);

// Parses a graph line in edge_list format from http://combos.org/nauty
void parse_graph_line(const unsigned int &n_qubits, std::string &line, unsigned int* Adj);

// Parses the graph_num graph (as numerated in the graph list) with n unlabeled nodes as the adjacency matrix in Adj from the edge list. Assumes the file is named as n_qubits.txt
void parse_graph_from_edge_list(const unsigned int &n_qubits, const unsigned int &graph_num, unsigned int* Adj);

// Prompts the user to select a graph from the database
void ask_manual_graph_params(unsigned int &n_qubits, unsigned int &graph_num);

// Prompts for a type of graph, the number of qubits and returns its adjacency matrix and a suffix for filenames to indicate its type
void generate_selected_graph(unsigned int &n_qubits, unsigned int *Adj, std::string &filename);

// Returns the graph characteristic function C_A of Adj
Eigen::Tensor<int, 3> graph_characteristic(unsigned int const &n_qubits, unsigned int const &qubitstate_size, unsigned int *Adj);

// Saves the graph characteristic function characteristic
void save_characteristic(Eigen::Tensor<int, 3> const &characteristic, std::string const &filename);

// Loops over all graphs with n_qubits, calculating their adjacency matrix and executes a particular function acting on it and the graph number. Check that this function is doubly defined in graphstate.cpp
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

class simple_graph {
public:
    // Build a new simple graph with n_vertices vertices, where connected determines if any two vertices are connected by default
    simple_graph(unsigned int const &n_vertices, bool const &connected = false);
    // Adds (or removes if already present) edge (a,b) to the graph. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
    simple_graph& add_edge(const unsigned int &a, const unsigned int &b);
    // Adds (or removes if already present) edge (a,b) to the graph. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
    simple_graph& add_edge(const std::pair<unsigned int, unsigned int> &edge);
    // Adds (or removes if already present) all edges from Edge_list to the graph. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
    simple_graph& add_edge_list(const Edge_list &edge_list);
    // Adds (or removes if already present) all cyclic edges to the graph
    simple_graph& add_cyclic_edges();
    // Clears all edges from the graph
    simple_graph& clear_edges();
    // Prompts user to create a graph manually via introducing the vertex connections
    simple_graph& add_manual_edges();
    // Returns the maximum degree of a vertex in this
    unsigned int min_degree() const;
    // Returns the maximum degree of a vertex in this
    unsigned int max_degree() const;
    // Returns the average degree of vertices in this
    double avg_degree() const;
    // Returns the minimum, maximum and average degrees of vertices in this, as a tuple in that order
    std::tuple<unsigned int, unsigned int, double> degrees() const;
    // Returns the minimum, maximum and average cut-rank of this, as a tuple in that order
    std::tuple<unsigned int, unsigned int, double> cut_ranks() const;
    // Returns the rank-width of this
    double rank_width() const;
    // Clears the graph, then parses line as a graph in edge_list format from http://combos.org/nauty
    simple_graph& parse_graph_from_line(std::string const &line);
    // Clears the graph, then parses the graph_num graph (as numerated in the graph list) with n_vertices unlabeled nodes from the edge list. Assumes the graph library file is named as n_qubits.txt
    simple_graph& parse_graph_from_edge_list(const unsigned int &graph_num);
    // Returns the graph characteristic function C_A of Adj
    Eigen::Tensor<int, 3> graph_characteristic() const;
    // Returns reference to the adjacency matrix of this as a GF2N_matrix
    GF2N_matrix& get_adj();
    // Returns const reference to the adjacency matrix of this as a GF2N_matrix
    GF2N_matrix const& get_adj() const;
    // Returns the number of vertices as a copy
    unsigned int n_nodes() const;
private:
    unsigned int const n_vertices;
    GF2N_matrix adj;
};

// Loops over all connected graphs with n_vertices nodes, parsing them as a simple_graph and executes a particular function acting on it and the graph number
template<typename LoopFunc> 
void for_all_connected_graphs(const unsigned int &n_vertices, LoopFunc operate_graph) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string graphs_suffix = std::to_string(n_vertices) + ".txt";
    std::ifstream input_file(cwd.string()+"/data/graphs/"+graphs_suffix,std::ifstream::in);
    std::string line;
    unsigned int graph_num = 1;
    
    simple_graph graph(n_vertices, false);
    if (input_file.is_open()) {
        while (std::getline(input_file, line)) {
            graph.parse_graph_from_line(line);
            
            operate_graph(graph, graph_num);

            graph_num++;
        }
    } else {
        std::cout << "Could not parse graphs" << std::endl;
    }
}

// Loops over all (not necessarily connected) graphs with n_vertices nodes, parsing them as a simple_graph and executes a particular function acting on it and the graph number
template<typename LoopFunc> 
void for_all_graphs(const unsigned int &n_vertices, LoopFunc operate_graph) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string line;
    unsigned int graph_num = 1;
    simple_graph graph(n_vertices, false);
    for (int con_vertices = 0; con_vertices <= n_vertices; con_vertices++) {
        std::ifstream input_file(cwd.string()+"/data/graphs/"+std::to_string(con_vertices)+".txt",std::ifstream::in);
        if (input_file.is_open()) {
            while (std::getline(input_file, line)) {
                graph.parse_graph_from_line(line);
                
                operate_graph(graph, con_vertices, graph_num);
    
                graph_num++;
            }
        } else {
            std::cout << "Could not parse graphs" << std::endl;
        }
    }
    
}

void calc_save_all_graph_properties();

#endif