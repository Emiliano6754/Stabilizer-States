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

class simple_graph {
public:
    // Build a new simple graph with n_vertices vertices, where connected determines if any two vertices are connected by default
    simple_graph(unsigned int const &n_vertices, bool const &connected = false);
    // Copy constructor
    simple_graph(simple_graph const &other);
    // Move constructor
    simple_graph(simple_graph &&other) = default;
    // Adds (or removes if already present) edge (a,b) to the graph. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
    simple_graph& add_edge(const unsigned int &a, const unsigned int &b);
    // Adds (or removes if already present) edge (a,b) to the graph. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
    simple_graph& add_edge(const std::pair<unsigned int, unsigned int> &edge);
    // Adds (or removes if already present) all edges from Edge_list to the graph. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
    simple_graph& add_edge_list(const Edge_list &edge_list);
    // Adds (or removes if already present) all cyclic edges to the graph
    simple_graph& add_cyclic_edges();
    // Adds (or removes if already present) all edges to the graph
    simple_graph& add_all_edges();
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
    Eigen::Tensor<int, 3> characteristic() const;
    // Computes the graph characteristic function C_A of Adj in characteristic, to avoid allocations
    void characteristic(Eigen::Tensor<int, 3> &characteristic) const;
    // Returns reference to the adjacency matrix of this as a GF2N_matrix
    GF2N_matrix& get_adj();
    // Returns const reference to the adjacency matrix of this as a GF2N_matrix
    GF2N_matrix const& get_adj() const;
    // Returns the number of vertices as a const reference
    unsigned int const& n_nodes() const;
    // Copy assignment operator
    simple_graph& operator=(simple_graph const& other);
    // Move assignment operator
    simple_graph& operator=(simple_graph&&) = default;
    // Vector product of vec by the adjacency matrix of this
    unsigned int operator*(unsigned int const &vec) const;
private:
    unsigned int const n_vertices;
    GF2N_matrix adj;
};

// Loops over all graphs with n_vertices nodes, parsing them as a simple_graph and executes a particular function acting on it and the graph number. If connected is true, it restricts to connected graphs. Could be significantly improved at the cost of some memory if all adjacency matrices are parsed first, then operate_graph executed over those, in order
template<typename LoopFunc> 
void for_all_graphs(const unsigned int &n_vertices, bool const &connected, LoopFunc operate_graph) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    const std::string filepath = connected ? "connected/" : "general/";
    std::ifstream input_file(cwd.string()+"/data/graphs/"+filepath+std::to_string(n_vertices)+".txt",std::ifstream::in);
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

// Calculates interesting properties of all graphs with a selected number of vertices and saves them in /data/graphs/
void calc_save_all_graph_properties();

// Prompts for a type of graph, the number of qubits and returns its adjacency matrix and a suffix for filenames to indicate its type
simple_graph generate_selected_graph(unsigned int const &n_qubits, std::string &filename);

#endif