#ifndef GRAPH_H
#define GRAPH_H

#include<vector>
#include<string>
#include<utility> // std::pair

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

#endif