#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H

#include<vector>
#include<utility>

typedef std::vector<std::pair<unsigned int, unsigned int>> Edge_list;

// Initializes the mersenne twister engine with a non-deterministic random seed. Sets engine_initialized to true
void initialize_engine();

// Consumes the given seeds to produce a more spread sequence of 32 bit seeds, which are then fed into the engine. Given a set of seeds, the resulting generated numbers are deterministic. Also sets engine_initialized to true
void set_engine_seed(std::vector<unsigned int> &seeds);

// Generates a random connected graph of n_nodes nodes with n_edges edges, and returns it in edge_list. Edges are returned as std::pairs, where in each pair the lowest numbered node comes first. Requires the random engine to be initiliazed first
void generate_connected_graph(const unsigned int &n_nodes, const unsigned int &n_edges, Edge_list &edge_list);

// Generates a random connected graph of n_nodes nodes with a random number of edges, and returns it in edge_list. Edges are returned as std::pairs, where in each pair the lowest numbered node comes first. Requires the random engine to be initiliazed first, but its output for a given seed is deterministic
void generate_random_edge_connected_graph(const unsigned int &n_nodes, Edge_list &edge_list);

// Generates a set of n_graphs graphs with n_nodes nodes with a random number of edges in graphs, so that no graph is repeated. The edge list has pairs ordered so that the lowest numbered node comes first. Requires the random engine to be initialized first
void generate_random_edge_connected_graph_set(const unsigned int &n_nodes, const unsigned int &n_graphs, std::vector<Edge_list> &graphs);

#endif