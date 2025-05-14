#include "graph_generator.h"
#include<iostream>
#include<random>
#include<list>
#include<algorithm>

static std::mt19937 random_engine;
static bool engine_initialized = false;

// Initializes the mersenne twister engine with a non-deterministic random seed. Sets engine_initialized to true
void initialize_engine() {
    std::random_device rand_dev;
    random_engine.seed(rand_dev());
    engine_initialized = true;
}

// Consumes the given seeds to produce a more spread sequence of 32 bit seeds, which are then fed into the engine. Given a set of seeds, the resulting generated numbers are deterministic. Also sets engine_initialized to true
void set_engine_seed(std::vector<unsigned int> &seeds) {
    std::seed_seq seq(seeds.begin(), seeds.end());
    random_engine.seed(seq);
    engine_initialized = true;
}

// Generates a random connected graph of n_nodes nodes with n_edges edges, and returns it in edge_list. Requires the random engine to be initiliazed first
void generate_connected_graph(const unsigned int &n_nodes, const unsigned int &n_edges, std::vector<std::pair<unsigned int, unsigned int>> &edge_list) {
    if (n_edges < n_nodes || n_edges > (n_nodes * (n_nodes - 1)) / 2) {
        std::cout << n_edges << " are not enough or excessive edges for a connected graph with " << n_nodes << " nodes" << std::endl;
        return;
    }
    if (!engine_initialized) {
        std::cout << "Engine not initialized" << std::endl;
        return;
    }
    static std::uniform_int_distribution<unsigned int> random_dist(0, n_nodes-1);
    unsigned int current_node, neighbor_node;
    std::list<unsigned int> new_nodes, visited_nodes;
    std::list<unsigned int>::iterator node_pos; 
    edge_list.reserve(n_edges);
    for (unsigned int n = 0; n < n_nodes; n++) {
        new_nodes.push_back(n);
    }
    // Move the starting node to the visited nodes
    current_node = random_dist(random_engine);
    node_pos = new_nodes.begin();
    std::advance(node_pos, current_node);
    visited_nodes.splice(visited_nodes.end(), new_nodes, node_pos);
    
    while (new_nodes.size()) {
        // Select one of the nodes randomly as the future position
        neighbor_node = random_dist(random_engine);
        
        node_pos = std::find(new_nodes.begin(), new_nodes.end(), neighbor_node);
        // If the node has not been visited (i.e. it's still new), add an edge connecting it to the last node and mark it as visited
        if (node_pos != new_nodes.end()) {
            edge_list.push_back({current_node, neighbor_node});
            visited_nodes.splice(visited_nodes.end(), new_nodes, node_pos);
        }
        // Move to this node
        current_node = neighbor_node;
    }
    
    std::pair<unsigned int, unsigned int> edge;
    // Add random edges until reaching the desired number of edges
    while (edge_list.size() < n_edges) {
        edge = {random_dist(random_engine), random_dist(random_engine)};
        // If the edge is not already present, add it
        if (std::find(edge_list.begin(), edge_list.end(), edge) == edge_list.end()) {
            edge_list.push_back(edge);
        }
    }
}

// Generates a random connected graph of n_nodes nodes with a random number of edges, and returns it in edge_list. Requires the random engine to be initiliazed first, but its output for a given seed is deterministic
void generate_random_edge_connected_graph(const unsigned int &n_nodes, std::vector<std::pair<unsigned int, unsigned int>> &edge_list) {
    std::uniform_int_distribution<unsigned int> random_dist(n_nodes, (n_nodes * (n_nodes - 1)) / 2);
    generate_connected_graph(n_nodes, random_dist(random_engine), edge_list);
}

// To be implemented. Generates a set of n_graphs graphs with n_nodes nodes with a random number of edges, so that no graph is repeated
void generate_random_edge_connected_graph_set(const unsigned int &n_nodes, const unsigned int &n_graphs) {

}