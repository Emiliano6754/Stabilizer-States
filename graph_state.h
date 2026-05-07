#ifndef GRAPH_STATE_H
#define GRAPH_STATE_H

#include<utility>
#include<Eigen/Dense>
#include<unsupported/Eigen/CXX11/Tensor>
#include<kravchuk.h>
#include "graph.h"

class graph_state {
public:
    // Creates a graph state with an empty graph of n_qubits vertices
    graph_state(unsigned int const& n_qubits);
    // Creates the graph state of graph
    graph_state(simple_graph const& graph);
    // Copy constructor
    graph_state(graph_state& other);
    // Move constructor
    graph_state(graph_state&& other) = default;
    // Sets the graph of this to graph
    graph_state& set_graph(simple_graph const& graph);
    // Sets the graph of this to the graph_num-th graph, where connected determines if the graphs counted are connected
    graph_state& set_graph(unsigned int const& graph_num, bool const &connected = true);
    // Returns the characteristic function of the graph as an Eigen::Tensor<double, 3> const reference. Computes characteristic if not valid
    Eigen::Tensor<int, 3> const& get_characteristic();
    // Returns the symQ function as a kravchuk_exp const reference. Computes kravchuk_symQ if not valid
    kravchuk_exp const& get_kravchuk_symQ();
    // Returns a const reference to symQ. Computes symQ if not valid
    Eigen::Tensor<double, 3> const& get_symQ();
    // Returns a const reference to Qfunc. Computes Qfunc if not valid
    Eigen::MatrixXd const& get_Qfunc();
    // Returns the maximum distances to the Gaussian envelope and to the random distribution, in that order, over all clifford operations
    std::pair<double, double> get_max_distances();
    // Saves the characteristic in data/characteristics/
    graph_state const& save_characteristic(std::string const &filename);
    // Saves the kravchuk_exp in data/expansions/
    graph_state const& save_kravchuk_exp(std::string const &filename);
    // Saves the symmetrized Q function in data/symQfuncs/
    graph_state const& save_symQ(std::string const &filename);
    // Saves the full Q function in data/Qfuncs/
    graph_state const& save_graph_Qfunc(std::string const &filename);
    
private:
    unsigned int n_qubits;
    unsigned int state_size;
    // bool valid_graph; // If the number of qubits is to be mutable, then a check for a valid_graph must be performed on every function that uses graph. Otherwise, the setter function for the number of qubits must also set graph to a valid state
    simple_graph graph;
    bool valid_characteristic;
    Eigen::Tensor<int, 3> characteristic;
    bool valid_kravchuk_symQ;
    kravchuk_exp kravchuk_symQ;
    bool valid_symQ;
    Eigen::Tensor<double, 3> symQ;
    bool valid_Qfunc;
    Eigen::MatrixXd Qfunc;

    // Updates the state of all bools from a state change
    void mark_state_change();
    // Sets characteristic to a valid state
    void validate_characteristic();
    // Sets kravchuk_symQ to a valid state
    void validate_kravchuk_symQ();
    // Sets the symmetric Q function to a valid state
    void validate_symQ();
    // Sets the Q function to a valid state
    void validate_Qfunc();
    // Computes the characteristic, setting valid_characteristic to true
    inline void compute_characteristic();
    // Computes the kravchuk expansion of the symQ, setting valid_kravchuk_symQ to true
    void compute_kravchuk_symQ();
    // Computes the symQ, setting valid_symQ to true
    void compute_symQ();
    // Computes the Q function, setting valid_Qfunc to true
    void compute_Qfunc();
};

template<typename loop_func>
void for_all_graph_states(unsigned int const &n_qubits, bool const &connected, loop_func operate_state) {
    graph_state state(n_qubits);
    for_all_graphs(n_qubits, connected, [&](simple_graph const &graph, unsigned int const &graph_num) {
        state.set_graph(graph);
        operate_state(state, graph_num);
    });
}

// Asks the user what properties of the graph_state to compute and save, whether it is for all graphs with a fixed number of vertices or for a specified graph
void calc_save_graph_state_props();

#endif