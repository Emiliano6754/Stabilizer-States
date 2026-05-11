#include "graph_state.h"
#include<chrono>
#include<sym_space.h>
#include<discrete_math.h>
#include<Qfunc.h>

// Creates a graph state with an empty graph of n_qubits vertices
graph_state::graph_state(unsigned int const& n_qubits) : n_qubits(n_qubits), graph(n_qubits, false), kravchuk_symQ(n_qubits) {
    state_size = 1 << n_qubits;
    mark_state_change();
}

// Creates the graph state of graph
graph_state::graph_state(simple_graph const& graph) : n_qubits(graph.n_nodes()), graph(graph), kravchuk_symQ(graph.n_nodes()) {
    state_size = 1 << n_qubits;
    mark_state_change();
}

// Copy constructor
graph_state::graph_state(graph_state& other) : n_qubits(other.n_qubits), state_size(other.state_size), graph(other.graph), valid_characteristic(other.valid_characteristic), characteristic(other.characteristic), valid_kravchuk_symQ(other.valid_kravchuk_symQ),  kravchuk_symQ(other.kravchuk_symQ), valid_symQ(other.valid_symQ), symQ(other.symQ), valid_Qfunc(other.valid_Qfunc), Qfunc(other.Qfunc) {

}

// Sets the graph of this to graph
graph_state& graph_state::set_graph(simple_graph const& in_graph) {
    graph = in_graph;
    mark_state_change();
    return *this;
}

// Sets the graph of this to the graph_num-th graph, where connected determines if the graphs counted are connected
graph_state& graph_state::set_graph(unsigned int const& graph_num, bool const &connected) {
    mark_state_change();
    return *this;
}

// Returns the characteristic function of the graph as an Eigen::Tensor<double, 3>. Computes characteristic if not valid
Eigen::Tensor<int, 3> const& graph_state::get_characteristic() {
    validate_characteristic();
    return characteristic;
}

// Returns the symQ function as a kravchuk_exp. Computes kravchuk_symQ if not valid
kravchuk_exp const& graph_state::get_kravchuk_symQ() {
    validate_kravchuk_symQ();
    return kravchuk_symQ;
}

// Returns a const reference to symQ. Computes symQ if not valid
Eigen::Tensor<double, 3> const& graph_state::get_symQ() {
    validate_symQ();
    return symQ;
}


// Returns a const reference to Qfunc. Computes Qfunc if not valid
Eigen::MatrixXd const& graph_state::get_Qfunc() {
    validate_Qfunc();
    return Qfunc;
}

// Returns the maximum distances to the Gaussian envelope and to the random distribution, in that order, over all clifford operations
std::pair<double, double> graph_state::get_max_distances() {
    std::pair<double, double> max_distances(0, 0);
    return max_distances;
}

// Saves the characteristic in data/characteristics/. Automatically includes a _q*n_qubits*.txt
graph_state const& graph_state::save_characteristic(std::string const& filename) {
    validate_characteristic();
    save_sym_func(characteristic, "characteristic/"+filename+"_q"+std::to_string(n_qubits)+".txt");
    return *this;
}
// Saves the kravchuk_exp in data/expansions/
graph_state const& graph_state::save_kravchuk_exp(std::string const& filename) {
    validate_kravchuk_symQ();
    save_sym_func(kravchuk_symQ.get_coeffs(), "kravchuk_exp/"+filename+"_q"+std::to_string(n_qubits)+".txt");
    return *this;
}
// Saves the symmetrized Q function in data/symQfuncs/
graph_state const& graph_state::save_symQ(std::string const& filename) {
    validate_symQ();
    save_symQfunc(symQ, filename+"_q"+std::to_string(n_qubits)+".txt");
    return *this;
}

// Saves the full Q function in data/Qfuncs/
graph_state const& graph_state::save_graph_Qfunc(std::string const& filename) {
    validate_Qfunc();
    save_Qfunc(Qfunc, filename+"_q"+std::to_string(n_qubits)+".txt");
    return *this;
}

// Updates the state of all bools from a state change
void graph_state::mark_state_change() {
    valid_characteristic = false;
    valid_kravchuk_symQ = false;
    valid_symQ = false;
    valid_Qfunc = false;
}

// Sets characteristic to a valid state
void graph_state::validate_characteristic() {
    if (!valid_characteristic) {
        compute_characteristic();
    }
}

// Sets kravchuk_symQ to a valid state
void graph_state::validate_kravchuk_symQ() {
    if (!valid_kravchuk_symQ) {
        compute_kravchuk_symQ();
    }
}

// Sets the symmetric Q function to a valid state
void graph_state::validate_symQ() {
    if (!valid_symQ) {
        compute_symQ();
    }
}

// Sets the Q function to a valid state
void graph_state::validate_Qfunc() {
    if (!valid_Qfunc) {
        compute_Qfunc();
    }
}

// Computes the characteristic, setting valid_characteristic to true
inline void graph_state::compute_characteristic() {
    graph.characteristic(characteristic);
    valid_characteristic = true;
}

// Computes the kravchuk expansion of the symQ, setting valid_kravchuk_symQ to true
void graph_state::compute_kravchuk_symQ() {
    validate_characteristic();
    kravchuk_symQ.set_zero();
    static std::vector<kravchuk_exp> gmnk = get_all_gmnk(n_qubits);
    std::vector<double> const& sqrt3_buffer = cached_power_buffer(1.0 / SQRT3, n_qubits);
    double norm = 1.0 / (1 << n_qubits);
    // p is the last index so that it runs faster, for cache efficiency
    sym_space_loop(n_qubits, [&](int const &r, int const &q, int const &p) {
        kravchuk_symQ.sum_mult(gmnk[p + (q + r * (n_qubits + 1)) * (n_qubits + 1)], norm * sqrt3_buffer[(p+q+r)/2] * characteristic(p, q, r));
    });
    valid_kravchuk_symQ = true;
}

// Computes the symQ, setting valid_symQ to true. Automatically removes negative values that come from numeric error with the kravchuk expansion
void graph_state::compute_symQ() {
    validate_kravchuk_symQ();
    symQ = kravchuk_symQ.as_binom_tensor().unaryExpr(negatives_to_zero);
    valid_symQ = true;
}

// Computes the Q function, setting valid_Qfunc to true
void graph_state::compute_Qfunc() {
    Qfunc.resize(state_size, state_size);
    Qfunc.setZero();
    std::vector<double> const& sqrt3_buffer = cached_power_buffer(1.0 / SQRT3, n_qubits);

    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        double coeff = 0;
        unsigned int mult = 0;
        #pragma omp for collapse(2)
        for (unsigned int alpha = 0; alpha < state_size; alpha++) {
            for (unsigned int beta = 0; beta < state_size; beta++) {
                coeff = 0;
                for (unsigned int eta = 0; eta < state_size; eta++) {
                    mult = this->graph * eta;
                    coeff += sign(beta & mult, alpha & eta) * sqrt3_buffer[(std::popcount(mult) + std::popcount(eta) + std::popcount(mult ^ eta)) / 2];
                }
                Qfunc(alpha,beta) = coeff / state_size;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating Qfunc took " << duration.count() << "s" << std::endl;
    valid_Qfunc = true;
}

static void save_state_props(graph_state &state, std::string const &filename, bool const &save_characteristic, bool const &save_kravchuk, bool const &save_symQ, bool const &save_Qfunc) {
    if (save_characteristic) {
        state.save_characteristic(filename);
    }
    if (save_kravchuk) {
        state.save_kravchuk_exp(filename);
    }
    if (save_symQ) {
        state.save_symQ(filename);
    }
    if (save_Qfunc) {
        state.save_graph_Qfunc(filename);
    }
}

// Asks the user what properties of the graph_state to compute and save, whether it is for all graphs with a fixed number of vertices or for a specified graph
void calc_save_graph_state_props() {
    bool all_graphs = false, connected = false, save_characteristic = false, save_kravchuk = false, save_symQ = false, save_Qfunc = false, maximize_distances = false;
    all_graphs = ask_bool("Compute all graphs?");
    save_characteristic = ask_bool("Save characteristic function?");
    save_kravchuk = ask_bool("Save kravchuk_exp coefficients?");
    save_symQ = ask_bool("Compute and save symmetric Q function?");
    save_Qfunc = ask_bool("Compute and save full Q function?");
    maximize_distances = ask_bool("Maximize distances?");
    if (all_graphs) {
        unsigned int n_qubits = ask_unsigned_int("Enter the number of qubits");
        connected = ask_bool("Restrict to connected graphs?");
        std::string filepath = connected ? "connected/" : "general/";
        if(maximize_distances) {
            const std::filesystem::path cwd = std::filesystem::current_path();
            std::string distances_filename = "/data/distances/"+filepath+"_q"+std::to_string(n_qubits)+".txt";
            std::ofstream distances_file(cwd.string()+distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
            std::pair<double, double> distances = {0, 0};
            if (distances_file.is_open()) {
                for_all_graph_states(n_qubits, connected, [&](graph_state &state, unsigned int const &graph_num) {
                    save_state_props(state, filepath + std::to_string(graph_num), save_characteristic, save_kravchuk, save_symQ, save_Qfunc);
                    distances = state.get_max_distances();
                    distances_file << distances.first << ", " << distances.second << "\n";
                });
            } else {
                std::cout << "Could not save distances in " << distances_filename << std::endl;
                return;
            }
        } else{
            for_all_graph_states(n_qubits, connected, [&](graph_state &state, unsigned int const &graph_num) {
                save_state_props(state, filepath + std::to_string(graph_num), save_characteristic, save_kravchuk, save_symQ, save_Qfunc);
            });
        }
    } else {
        unsigned int n_qubits = 0;
        std::string graph_type;
        simple_graph selected_graph = generate_selected_graph(n_qubits, graph_type);
        graph_state state(selected_graph);
        save_state_props(state, graph_type, save_characteristic, save_kravchuk, save_symQ, save_Qfunc);
        if(maximize_distances) {
            const std::filesystem::path cwd = std::filesystem::current_path();
            std::string distances_filename = "/data/distances/"+graph_type + "_q" + std::to_string(n_qubits) + ".txt";
            std::ofstream distances_file(cwd.string()+distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
            if (distances_file.is_open()) {
                std::pair<double, double> distances = state.get_max_distances();
                distances_file << distances.first << ", " << distances.second << "\n";
            } else {
                std::cout << "Could not save distances in " << distances_filename << std::endl;
                return;
            }
        }
    }
}

// Implement special maximization of distances? Maybe that should be performed over a general symQ, since it only needs to be performed from it?