#include "graph.h"
#include<algorithm>
#include<immintrin.h>
#include<bit>
#include<bitset>

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

// Saves the graph characteristic function
void save_characteristic(Eigen::Tensor<int, 3> const &characteristic, std::string const &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/symQfuncs/characteristic/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    Eigen::TensorIOFormat csv_format = Eigen::TensorIOFormat(/*separator=*/{",\n", ""}, /*prefix=*/{"", ""}, /*suffix=*/{"", ""}, /*precision=*/Eigen::FullPrecision, /*flags=*/0, /*tenPrefix=*/"", /*tenSuffix=*/"");
    if (output_file.is_open()) {
        output_file << characteristic.format(csv_format) << std::endl;
    } else {
        std::cout << "Could not save characteristic" << std::endl;
    }
}


// Build a new simple graph with n_vertices vertices, where connected determines if any two vertices are connected by default
simple_graph::simple_graph(unsigned int const &n_vertices, bool const &connected) : n_vertices(n_vertices), adj(n_vertices, n_vertices, 0u) {
    if (connected) {
        unsigned int mask = 1 << n_vertices - 1;
        for (int j = 0; j < n_vertices; j++) {
            adj[j] ^= mask ^ (1 << j);
        }
    }
}

// Adds (or removes if already present) edge (a,b) to the graph. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
simple_graph& simple_graph::add_edge(const unsigned int &a, const unsigned int &b) {
    adj[a] ^= 1 << b;
    adj[b] ^= 1 << a;
    return *this;
}

// Adds (or removes if already present) edge (a,b) to the graph. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
simple_graph& simple_graph::add_edge(const std::pair<unsigned int, unsigned int> &edge) {
    adj[edge.first] ^= 1 << edge.second;
    adj[edge.second] ^= 1 << edge.first;
    return *this;
}

// Adds (or removes if already present) all edges from Edge_list to the graph. Edges (a,a) can't be added by this function, which is wanted behavior as we only allow simple graphs
simple_graph& simple_graph::add_edge_list(const Edge_list &edge_list) {
    for (int j = 0; j < edge_list.size(); j++) {
        adj[edge_list[j].first] ^= 1 << edge_list[j].second;
        adj[edge_list[j].second] ^= 1 << edge_list[j].first;
    }
    return *this;
}

// Adds (or removes if already present) all cyclic edges to the graph
simple_graph& simple_graph::add_cyclic_edges() {
    for (int j = 1; j < n_vertices; j++) {
        add_edge(j-1, j);
    }
    add_edge(n_vertices-1, 0);
    return *this;
}

// Clears all edges from the graph
simple_graph& simple_graph::clear_edges() {
    for (int j = 0; j < n_vertices; j++) {
        adj[j] = 0;
    }
    return *this;
}

// Prompts user to create a graph manually via introducing the vertex connections
simple_graph& simple_graph::add_manual_edges() {
    unsigned int q1, q2;
    bool finished = false;
    std::string input;
    while (!finished) {
        std::cout << "Enter a vertex connection" << std::endl;
        get_unsignedint(q1);
        get_unsignedint(q2);
        if (q1 < n_vertices || q2 < n_vertices) {
            add_edge(q1, q2);
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
    return *this;
}

// Returns the maximum degree of a vertex in this
unsigned int simple_graph::min_degree() const {
    unsigned int min = n_vertices;
    for (unsigned int n = 0; n < n_vertices; n++) {
        if (min > std::popcount(adj[n])) {
            min = std::popcount(adj[n]);
        }
    }
    return min;
}

// Returns the maximum degree of a vertex in this
unsigned int simple_graph::max_degree() const {
    unsigned int max = 0;
    for (unsigned int n = 0; n < n_vertices; n++) {
        if (max < std::popcount(adj[n])) {
            max = std::popcount(adj[n]);
        }
    }
    return max;
}

// Returns the average degree of vertices in this
double simple_graph::avg_degree() const {
    double avg = 0;
    for (unsigned int n = 0; n < n_vertices; n++) {
        avg += std::popcount(adj[n]);
    }
    return avg / n_vertices;
}

// Returns the minimum, maximum and average degrees of vertices in this, as a tuple in that order
std::tuple<unsigned int, unsigned int, double> simple_graph::degrees() const {
    unsigned int min = n_vertices, max = 0;
    double avg = 0;
    for (unsigned int n = 0; n < n_vertices; n++) {
        if (min > std::popcount(adj[n])) {
            min = std::popcount(adj[n]);
        }
        if (max < std::popcount(adj[n])) {
            max = std::popcount(adj[n]);
        }
        avg += std::popcount(adj[n]);
    }
    return std::tuple(min, max, avg / n_vertices);
}

// Returns the minimum, maximum and average cut-rank of this, as a tuple in that order. For now, the slowing factor seems to be allocation of memory. Could potentially define versions of GF2N_matrix where the number of rows and columns can be modified (as long as it fits the previous), so that no reallocation is needed? And only perform reallocation when actually needed, then start with the largest subset
std::tuple<unsigned int, unsigned int, double> simple_graph::cut_ranks() const {
    const uint32_t powerset_size = 1 << n_vertices;
    const uint32_t full_mask = powerset_size - 1;
    uint32_t inv_subset = 0, shifted_subset = 0, leading_pos = 0;
    unsigned int subset_size = 0, min_rank = n_vertices, max_rank = 0, current_rank = 0, trailing_zeros = 0;
    double avg_rank = 0;
    GF2N_matrix submatrix(n_vertices, n_vertices, 0u);
    for (uint32_t subset = 0; subset <= full_mask; subset++) {
        // Invert subset as a mask to get connections to non-active bits
        inv_subset = subset ^ full_mask;
        subset_size = std::popcount(subset);
        // Submatrix with the connections from subset to its complement
        leading_pos = 0;
        shifted_subset = subset;
        submatrix.set_zero();
        for (uint32_t j = 0; j < subset_size; j++) {
            // Find where the jth active bit in subset is
            trailing_zeros = std::countr_zero(shifted_subset);
            shifted_subset >>= trailing_zeros + 1;
            leading_pos += trailing_zeros;
            // Next row corresponds to the connections of the jth active bit with all non-active bits, in contiguous low bits
            submatrix[j] = _pext_u32(adj[leading_pos], inv_subset);
            leading_pos += 1;
        }
        current_rank = submatrix.rank();
        min_rank = std::min(min_rank, current_rank);
        max_rank = std::max(max_rank, current_rank);
        avg_rank += current_rank;
    }
    return std::tuple(min_rank, max_rank, avg_rank / powerset_size);
}

// Returns the rank-width of this. To be implemented
double simple_graph::rank_width() const {
    return 0;
}

// Clears the graph, then parses line as a graph in edge_list format from http://combos.org/nauty
simple_graph& simple_graph::parse_graph_from_line(std::string const &line) {
    clear_edges();
    std::string first, second;
    bool finished = false;
    unsigned int next_sc = 0;
    unsigned int comma_pos = 0;
    unsigned int last_sc = 0;
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
        add_edge(std::stoul(first) - 1, std::stoul(second) - 1);
        
        last_sc = next_sc;
    }
    return *this;
}

// Clears the graph, then parses the graph_num graph (as numerated in the graph list) with n_vertices unlabeled nodes from the edge list. Assumes the graph library file is named as n_qubits.txt
simple_graph& simple_graph::parse_graph_from_edge_list(const unsigned int &graph_num) {
    clear_edges();
    const std::filesystem::path cwd = std::filesystem::current_path();
    const std::string filename = std::to_string(n_vertices) + ".txt";
    std::ifstream input_file(cwd.string()+"/data/graphs/"+filename,std::ifstream::in);
    std::string line;
    unsigned int pos = 1;
    if (input_file.is_open()) {
        while (std::getline(input_file, line)) {
            if (pos == graph_num) {
                parse_graph_from_line(line);
                break;
            }
            pos++;
        }
    } else {
        std::cout << "Could not parse graph" << std::endl;
    }
    return *this;
}

// Returns the graph characteristic function C_A of Adj
Eigen::Tensor<int, 3> simple_graph::graph_characteristic() const {
    Eigen::Tensor<int, 3> C_A(n_vertices+1, n_vertices+1, n_vertices+1);
    C_A.setZero();
    unsigned int powerset_size = 1 << n_vertices;
    unsigned int mult = 0;
    for (unsigned int eta = 0; eta < powerset_size; eta++) {
        mult = adj.lmult(eta);
        C_A(std::popcount(eta), std::popcount(mult), std::popcount(mult^eta)) += 1;
    }
    return C_A;
}

// Returns reference to the adjacency matrix of this as a GF2N_matrix
GF2N_matrix& simple_graph::get_adj() {
    return adj;
}

// Returns const reference to the adjacency matrix of this as a GF2N_matrix
GF2N_matrix const& simple_graph::get_adj() const {
    return adj;
}

// Returns the number of vertices as a copy
unsigned int simple_graph::n_nodes() const {
    return n_vertices;
}

void calc_save_all_graph_properties() {
    unsigned int n_vertices = 0;
    std::cout << "Enter the number of vertices" << std::endl;
    get_unsignedint(n_vertices);
    const std::filesystem::path cwd = std::filesystem::current_path();
    const std::string graphs_suffix = std::to_string(n_vertices) + "_props.txt";
    std::ofstream save_file(cwd.string()+"/data/graphs/"+graphs_suffix,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (save_file.is_open()) {
        for_all_graphs(n_vertices, [&](const simple_graph &graph, unsigned int const &graph_num) {
            std::cout << "Calculating degrees" << std::endl;
            std::tuple<unsigned int, unsigned int, double> degrees = graph.degrees();
            std::cout << "Calculating ranks" << std::endl;
            std::tuple<unsigned int, unsigned int, double> ranks = graph.cut_ranks();
            std::cout << "Saving degrees and ranks" << std::endl;
            save_file << std::get<0>(degrees) << ", " << std::get<1>(degrees) << ", " << std::get<2>(degrees) << ", " << std::get<0>(ranks) << ", " << std::get<1>(ranks) << ", " << std::get<2>(ranks) << "\n";
        });
        save_file.close();
    } else {
        std::cout << "Could not save graph properties" << std::endl;
    }
}