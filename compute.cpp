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
#include<sstream>
#include<memory>
#include<omp.h>
#include<displaced_Qfunc.h>
#include<sym_space.h>
#include<Qfunc.h>
#include<GF2N.h>
#include<discrete_math.h>
#include "graph.h"
#include "graph_state.h"
#include "graph_generator.h"

// // Calculates the symmetric Q function of a maximally connected graph state with removed cyclic edges
// void generate_acyclic_symQ(const unsigned int &n_qubits) {
//     const std::string filename = "ac_q" + std::to_string(n_qubits)+".txt";
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     init_Adj(Adj,n_qubits,1);
//     add_cyclic_edges(n_qubits,Adj);
//     calc_save_symQ(n_qubits,Adj,filename);
// }

// // Calculates the symmetric Q function of a cyclically connected graph state
// void generate_cyclic_symQ(const unsigned int &n_qubits) {
//     const std::string filename = "cc_q" + std::to_string(n_qubits)+".txt";
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     init_Adj(Adj,n_qubits,0);
//     add_cyclic_edges(n_qubits,Adj);
//     calc_save_symQ(n_qubits,Adj,filename);
// }

// // Calculates the symmetric Q function of a maximally connected graph state
// void generate_maxcon_symQ(const unsigned int &n_qubits) {
//     const std::string filename = "mc_q" + std::to_string(n_qubits)+".txt";
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     init_Adj(Adj,n_qubits,1);
//     calc_save_symQ(n_qubits,Adj,filename);
// }

// void generate_discon_symQ(const unsigned int &n_qubits) {
//     const std::string filename = "dc_q" + std::to_string(n_qubits)+".txt";
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     init_Adj(Adj,n_qubits,0);
//     calc_save_symQ(n_qubits,Adj,filename);
// }

// // Calculates the Q function of a maximally connected graph state with removed cyclic edges
// void generate_acyclic_graphQ(const unsigned int &n_qubits) {
//     const std::string filename = "ac_q" + std::to_string(n_qubits)+".txt";
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     init_Adj(Adj,n_qubits,1);
//     add_cyclic_edges(n_qubits,Adj);
//     calc_save_graph_symQ(n_qubits,Adj,filename);
// }

// // Calculates the Q function of a cyclically connected graph state
// void generate_cyclic_graphQ(const unsigned int &n_qubits) {
//     const std::string filename = "cc_q" + std::to_string(n_qubits)+".txt";
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     init_Adj(Adj,n_qubits,0);
//     add_cyclic_edges(n_qubits,Adj);
//     calc_save_graph_symQ(n_qubits,Adj,filename);
// }

// // Calculates the Q function of a maximally connected graph state
// void generate_maxcon_graphQ(const unsigned int &n_qubits) {
//     const std::string filename = "mc_q" + std::to_string(n_qubits)+".txt";
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     init_Adj(Adj,n_qubits,1);
//     calc_save_graph_symQ(n_qubits,Adj,filename);
// }

// void generate_discon_graphQ(const unsigned int &n_qubits) {
//     const std::string filename = "dc_q" + std::to_string(n_qubits)+".txt";
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     init_Adj(Adj,n_qubits,0);
//     calc_save_graph_symQ(n_qubits,Adj,filename);
// }

// unsigned int parse_unsignedint(const std::string &input) {
//     try {
//         unsigned long u = std::stoul(input);
//         if (u > std::numeric_limits<unsigned int>::max())
//             throw std::out_of_range(input);

//         return u;
//     } catch (const std::invalid_argument& e) {
//         std::cout << "Input could not be parsed: " << e.what() << std::endl;
//     } catch (const std::out_of_range& e) {
//         std::cout << "Input out of range: " << e.what() << std::endl;
//     }
//     return 0;
// }

// void sel_calc_state_symQ(const unsigned int &n_qubits) {
//     bool selected = false;
//     while (!selected) {
//         std::cout << "Select the graph type [m(aximmally connected),c(yclically connected),a(cyclically connected),d(isconnected)]" << std::endl;
//         std::string input;
//         std::cin >> input;
//         if (input == "mc" || input == "m") {
//             generate_maxcon_symQ(n_qubits);
//             selected = true;
//         } else if (input == "cc" || input == "c") {
//             generate_cyclic_symQ(n_qubits);
//             selected = true;
//         } else if (input == "ac" || input == "a") {
//             generate_acyclic_symQ(n_qubits);
//             selected = true;
//         } else if (input == "dc" || input == "d") {
//             generate_discon_symQ(n_qubits);
//             selected = true;
//         }
//     }
// }

// void sel_calc_state_graph_symQ(const unsigned int &n_qubits) {
//     bool selected = false;
//     while (!selected) {
//         std::cout << "Select the graph type [m(aximmally connected),c(yclically connected),a(cyclically connected),d(isconnected)]" << std::endl;
//         std::string input;
//         std::cin >> input;
//         if (input == "mc" || input == "m") {
//             generate_maxcon_graphQ(n_qubits);
//             selected = true;
//         } else if (input == "cc" || input == "c") {
//             generate_cyclic_graphQ(n_qubits);
//             selected = true;
//         } else if (input == "ac" || input == "a") {
//             generate_acyclic_graphQ(n_qubits);
//             selected = true;
//         } else if (input == "dc" || input == "d") {
//             generate_discon_graphQ(n_qubits);
//             selected = true;
//         }
//     }
// }

// static void get_unsignedint(unsigned int &parsed_input) {
//     std::string input = "";
//     std::cin >> input;
//     try {
//         unsigned long u = std::stoul(input);
//         if (u > std::numeric_limits<unsigned int>::max())
//             throw std::out_of_range(input);

//         parsed_input = u;
//     } catch (const std::invalid_argument& e) {
//         std::cout << "Input could not be parsed: " << e.what() << std::endl;
//     } catch (const std::out_of_range& e) {
//         std::cout << "Input out of range: " << e.what() << std::endl;
//     }
// }

// static void get_double(double &parsed_input) {
//     std::string input = "";
//     std::cin >> input;
//     try {
//         double u = std::stod(input);
//         if (u > std::numeric_limits<double>::max())
//             throw std::out_of_range(input);

//         parsed_input = u;
//     } catch (const std::invalid_argument& e) {
//         std::cout << "Input could not be parsed: " << e.what() << std::endl;
//     } catch (const std::out_of_range& e) {
//         std::cout << "Input out of range: " << e.what() << std::endl;
//     }
// }

// void calc_manual_graph() {
//     unsigned int n_qubits = 0;
//     std::cout << "Enter the number of qubits" << std::endl;
//     get_unsignedint(n_qubits);
//     unsigned int* Adj = static_cast<unsigned int*>(_malloca(n_qubits * n_qubits * sizeof(unsigned int)));
//     std::string filename = "";
//     parse_manual_graph(n_qubits, Adj, filename);
//     calc_save_graph_symQ(n_qubits, Adj, filename);
// }

// void calc_gen_graph_symQ() {
//     unsigned int N = 0;
//     std::cout << "Enter the number of qubits" << std::endl;
//     get_unsignedint(N);
//     sel_calc_state_symQ(N);
// }

// void calc_gen_graph_graph_symQ() {
//     unsigned int N = 0;
//     std::cout << "Enter the number of qubits" << std::endl;
//     get_unsignedint(N);
//     sel_calc_state_graph_symQ(N);
// }

// void graphQ_from_file(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &graph_num, unsigned int* Adj, Eigen::MatrixXd &Qfunc) {
//     parse_graph_from_edge_list(n_qubits, graph_num, Adj);
    
//     Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
//     auto start = std::chrono::high_resolution_clock::now();
//     graphQ(Qfunc, sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<float> duration = end - start;
//     std::cout << "Calculating Q took " << duration.count() << "s" << std::endl;
// }

// Calculates the full displaced entropies of a particular graph state, specified by the number of qubits and graph_num
// void calc_full_displaced_graph_entropy(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &graph_num) {
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
//     std::string filename = "entropies/" + std::to_string(n_qubits) + "q_" + std::to_string(graph_num) + ".txt"; // Use folder inside Qfuncs as this is secondary
    
//     graphQ_from_file(n_qubits, qubitstate_size, graph_num, Adj, Qfunc);

//     Eigen::MatrixXd entropies(qubitstate_size, qubitstate_size);
//     std::tuple<unsigned int, unsigned int> max_displacement = {0, 0};
//     std::tuple<unsigned int, unsigned int> min_displacement = {0, 0};
//     auto start = std::chrono::high_resolution_clock::now();
//     calc_full_displaced_maxmin_entropy(Qfunc, n_qubits, qubitstate_size, entropies, max_displacement, min_displacement);
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = end - start;
//     std::cout << "Calculating displacements took " << duration.count() << "s" << std::endl;
//     save_Qfunc(entropies, filename);
// }

// void calc_all_displaced_graph_symQ(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &graph_num) {
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
//     std::string filepath = std::to_string(n_qubits) + "q_" + std::to_string(graph_num);

//     graphQ_from_file(n_qubits, qubitstate_size, graph_num, Adj, Qfunc);

//     calc_all_displaced_symQ(Qfunc, n_qubits, qubitstate_size, filepath);
// }

// void manual_minmax_displaced_graph_distance() {
//     unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
//     unsigned int const qubitstate_size = 1 << n_qubits;
//     unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));;
//     std::string filename = "";
//     generate_selected_graph(n_qubits, Adj, filename);

//     double max_distance = 0;
//     double min_distance = 1.0e10;
//     std::tuple<unsigned int, unsigned int> max_displacement = {0, 0};
//     std::tuple<unsigned int, unsigned int> min_displacement = {0, 0};

//     Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
//     Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    
//     auto start = std::chrono::high_resolution_clock::now();
//     graphQ(Qfunc, sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<float> duration = end - start;
//     std::cout << "Calculating Q took " << duration.count() << "s" << std::endl;

//     start = std::chrono::high_resolution_clock::now();
//     max_displaced_distances(n_qubits, qubitstate_size, Qfunc, sym_Qfunc, min_distance, max_distance, min_displacement, max_displacement);
//     end = std::chrono::high_resolution_clock::now();
//     duration = end - start;
//     std::cout << "Calculating displacements took " << duration.count() << "s" << std::endl;
//     std::cout << "Max distance: " << max_distance << std::endl;
//     std::cout << "Min distance: " << min_distance << std::endl;
//     std::cout << "Max displacement: " << std::get<0>(max_displacement) << ", " << std::get<1>(max_displacement) << std::endl;
//     std::cout << "Min displacement: " << std::get<0>(min_displacement) << ", " << std::get<1>(min_displacement) << std::endl;
// }

// void manual_minmax_lClifford_graph_distance() {
//     unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
//     unsigned int const qubitstate_size = 1 << n_qubits;
//     unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));;
//     std::string filename = "";
//     generate_selected_graph(n_qubits, Adj, filename);

//     double max_distance = 0;
//     double min_distance = 1.0e10;
//     std::tuple<unsigned int, unsigned int, unsigned int> max_displacement = {0, 0, 0};
//     std::tuple<unsigned int, unsigned int, unsigned int> min_displacement = {0, 0, 0};

//     Eigen::MatrixXd Qfunc(qubitstate_size,qubitstate_size);
//     Eigen::Tensor<double,3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
    
//     auto start = std::chrono::high_resolution_clock::now();
//     graphQ(Qfunc, sym_Qfunc.setZero(), n_qubits, qubitstate_size, Adj);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<float> duration = end - start;
//     std::cout << "Calculating Q took " << duration.count() << "s" << std::endl;

//     start = std::chrono::high_resolution_clock::now();
//     max_lClifford_distances(n_qubits, qubitstate_size, Qfunc, sym_Qfunc, min_distance, max_distance, min_displacement, max_displacement);
//     end = std::chrono::high_resolution_clock::now();
//     duration = end - start;
//     std::cout << "Calculating displacements took " << duration.count() << "s" << std::endl;
//     std::cout << "Max distance: " << max_distance << std::endl;
//     std::cout << "Min distance: " << min_distance << std::endl;
//     std::cout << "Max displacement: " << std::get<0>(max_displacement) << ", " << std::get<1>(max_displacement) << std::endl;
//     std::cout << "Min displacement: " << std::get<0>(min_displacement) << ", " << std::get<1>(min_displacement) << std::endl;
// }

// Loops over all graphs with n_qubits, calculating both their Q and symmetrized Q functions and executes a particular function acting on them and the graph number
// template<typename LoopFunc> 
// void for_all_graphs_Qfuncs(const unsigned int &n_qubits, LoopFunc operate_graph) {
//     const std::filesystem::path cwd = std::filesystem::current_path();
//     std::string graphs_suffix = std::to_string(n_qubits) + ".txt";
//     std::ifstream input_file(cwd.string()+"/data/graphs/"+graphs_suffix,std::ifstream::in);
//     std::string line;
//     unsigned int graph_num = 1;
    
//     unsigned int qubitstate_size = 1 << n_qubits;
//     unsigned int* const Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     Eigen::MatrixXd graph_Qfunc(qubitstate_size, qubitstate_size);
//     Eigen::Tensor<double, 3> graph_symQ(n_qubits + 1, n_qubits + 1, n_qubits + 1);

//     if (input_file.is_open()) {
//         while (std::getline(input_file, line)) {
//             parse_graph_line(n_qubits, line, Adj);
//             graphQ(graph_Qfunc, graph_symQ.setZero(), n_qubits, qubitstate_size, Adj);
            
//             operate_graph(graph_Qfunc, graph_symQ, graph_num);

//             graph_num++;
//         }
//     } else {
//         std::cout << "Could not parse graphs" << std::endl;
//     }
// }


// void max_all_displaced_graphs_distances() {
//     unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
//     unsigned int const qubitstate_size = 1 << n_qubits;

//     const std::filesystem::path cwd = std::filesystem::current_path();
//     std::string save_folder = cwd.string()+"/data/disp_dist/";
//     std::string max_G_distances_filename = "max_G_q" + std::to_string(n_qubits) + ".txt";
//     std::string max_R_distances_filename = "max_R_q" + std::to_string(n_qubits) + ".txt";
//     std::string max_disp_G_filename = "max_disp_G_q" + std::to_string(n_qubits) + ".txt";
//     std::string max_disp_R_filename = "max_disp_R_q" + std::to_string(n_qubits) + ".txt";
//     std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
//     std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
//     std::ofstream max_disp_G_file(save_folder + max_disp_G_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
//     std::ofstream max_disp_R_file(save_folder + max_disp_R_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

//     double max_G_distance, max_R_distance;
//     std::tuple<unsigned int, unsigned int> max_G_parameters, max_R_parameters;
    
//     if (max_G_distances_file.is_open() && max_R_distances_file.is_open() && max_disp_G_file.is_open() && max_disp_R_file.is_open()) {
//         for_all_graphs_Qfuncs(
//             n_qubits,
//             [&](const Eigen::MatrixXd &graph_Qfunc, const Eigen::Tensor<double, 3> &graph_symQ, const unsigned int &graph_num) {
//                 max_displaced_distances(n_qubits, qubitstate_size, graph_Qfunc, graph_symQ, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
//                 max_G_distances_file << max_G_distance << "\n";
//                 max_R_distances_file << max_R_distance << "\n";
//                 max_disp_G_file << std::get<0>(max_G_parameters) << ", " << std::get<1>(max_G_parameters) << "\n";
//                 max_disp_R_file << std::get<0>(max_R_parameters) << ", " << std::get<1>(max_R_parameters) << "\n";
//                 std::cout << graph_num << std::endl;
//             }
//         );
//         max_G_distances_file.close();
//         max_R_distances_file.close();
//         max_disp_G_file.close();
//         max_disp_R_file.close();

//     } else {
//         std::cout << "Could not save minmax results" << std::endl;
//     }
// }

// void max_all_lClifford_graphs_distances() {
//     unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
//     unsigned int const qubitstate_size = 1 << n_qubits;

//     const std::filesystem::path cwd = std::filesystem::current_path();
//     std::string save_folder = cwd.string()+"/data/clif_dist/";
//     std::string max_G_distances_filename = "max_G_q" + std::to_string(n_qubits) + ".txt";
//     std::string max_R_distances_filename = "max_R_q" + std::to_string(n_qubits) + ".txt";
//     std::string max_lClifford_G_filename = "max_cliff_G_q" + std::to_string(n_qubits) + ".txt";
//     std::string max_lClifford_R_filename = "max_cliff_R_q" + std::to_string(n_qubits) + ".txt";
//     std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
//     std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
//     std::ofstream max_lClifford_G_file(save_folder + max_lClifford_G_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
//     std::ofstream max_lClifford_R_file(save_folder + max_lClifford_R_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

//     double max_G_distance, max_R_distance;
//     std::tuple<unsigned int, unsigned int, unsigned int> max_G_parameters, max_R_parameters;
    
//     if (max_G_distances_file.is_open() && max_R_distances_file.is_open() && max_lClifford_G_file.is_open() && max_lClifford_R_file.is_open()) {
//         for_all_graphs_Qfuncs(
//             n_qubits,
//             [&](const Eigen::MatrixXd &graph_Qfunc, const Eigen::Tensor<double, 3> &graph_symQ, const unsigned int &graph_num) {
//                 max_lClifford_distances(n_qubits, qubitstate_size, graph_Qfunc, graph_symQ, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
//                 max_G_distances_file << max_G_distance << "\n";
//                 max_R_distances_file << max_R_distance << "\n";
//                 max_lClifford_G_file << std::get<0>(max_G_parameters) << ", " << std::get<1>(max_G_parameters) << ", " << std::get<2>(max_G_parameters) << "\n";
//                 max_lClifford_R_file << std::get<0>(max_R_parameters) << ", " << std::get<1>(max_R_parameters) << ", " << std::get<2>(max_R_parameters) << "\n";
//                 std::cout << graph_num << std::endl;
//             }
//         );
//         max_G_distances_file.close();
//         max_R_distances_file.close();
//         max_lClifford_G_file.close();
//         max_lClifford_R_file.close();

//     } else {
//         std::cout << "Could not save minmax results" << std::endl;
//     }
// }

// void max_random_displaced_graphs_distances() {
//     unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
//     unsigned int const qubitstate_size = 1 << n_qubits;
//     unsigned int const n_graphs = ask_unsigned_int("Enter the number of distintict graphs to be generated");

//     std::vector<unsigned int> seeds = ask_unsigned_ints("Enter random engine seeds");
//     std::vector<Edge_list> graphs(n_graphs);
//     set_engine_seed(seeds);
//     generate_random_edge_connected_graph_set(n_qubits, n_graphs, graphs);

//     const std::filesystem::path cwd = std::filesystem::current_path();
//     std::string save_folder = cwd.string()+"/data/disp_dist/";
//     std::string max_G_distances_filename = "rand_max_G_q" + std::to_string(n_qubits) + ".txt";
//     std::string max_R_distances_filename = "rand_max_R_q" + std::to_string(n_qubits) + ".txt";
//     std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
//     std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

//     double max_G_distance, max_R_distance;
//     std::tuple<unsigned int, unsigned int> max_G_parameters, max_R_parameters;
//     unsigned int* Adj = static_cast<unsigned int*>(alloca(n_qubits * sizeof(unsigned int)));
//     Eigen::MatrixXd graph_Qfunc(qubitstate_size,qubitstate_size);
//     Eigen::Tensor<double,3> graph_symQ(n_qubits+1,n_qubits+1,n_qubits+1);
    
//     unsigned int count = 1;
//     if (max_G_distances_file.is_open() && max_R_distances_file.is_open()) {
//         for (Edge_list edge_list : graphs) {
//             init_Adj(Adj, n_qubits, 0);
//             add_edge_list(n_qubits, edge_list, Adj);
//             graphQ(graph_Qfunc, graph_symQ.setZero(), n_qubits, qubitstate_size, Adj);
//             max_displaced_distances(n_qubits, qubitstate_size, graph_Qfunc, graph_symQ, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
//             max_G_distances_file << max_G_distance << "\n";
//             max_R_distances_file << max_R_distance << "\n";
//             std::cout << count << std::endl;
//             count++;
//         }
//         max_G_distances_file.close();
//         max_R_distances_file.close();

//     } else {
//         std::cout << "Could not save minmax results" << std::endl;
//     }
// }

// Returns the expected value of Sx^r for a graph state given by its adjacency matrix Adj
int exp_val_Sx_r(const unsigned int &n_qubits, const unsigned int &r, GF2N_matrix const &adj) {
    int exp_val = 0;
    unsigned int adj_sums = 0, eta = 0, pairs = 0;
    nested_basis_loop(n_qubits, r, 
    [&] (unsigned int const* const j_vector) {
        adj_sums = eta = pairs = 0;
        for (int l = 0; l < r; l++) {
            adj_sums ^= adj[j_vector[l] - 1]; // Sum of the j_vector rows of Adj (j starts at 1, but the first element in Adj is 0)
            eta ^= 1 << (j_vector[l] - 1); // Sum of the j_vector basis vectors (j starts at 1, but the first basis vector should be 0x1)
        }
        if (adj_sums == 0) {
            pairs = 0;
            for (int m = 1; m <= n_qubits; m++) {
                for (int k = 1; k <= m; k++) {
                    pairs += get_bit(adj[m], k) * get_bit(eta, m) * get_bit(eta, k);
                }
            }
            exp_val += 1 - 2 * (pairs % 2);
        }
    }
    );
    return exp_val;
}

// Calculates the exact expected values of Sx^r for r = 1, ..., 4 for all graph states with a prompted number of qubits and saves them in the same order as the database
void classify_graphs() {
    unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
    bool connected = ask_bool("Restrict to connected graphs?");
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream save_file(cwd.string()+"/data/graphs/Sx_exp_q"+std::to_string(n_qubits)+".txt",std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    if (save_file.is_open()) {
        for_all_graphs(n_qubits, connected,
        [&] (simple_graph const &graph, const unsigned int &graph_num) {
            for (int j = 1; j <= 4; j++) {
                save_file << exp_val_Sx_r(n_qubits, j, graph.get_adj()) << ", ";
            }
            save_file << "\n";
            std::cout << graph_num << "\n";
        }
        );
    } else {
        std::cout << "Could not save expected values" << std::endl;
    }
}

void save_gmnk() {
    unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
    unsigned int const qubitstate_size = 1 << n_qubits;
    std::vector<kravchuk_exp> gmnk = get_all_gmnk(n_qubits);
    int num = 0;
    std::string prefix = "gmnk/q" + std::to_string(n_qubits) + "_";
    sym_space_loop(n_qubits, [&] (int const &r, int const &q, int const &p) {
        save_symQfunc(gmnk[p + (q + r * (n_qubits + 1)) * (n_qubits + 1)].as_binom_tensor(), prefix + std::to_string(num) + ".txt");
        num++;
    });
}

void compare_graph_localization() {
    unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
    unsigned int const qubitstate_size = 1 << n_qubits;
    bool connected = ask_bool("Restrict to connected graphs?");
    Eigen::Tensor<double, 0> loc;
    Eigen::Tensor<int, 0> loc2;
    
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream save_file(cwd.string()+"/data/graphs/loc_q"+std::to_string(n_qubits)+".txt",std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (save_file.is_open()) {
        auto start = std::chrono::high_resolution_clock::now();
        for_all_graph_states(n_qubits, connected, [&] (graph_state &state, unsigned int const &graph_num) {
            Eigen::Tensor<double, 3> const& symQ = state.get_symQ();
            loc = symQ.square().sum();
            save_file << loc(0) << ", ";
            Eigen::Tensor<int, 3> const& C_A = state.get_characteristic();
            loc2 = C_A.square().sum();
            save_file << loc2(0) << "\n";
            std::cout << graph_num << "\n";
        });
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        std::cout << "Calculating took " << duration.count() << "s" << std::endl;
    } else {
        std::cout << "Could not save localization values" << std::endl;
    }
}

void save_coherent_state_symQ() {
    unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
    unsigned int const qubitstate_size = 1 << n_qubits;
    unsigned int const type = ask_unsigned_int("Select coherent state type: 0 for SU(2) coherent, 1 for discrete coherent");
    Eigen::Tensor<double, 3> characteristic, symQ;
    std::string filename;
    double n_x, n_y, n_z, norm;
    unsigned int s, t, u;
    switch(type) {
        case 0:
        std::cout << "Enter a Bloch vector (possibly non-normalized)" << std::endl;
        n_x = parse_double();
        n_y = parse_double();
        n_z = parse_double();
        norm = n_x * n_x + n_y * n_y + n_z * n_z;
        n_x /= norm;
        n_y /= norm;
        n_z /= norm;
        characteristic = get_Rmnk(n_qubits);
        sym_space_loop(n_qubits, [&] (int const &r, int const &q, int const &p) {
            characteristic(p, q, r) *= std::pow(n_x, (-p + q + r) / 2) * std::pow(n_y, (p + q - r) / 2) * std::pow(n_z, (p - q + r) / 2);
        });
        symQ = characteristic_symQ(n_qubits, qubitstate_size, characteristic);
        std::cout << "Enter a filename, without number of qubits and filetype" << std::endl;
        std::cin >> filename;
        save_symQfunc(symQ, filename + "_q" + std::to_string(n_qubits) + ".txt");
        return;
        case 1:
        std::cout << "Enter valid weights of the coherent state" << std::endl;
        s = parse_unsigned_int();
        t = parse_unsigned_int();
        u = parse_unsigned_int();
        characteristic = get_gmnk(n_qubits, s, t, u);
        symQ = characteristic_symQ(n_qubits, qubitstate_size, characteristic);
        filename = "coherent_" + std::to_string(s) + "_" + std::to_string(t) + "_" + std::to_string(u) + "_q" + std::to_string(n_qubits) + ".txt";
        save_symQfunc(symQ, filename);
        return;
        default:
        std::cout << "Non valid coherent type" << std::endl;
        return;
    }
}

void max_displaced_disconnected_graphs_distances() {
    unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
    unsigned int const qubitstate_size = 1 << n_qubits;

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string save_folder = cwd.string()+"/data/disp_dist/complete/";
    std::string max_G_distances_filename = "max_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_R_distances_filename = "max_R_q" + std::to_string(n_qubits) + ".txt";
    std::string max_disp_G_filename = "max_disp_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_disp_R_filename = "max_disp_R_q" + std::to_string(n_qubits) + ".txt";
    std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_disp_G_file(save_folder + max_disp_G_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_disp_R_file(save_folder + max_disp_R_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    double max_G_distance, max_R_distance;
    std::tuple<unsigned int, unsigned int> max_G_parameters, max_R_parameters;


    if (max_G_distances_file.is_open() && max_R_distances_file.is_open() && max_disp_G_file.is_open() && max_disp_R_file.is_open()) {
        for_all_graph_states(
            n_qubits, false, 
            [&](graph_state &state, const unsigned int &graph_num) {
                max_displaced_distances(n_qubits, qubitstate_size, state.get_Qfunc(), state.get_symQ(), max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
                max_G_distances_file << max_G_distance << "\n";
                max_R_distances_file << max_R_distance << "\n";
                max_disp_G_file << std::get<0>(max_G_parameters) << ", " << std::get<1>(max_G_parameters) << "\n";
                max_disp_R_file << std::get<0>(max_R_parameters) << ", " << std::get<1>(max_R_parameters) << "\n";
                std::cout << graph_num << "\n";
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

void max_clifford_disconnected_graphs_distances() {
    unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
    unsigned int const qubitstate_size = 1 << n_qubits;

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string save_folder = cwd.string()+"/data/clif_dist/complete/";
    std::string max_G_distances_filename = "max_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_R_distances_filename = "max_R_q" + std::to_string(n_qubits) + ".txt";
    std::string max_cliff_G_filename = "max_cliff_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_cliff_R_filename = "max_cliff_R_q" + std::to_string(n_qubits) + ".txt";
    std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_cliff_G_file(save_folder + max_cliff_G_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_cliff_R_file(save_folder + max_cliff_R_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    double max_G_distance, max_R_distance;
    std::tuple<unsigned int, unsigned int, unsigned int> max_G_parameters, max_R_parameters;


    if (max_G_distances_file.is_open() && max_R_distances_file.is_open() && max_cliff_G_file.is_open() && max_cliff_R_file.is_open()) {
        for_all_graph_states(
            n_qubits, false, 
            [&](graph_state &state, const unsigned int &graph_num) {
                max_lClifford_distances(n_qubits, qubitstate_size, state.get_Qfunc(), state.get_symQ(), max_G_distance, max_R_distance, max_G_parameters, max_R_parameters);
                max_G_distances_file << max_G_distance << "\n";
                max_R_distances_file << max_R_distance << "\n";
                max_cliff_G_file << std::get<0>(max_G_parameters) << ", " << std::get<1>(max_G_parameters) << ", " << std::get<2>(max_G_parameters) << "\n";
                max_cliff_R_file << std::get<0>(max_R_parameters) << ", " << std::get<1>(max_R_parameters) << ", " << std::get<2>(max_R_parameters) << "\n";
                std::cout << graph_num << "\n";
            }
        );
        max_G_distances_file.close();
        max_R_distances_file.close();
        max_cliff_G_file.close();
        max_cliff_R_file.close();

    } else {
        std::cout << "Could not save minmax results" << std::endl;
    }
}

int main() {
    max_clifford_disconnected_graphs_distances();

    return 0;
}