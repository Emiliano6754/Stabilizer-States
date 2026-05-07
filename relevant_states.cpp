#include<fstream>
#include<iostream>
#include<Eigen/Dense>
#include<filesystem>
#include<chrono>
#include<tuple>
#include<states.h>
#include<displaced_Qfunc.h>
#include<Qfunc.h>
#include<sym_space.h>
#include<discrete_math.h>

# define M_PI           3.14159265358979323846  /* pi */

void max_coherent_state(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, double &max_G_distance, double &max_R_distance, std::tuple<unsigned int, unsigned int, unsigned int> &max_G_parameters, std::tuple<unsigned int, unsigned int, unsigned int> &max_R_parameters, const std::string &filename) {
    Eigen::VectorXcd coherent_state = su2_coherent_state(n_qubits, qubitstate_size, theta, phi);
    max_state_lClifford_distances(n_qubits, qubitstate_size, coherent_state, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters, filename);
}

void max_interesting_states() {
    unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
    unsigned int const qubitstate_size = 1 << n_qubits;

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string save_folder = cwd.string()+"/data/clif_dist/";
    std::string max_G_distances_filename = "states_max_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_R_distances_filename = "states_max_R_q" + std::to_string(n_qubits) + ".txt";
    std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    double max_G_distance, max_R_distance;
    std::tuple<unsigned int, unsigned int, unsigned int> max_G_parameters, max_R_parameters;

    
    auto start = std::chrono::high_resolution_clock::now();
    if (max_G_distances_file.is_open() && max_R_distances_file.is_open()) {
        max_coherent_state(n_qubits, qubitstate_size, M_PI/2, 0, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters, "coherent_q"+std::to_string(n_qubits)+".txt");
        max_G_distances_file << max_G_distance << "\n";
        max_R_distances_file << max_R_distance << "\n";
        
        Eigen::VectorXd state = GHZ_state(n_qubits);
        max_state_lClifford_distances(n_qubits, qubitstate_size, state, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters, "GHZ_q"+std::to_string(n_qubits)+".txt");
        max_G_distances_file << max_G_distance << "\n";
        max_R_distances_file << max_R_distance << "\n";
        
        state = cluster_state(n_qubits, qubitstate_size);
        max_state_lClifford_distances(n_qubits, qubitstate_size, state, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters, "cluster_q"+std::to_string(n_qubits)+".txt");
        max_G_distances_file << max_G_distance << "\n";
        max_R_distances_file << max_R_distance << "\n";
        
        state = singlet_state(n_qubits);
        max_state_lClifford_distances(n_qubits, qubitstate_size, state, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters, "singlet_q"+std::to_string(n_qubits)+".txt");
        max_G_distances_file << max_G_distance << "\n";
        max_R_distances_file << max_R_distance << "\n";
        
        state = W_state(n_qubits, qubitstate_size);
        max_state_lClifford_distances(n_qubits, qubitstate_size, state, max_G_distance, max_R_distance, max_G_parameters, max_R_parameters, "W_q"+std::to_string(n_qubits)+".txt");
        max_G_distances_file << max_G_distance << "\n";
        max_R_distances_file << max_R_distance << "\n";

        max_G_distances_file.close();
        max_R_distances_file.close();

    } else {
        std::cout << "Could not save minmax results" << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating took " << duration.count() << "s" << std::endl;


}

void get_state_distances_Kravchuk(const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::VectorXcd &state, const std::string &prefix, double &distance_G, double &distance_R) {
    const Eigen::MatrixXd Qfunc = pure_Qfunc_from_operational(n_qubits, qubitstate_size, state);
    const Eigen::Tensor<double, 3> sym_Qfunc = get_symQ(n_qubits, qubitstate_size, Qfunc);
    const Eigen::Tensor<double, 3> Rmnk = get_Rmnk(n_qubits);
    Eigen::Tensor<double, 3> Gfunc(n_qubits+1, n_qubits+1, n_qubits+1);
    Eigen::Tensor<double, 3> Kravchuk_exp(n_qubits+1, n_qubits+1, n_qubits+1);
    get_Kravchuk_expansion_Gfunc(n_qubits, qubitstate_size, sym_Qfunc, Gfunc, Kravchuk_exp);
    save_symQfunc(sym_Qfunc, prefix+"_symQ_q"+std::to_string(n_qubits)+".txt");
    save_symQfunc(Gfunc, prefix+"_Gfunc_q"+std::to_string(n_qubits)+".txt");
    save_symQfunc(Kravchuk_exp, prefix+"_Kravchuk_q"+std::to_string(n_qubits)+".txt");
    Eigen::Tensor<double, 0> result;
    result = (Gfunc * sym_Qfunc).sqrt().sum();
    distance_G = result(0);
    result = (Rmnk * sym_Qfunc).sqrt().sum();
    distance_R = result(0);
}

void get_state_distances_Kravchuk(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXd &state, const std::string &prefix, double &distance_G, double &distance_R) {
    const Eigen::MatrixXd Qfunc = pure_Qfunc_from_operational(n_qubits, qubitstate_size, state);
    const Eigen::Tensor<double, 3> sym_Qfunc = get_symQ(n_qubits, qubitstate_size, Qfunc);
    const Eigen::Tensor<double, 3> Rmnk = get_Rmnk(n_qubits);
    Eigen::Tensor<double, 3> Gfunc(n_qubits+1, n_qubits+1, n_qubits+1);
    Eigen::Tensor<double, 3> Kravchuk_exp(n_qubits+1, n_qubits+1, n_qubits+1);
    get_Kravchuk_expansion_Gfunc(n_qubits, qubitstate_size, sym_Qfunc, Gfunc, Kravchuk_exp);
    save_symQfunc(sym_Qfunc, prefix+"_symQ_q"+std::to_string(n_qubits)+".txt");
    save_symQfunc(Gfunc, prefix+"_Gfunc_q"+std::to_string(n_qubits)+".txt");
    save_symQfunc(Kravchuk_exp, prefix+"_Kravchuk_q"+std::to_string(n_qubits)+".txt");
    Eigen::Tensor<double, 0> result;
    result = (Gfunc * sym_Qfunc).sqrt().sum();
    distance_G = result(0);
    result = (Rmnk * sym_Qfunc).sqrt().sum();
    distance_R = result(0);
}

void expand_coherent_state(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const std::string &prefix, double &G_distance, double &R_distance) {
    Eigen::VectorXcd coherent_state = su2_coherent_state(n_qubits, qubitstate_size, theta, phi);
    get_state_distances_Kravchuk(n_qubits, qubitstate_size, coherent_state, prefix, G_distance, R_distance);
}

void expand_interesting_states() {
    unsigned int const n_qubits = ask_unsigned_int("Enter the number of qubits");
    unsigned int const qubitstate_size = 1 << n_qubits;

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string save_folder = cwd.string()+"/data/clif_dist/";
    std::string max_G_distances_filename = "states_max_G_q" + std::to_string(n_qubits) + ".txt";
    std::string max_R_distances_filename = "states_max_R_q" + std::to_string(n_qubits) + ".txt";
    std::ofstream max_G_distances_file(save_folder + max_G_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::ofstream max_R_distances_file(save_folder + max_R_distances_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);

    double G_distance, R_distance;

    
    auto start = std::chrono::high_resolution_clock::now();
    if (max_G_distances_file.is_open() && max_R_distances_file.is_open()) {
        expand_coherent_state(n_qubits, qubitstate_size, M_PI/3, M_PI/4, "coherent", G_distance, R_distance);
        max_G_distances_file << G_distance << "\n";
        max_R_distances_file << R_distance << "\n";
        
        Eigen::VectorXd state = GHZ_state(n_qubits);
        get_state_distances_Kravchuk(n_qubits, qubitstate_size, state, "GHZ", G_distance, R_distance);
        max_G_distances_file << G_distance << "\n";
        max_R_distances_file << R_distance << "\n";
        
        state = cluster_state(n_qubits, qubitstate_size);
        get_state_distances_Kravchuk(n_qubits, qubitstate_size, state, "cluster", G_distance, R_distance);
        max_G_distances_file << G_distance << "\n";
        max_R_distances_file << R_distance << "\n";
        
        state = singlet_state(n_qubits);
        get_state_distances_Kravchuk(n_qubits, qubitstate_size, state, "singlet", G_distance, R_distance);
        max_G_distances_file << G_distance << "\n";
        max_R_distances_file << R_distance << "\n";
        
        state = W_state(n_qubits, qubitstate_size);
        get_state_distances_Kravchuk(n_qubits, qubitstate_size, state, "W", G_distance, R_distance);
        max_G_distances_file << G_distance << "\n";
        max_R_distances_file << R_distance << "\n";

        max_G_distances_file.close();
        max_R_distances_file.close();

    } else {
        std::cout << "Could not save minmax results" << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Calculating took " << duration.count() << "s" << std::endl;


}

int main() {
    expand_interesting_states();
    
    return 0;
}