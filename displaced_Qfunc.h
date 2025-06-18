#ifndef DISPLACEDQFUNCH
#define DISPLACEDQFUNCH
#include<Eigen/Dense>
#include<unsupported/Eigen/CXX11/Tensor>

void save_symQfunc(const Eigen::Tensor<double,3> &Qfunc, const std::string &filename); // Give access to save_symQfunc to displacedQfunc.cpp

// Maximizes the Hellinger distance of a Q function to its gaussian envelope and the Rmnk distribution, after displacements Z^mu X^nu. Returns both maximum distances, in the order {distance to G, distance to R}, together with the corresponding parameters of the optimizers in the order {mu, nu}, in the same order of distance to G then to R
void max_displaced_distances(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc, const Eigen::Tensor<double, 3> &symQ, double &max_distance_G, double &max_distance_R, std::tuple<unsigned int, unsigned int> &max_displacement_G, std::tuple<unsigned int, unsigned int> &max_displacement_R);

// Maximizes the Hellinger distance of a Q function to its gaussian envelope and the Rmnk distribution, after displacements and Hadamard gates H^gamma Z^mu X^nu (it is assumed that the displacement is applied first). Returns both maximum distances, in the order {distance to G, distance to R}, together with the corresponding parameters of the optimizers in the order {mu, nu, gamma}, in the same order of distance to G then to R
void max_lClifford_distances(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc, const Eigen::Tensor<double, 3> &symQ, double &max_distance_G, double &max_distance_R, std::tuple<unsigned int, unsigned int, unsigned int> &max_Clifford_G, std::tuple<unsigned int, unsigned int, unsigned int> &max_Clifford_R);

// Calculates the Rényi entropy after all possible displacements for the state given in Qfunc and outputs them in entropies. Both are assumed to already be of size 2^n_qubits x 2^n_qubits. For 15 qubits this requires >16 GB of ram. To circunvent this, the calculated values of entropy must be directly stored in memory
void calc_full_displaced_entropy(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &entropies);

// Calculates the Rényi entropy after all possible displacements for the state given in Qfunc and outputs them in entropies. Both are assumed to already be of size 2^n_qubits x 2^n_qubits. Also stores the displacements required for maximum and minimum entropy in max_displacement and min_displacement, respectively, as pairs. For 15 qubits this requires >16 GB of ram. To circunvent this, the calculated values of entropy must be directly stored in memory
void calc_full_displaced_maxmin_entropy(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &entropies, std::tuple<unsigned int, unsigned int> &max_displacement, std::tuple<unsigned int, unsigned int> &min_displacement);

// Calculates all displaced symQ functions for the given state and saves them on filepath, each on a file named by the corresponding displacement Z^mu X^nu as mu,nu.txt
void calc_all_displaced_symQ(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, std::string filepath);

#endif