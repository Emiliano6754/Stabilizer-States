#ifndef DISPLACEDQFUNCH
#define DISPLACEDQFUNCH
#include<Eigen/Dense>
#include<unsupported/Eigen/CXX11/Tensor>

void save_symQfunc(const Eigen::Tensor<double,3> &Qfunc, const std::string &filename); // Give access to save_symQfunc to displacedQfunc.cpp
void calc_full_displaced_entropy(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &entropies);
void calc_full_displaced_maxmin_entropy(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &entropies, std::tuple<unsigned int, unsigned int> &max_displacement, std::tuple<unsigned int, unsigned int> &min_displacement);
void calc_all_displaced_symQ(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, std::string filepath);
void minmax_displaced_distance(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc, const Eigen::Tensor<double, 3> &symQ, double &min_distance, double &max_distance, std::tuple<unsigned int, unsigned int> &min_displacement, std::tuple<unsigned int, unsigned int> &max_displacement);
void minmax_lClifford_distance(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc, const Eigen::Tensor<double, 3> &symQ, double &min_distance, double &max_distance, std::tuple<unsigned int, unsigned int, unsigned int> &min_Clifford, std::tuple<unsigned int, unsigned int, unsigned int> &max_Clifford);

#endif