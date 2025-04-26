#ifndef DISPLACEDQFUNCH
#define DISPLACEDQFUNCH
#include<Eigen/Dense>
#include<utility> // std::pair

void calc_full_displaced_entropy(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &entropies);
void calc_full_displaced_maxmin_entropy(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &entropies, std::pair<unsigned int, unsigned int> &max_displacement, std::pair<unsigned int, unsigned int> &min_displacement);

#endif