#include "displaced_Qfunc.h"
#include<Eigen/Dense>
#include<utility> // std::pair
#include<iostream>
#include<filesystem>
#include<fstream>

// Calculates the Rényi entropy after all possible displacements for the state given in Qfunc and outputs them in entropies. Both are assumed to already be of size 2^n_qubits x 2^n_qubits. For 15 qubits this requires >16 GB of ram. To circunvent this, the calculated values of entropy must be directly stored in memory
void calc_full_displaced_entropy(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &entropies) {
    Eigen::Array<Eigen::IndexPair<int>, 3, 1> contraction_indices = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(2,2)};
    #pragma omp parallel
    {
        Eigen::Tensor<double, 3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
        Eigen::Tensor<double, 0> entropy; // Necessary to force evaluation of the tensor expressions
        #pragma omp for
        for (unsigned int mu = 0; mu < qubitstate_size; mu++) {
            for (unsigned int nu = 0; nu < qubitstate_size; nu++) {
                sym_Qfunc.setZero();
                for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
                    for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                        sym_Qfunc(std::popcount(alpha ^ mu), std::popcount(beta ^ nu), std::popcount(alpha ^ beta ^ mu ^ nu)) += Qfunc(alpha, beta); // Should test if it is faster to make the sums in symQfunc or Qfunc
                    }
                }
                entropy = sym_Qfunc.contract(sym_Qfunc, contraction_indices);
                entropies(mu, nu) = entropy(0);
            }
        }
    }
}

// Calculates the Rényi entropy after all possible displacements for the state given in Qfunc and outputs them in entropies. Both are assumed to already be of size 2^n_qubits x 2^n_qubits. Also stores the displacements required for maximum and minimum entropy in max_displacement and min_displacement, respectively, as pairs. For 15 qubits this requires >16 GB of ram. To circunvent this, the calculated values of entropy must be directly stored in memory
void calc_full_displaced_maxmin_entropy(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &entropies, std::pair<unsigned int, unsigned int> &max_displacement, std::pair<unsigned int, unsigned int> &min_displacement) {
    Eigen::Array<Eigen::IndexPair<int>, 3, 1> contraction_indices = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(2,2)};
    double global_max_entropy = 0;
    double global_min_entropy = 1;
    #pragma omp parallel
    {
        Eigen::Tensor<double, 3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
        Eigen::Tensor<double, 0> entropy; // Necessary to force evaluation of the tensor expressions

        // Let each thread calculate a minimum/maximum and then start comparing their values as they finish
        std::pair<unsigned int, unsigned int> max_mu_nu = {0,0};
        std::pair<unsigned int, unsigned int> min_mu_nu = {0,0};
        double max_entropy = 0;
        double min_entropy = 1;
        #pragma omp for nowait
        for (unsigned int mu = 0; mu < qubitstate_size; mu++) {
            for (unsigned int nu = 0; nu < qubitstate_size; nu++) {
                sym_Qfunc.setZero();
                for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
                    for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                        sym_Qfunc(std::popcount(alpha ^ mu), std::popcount(beta ^ nu), std::popcount(alpha ^ beta ^ mu ^ nu)) += Qfunc(alpha, beta);
                    }
                }
                entropy = sym_Qfunc.contract(sym_Qfunc, contraction_indices);
                entropies(mu, nu) = entropy(0);
                if (entropy(0) > max_entropy) {
                    max_entropy = entropy(0);
                    max_mu_nu = {mu, nu};
                }
                if (entropy(0) < min_entropy) {
                    min_entropy = entropy(0);
                    min_mu_nu = {mu, nu};
                }
            }
        }
        #pragma omp critical
        {
            if (max_entropy > global_max_entropy) {
                global_max_entropy = max_entropy;
                max_displacement = max_mu_nu;
            }
            if (min_entropy < global_min_entropy) {
                global_min_entropy = min_entropy;
                min_displacement = min_mu_nu;
            }
        }
    }
}

void calc_all_displaced_symQ(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, std::string filepath) {
    #pragma omp parallel
    {
        Eigen::Tensor<double, 3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
        
        #pragma omp for
        for (unsigned int mu = 0; mu < qubitstate_size; mu++) {
            for (unsigned int nu = 0; nu < qubitstate_size; nu++) {
                sym_Qfunc.setZero();
                for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
                    for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                        sym_Qfunc(std::popcount(alpha ^ mu), std::popcount(beta ^ nu), std::popcount(alpha ^ beta ^ mu ^ nu)) += Qfunc(alpha, beta); // Should test if it is faster to make the sums in symQfunc or Qfunc
                    }
                }
                save_symQfunc(sym_Qfunc, filepath + "/" + std::to_string(mu) + "," + std::to_string(nu) + ".txt");
            }
        }
    }
}
