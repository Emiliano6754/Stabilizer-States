#include "displaced_Qfunc.h"
#include<Eigen/Dense>
#include<iostream>
#include<filesystem>
#include<fstream>
#include<tuple>
#include "discrete_space.h"

template <typename ThreadInitFunc, typename WorkFunc, typename CriticalFunc>
static void for_all_displaced_symQ(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc, ThreadInitFunc init_thread_vars, WorkFunc operate_symQ, CriticalFunc critical_func)
{
    #pragma omp parallel
    {
        auto thread_vars = init_thread_vars();

        Eigen::Tensor<double, 3> sym_Qfunc(n_qubits + 1, n_qubits + 1, n_qubits + 1);

        #pragma omp for collapse(2)
        for (unsigned int mu = 0; mu < qubitstate_size; ++mu) {
            for (unsigned int nu = 0; nu < qubitstate_size; ++nu) {
                sym_Qfunc.setZero();
                for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
                    for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                        sym_Qfunc(std::popcount(alpha ^ mu), std::popcount(beta ^ nu), std::popcount(alpha ^ beta ^ mu ^ nu)) += Qfunc(alpha, beta); // Should test if it is faster to make the sums in symQfunc or Qfunc
                    }
                }
                operate_symQ(sym_Qfunc, thread_vars, mu, nu);
            }
        }
        #pragma omp critical
        {
            critical_func(thread_vars);
        }
    }
}

void minmax_displaced_distance(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc, const Eigen::Tensor<double, 3> &symQ, double &max_distance, double &min_distance, std::tuple<unsigned int, unsigned int> &max_displacement, std::tuple<unsigned int, unsigned int> &min_displacement) {
    struct thread_variables {
        Eigen::Tensor<double, 0> current_distance;
        double local_max_distance = 0;
        double local_min_distance = 1e10; // Should check what the maximum distance can be
        std::tuple<unsigned int, unsigned int> local_max_displacement = {0,0};
        std::tuple<unsigned int, unsigned int> local_min_displacement = {0,0};
    };

    Eigen::Tensor<double, 3> Gfunc = get_Gfunc(n_qubits, qubitstate_size, symQ);

    max_distance = 0;
    min_distance = 2000;
    
    for_all_displaced_symQ(n_qubits, qubitstate_size, Qfunc,
    [&]()->thread_variables {
        thread_variables vars;
        return vars;
    },
    [&](const Eigen::Tensor<double, 3> &sym_Qfunc, thread_variables &thread_variables, const unsigned int &mu, const unsigned int &nu) {
        thread_variables.current_distance = (Gfunc - sym_Qfunc).square().sum();
        if (thread_variables.current_distance(0) > thread_variables.local_max_distance) {
            thread_variables.local_max_distance = thread_variables.current_distance(0);
            thread_variables.local_max_displacement = {mu, nu};
        }
        if (thread_variables.current_distance(0) < thread_variables.local_min_distance) {
            thread_variables.local_min_distance = thread_variables.current_distance(0);
            thread_variables.local_min_displacement = {mu, nu};
        }
    },
    [&](const thread_variables &thread_variables) {
        if (thread_variables.local_max_distance > max_distance) {
            max_distance = thread_variables.local_max_distance;
            max_displacement = thread_variables.local_max_displacement;
        }
        if (thread_variables.local_min_distance < min_distance) {
            min_distance = thread_variables.local_min_distance;
            min_displacement = thread_variables.local_min_displacement;
        }
    });
}

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
void calc_full_displaced_maxmin_entropy(const Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &entropies, std::tuple<unsigned int, unsigned int> &max_displacement, std::tuple<unsigned int, unsigned int> &min_displacement) {
    Eigen::Array<Eigen::IndexPair<int>, 3, 1> contraction_indices = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(2,2)};
    double global_max_entropy = 0;
    double global_min_entropy = 1;
    #pragma omp parallel
    {
        Eigen::Tensor<double, 3> sym_Qfunc(n_qubits+1,n_qubits+1,n_qubits+1);
        Eigen::Tensor<double, 0> entropy; // Necessary to force evaluation of the tensor expressions

        // Let each thread calculate a minimum/maximum and then start comparing their values as they finish
        std::tuple<unsigned int, unsigned int> max_mu_nu = {0,0};
        std::tuple<unsigned int, unsigned int> min_mu_nu = {0,0};
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
