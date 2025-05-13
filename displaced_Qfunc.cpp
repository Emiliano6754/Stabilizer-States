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
        unsigned int alpha_p, beta_p;

        #pragma omp for collapse(2) nowait
        for (unsigned int mu = 0; mu < qubitstate_size; ++mu) {
            for (unsigned int nu = 0; nu < qubitstate_size; ++nu) {
                sym_Qfunc.setZero();
                for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
                    alpha_p = alpha ^ mu;
                    for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                        beta_p = beta ^ nu;
                        sym_Qfunc(std::popcount(alpha_p), std::popcount(beta_p), std::popcount(alpha_p ^ beta_p)) += Qfunc(alpha, beta); // Should test if it is faster to make the sums in symQfunc or Qfunc
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

template <typename ThreadInitFunc, typename WorkFunc, typename CriticalFunc>
static void for_all_lClifford_symQ(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc, ThreadInitFunc init_thread_vars, WorkFunc operate_symQ, CriticalFunc critical_func)
{
    #pragma omp parallel
    {
        auto thread_vars = init_thread_vars();

        Eigen::Tensor<double, 3> sym_Qfunc(n_qubits + 1, n_qubits + 1, n_qubits + 1);
        unsigned int alpha_p, beta_p;
        unsigned int alpha_pp, beta_pp;

        #pragma omp for collapse(3) nowait
        for (unsigned int mu = 0; mu < qubitstate_size; ++mu) {
            for (unsigned int nu = 0; nu < qubitstate_size; ++nu) {
                for (unsigned int gamma = 0; gamma < qubitstate_size; gamma++) {
                    sym_Qfunc.setZero();
                    for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
                        alpha_p = alpha ^ mu;
                        for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                                beta_p = beta ^ nu;
                                alpha_pp = alpha_p & (~gamma) | (beta_p & gamma); 
                                beta_pp = beta_p & (~gamma) | (alpha_p & gamma); 
                                sym_Qfunc(std::popcount(alpha_pp), std::popcount(beta_pp), std::popcount(alpha_pp ^ beta_pp)) += Qfunc(alpha, beta); // Should test if it is faster to make the sums in symQfunc or Qfunc
                            }
                    }
                    operate_symQ(sym_Qfunc, thread_vars, mu, nu, gamma);
                }
            }
        }
        #pragma omp critical
        {
            critical_func(thread_vars);
        }
    }
}

// Maximizes the Hellinger distance of a Q function to its gaussian envelope and the Rmnk distribution, after displacements Z^mu X^nu. Returns both maximum distances, in the order {distance to G, distance to R}, together with the corresponding parameters of the optimizers in the order {mu, nu}, in the same order of distance to G then to R
void max_displaced_distances(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc, const Eigen::Tensor<double, 3> &symQ, double &max_distance_G, double &max_distance_R, std::tuple<unsigned int, unsigned int> &max_displacement_G, std::tuple<unsigned int, unsigned int> &max_displacement_R) {
    struct thread_variables {
        Eigen::Tensor<double, 0> current_distance;
        Eigen::Tensor<double, 3> Gfunc;
        double local_max_G = 0;
        double local_max_R = 0;
        std::tuple<unsigned int, unsigned int> local_max_G_displacement = {0, 0};
        std::tuple<unsigned int, unsigned int> local_max_R_displacement = {0, 0};
    };

    Eigen::Tensor<double, 3> Rmnk = get_Rmnk(n_qubits);

    max_distance_G = 0;
    max_distance_R = 0;

    double norm_const = qubitstate_size * std::sqrt(qubitstate_size);
    
    for_all_displaced_symQ(n_qubits, qubitstate_size, Qfunc,
    [&]()->thread_variables {
        thread_variables vars;
        vars.Gfunc = Eigen::Tensor<double, 3>(n_qubits + 1, n_qubits + 1, n_qubits + 1);
        vars.Gfunc.setZero();
        return vars;
    },
    [&](const Eigen::Tensor<double, 3> &sym_Qfunc, thread_variables &thread_variables, const unsigned int &mu, const unsigned int &nu) {
        // Calculate the new Gaussian envelope G
        get_Gfunc(n_qubits, qubitstate_size, sym_Qfunc, thread_variables.Gfunc);
        // Calculate distance to G
        thread_variables.current_distance = (thread_variables.Gfunc * sym_Qfunc).sqrt().sum();
        if ( 1.0 - ( thread_variables.current_distance(0) / static_cast<double>(qubitstate_size) ) > thread_variables.local_max_G) {
            thread_variables.local_max_G = 1.0 - ( thread_variables.current_distance(0) / static_cast<double>(qubitstate_size) );
            thread_variables.local_max_G_displacement = {mu, nu};
        }
        // Calculate distance to R
        thread_variables.current_distance = (Rmnk * sym_Qfunc).sqrt().sum();
        // The actual identity Q function is given by Rmnk/2^N, thus divide the distance by an extra 2^N/2
        if ( 1.0 - ( thread_variables.current_distance(0) / norm_const ) > thread_variables.local_max_R) {
            thread_variables.local_max_R = 1.0 - ( thread_variables.current_distance(0) / norm_const );
            thread_variables.local_max_R_displacement = {mu, nu};
        }
    },
    [&](const thread_variables &thread_variables) {
        // Compare distances to thread local optimizers and maximize globally
        if (thread_variables.local_max_G > max_distance_G) {
            max_distance_G = thread_variables.local_max_G;
            max_displacement_G = thread_variables.local_max_G_displacement;
        }
        if (thread_variables.local_max_R > max_distance_R) {
            max_distance_R = thread_variables.local_max_R;
            max_displacement_R = thread_variables.local_max_R_displacement;
        }
    });
}

// Maximizes the Hellinger distance of a Q function to its gaussian envelope and the Rmnk distribution, after displacements and Hadamard gates H^gamma Z^mu X^nu (it is assumed that the displacement is applied first). Returns both maximum distances, in the order {distance to G, distance to R}, together with the corresponding parameters of the optimizers in the order {mu, nu, gamma}, in the same order of distance to G then to R
void max_lClifford_distances(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc, const Eigen::Tensor<double, 3> &symQ, double &max_distance_G, double &max_distance_R, std::tuple<unsigned int, unsigned int, unsigned int> &max_Clifford_G, std::tuple<unsigned int, unsigned int, unsigned int> &max_Clifford_R) {
    struct thread_variables {
        Eigen::Tensor<double, 0> current_distance;
        Eigen::Tensor<double, 3> Gfunc;
        double local_max_G = 0;
        double local_max_R = 0;
        std::tuple<unsigned int, unsigned int, unsigned int> local_max_G_Clifford = {0, 0, 0};
        std::tuple<unsigned int, unsigned int, unsigned int> local_max_R_Clifford = {0, 0, 0};
    };

    Eigen::Tensor<double, 3> Rmnk = get_Rmnk(n_qubits);

    max_distance_G = 0;
    max_distance_R = 0;

    double norm_const = qubitstate_size * std::sqrt(qubitstate_size);
    
    for_all_lClifford_symQ(n_qubits, qubitstate_size, Qfunc,
    [&]()->thread_variables {
        thread_variables vars;
        vars.Gfunc = Eigen::Tensor<double, 3>(n_qubits + 1, n_qubits + 1, n_qubits + 1);
        vars.Gfunc.setZero();
        return vars;
    },
    [&](const Eigen::Tensor<double, 3> &sym_Qfunc, thread_variables &thread_variables, const unsigned int &mu, const unsigned int &nu, const unsigned int &gamma) {
        // Calculate the new Gaussian envelope G
        get_Gfunc(n_qubits, qubitstate_size, sym_Qfunc, thread_variables.Gfunc);
        // Calculate distance to G
        thread_variables.current_distance = (thread_variables.Gfunc * sym_Qfunc).sqrt().sum();
        if ( 1.0 - ( thread_variables.current_distance(0) / static_cast<double>(qubitstate_size) ) > thread_variables.local_max_G) {
            thread_variables.local_max_G = 1.0 - ( thread_variables.current_distance(0) / static_cast<double>(qubitstate_size) );
            thread_variables.local_max_G_Clifford = {mu, nu, gamma};
        }
        // Calculate distance to R
        thread_variables.current_distance = (Rmnk * sym_Qfunc).sqrt().sum();
        // The actual identity Q function is given by Rmnk/2^N, thus divide the distance by an extra 2^N/2
        if ( 1.0 - ( thread_variables.current_distance(0) / norm_const ) > thread_variables.local_max_R) {
            thread_variables.local_max_R = 1.0 - ( thread_variables.current_distance(0) / norm_const );
            thread_variables.local_max_R_Clifford = {mu, nu, gamma};
        }
    },
    [&](const thread_variables &thread_variables) {
        // Compare distances to thread local optimizers and maximize globally
        if (thread_variables.local_max_G > max_distance_G) {
            max_distance_G = thread_variables.local_max_G;
            max_Clifford_G = thread_variables.local_max_G_Clifford;
        }
        if (thread_variables.local_max_R > max_distance_R) {
            max_distance_R = thread_variables.local_max_R;
            max_Clifford_R = thread_variables.local_max_R_Clifford;
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
