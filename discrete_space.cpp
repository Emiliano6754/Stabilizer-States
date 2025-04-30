#include "discrete_space.h"
#include<cmath>

// Returns the P function of S•v, where v is defined by the unit vector v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
inline double Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return std::sqrt(3.0) * (n_qubits - 2.0 * (m * std::sin(theta) * std::cos(phi) + n * std::sin(theta) * std::sin(phi) + k * std::cos(theta) ) ) / qubitstate_size;
}

// Returns the P function of S•v, assuming v is normalized
inline double Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return std::sqrt(3.0) * (n_qubits - 2.0 * (m * v(0) + n * v(1) + k * v(2) ) ) / qubitstate_size;
}

void sym_operator_average(const Eigen::Tensor<double, 3> &operator_symP, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    Eigen::Tensor<double, 0> average_result;
    Eigen::Array<Eigen::IndexPair<int>, 3, 1> contraction_indices = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(2,2)};
    average_result = operator_symP.contract(state_symQ, contraction_indices);
    average = average_result(0);
}

// Calculates the expected value of S•n, where n is given by its angles. The constants multiplying the P function could be factorized for better perfomance. Why calculate the components on each iteration when they can be calculated once and use the other function?
void ang_operator_average(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    average = 0;
    #pragma omp parallel for collapse(3) reduction(+:average)
    for (unsigned int m = 0; m < state_symQ.dimension(0); m++) {
        for (unsigned int n = 0; n < state_symQ.dimension(1); n++) {
            for (unsigned int k = 0; k < state_symQ.dimension(2); k++) {
                average += Sv_Pfunc(n_qubits, qubitstate_size, theta, phi, m, n, k) * state_symQ(m, n, k);
            }
        }
    }
}

// Calculates the expected value of S•v, where v is given as an Eigen::Vector3d and is assummed to be normalized. The constants multiplying the P function could be factorized for better perfomance
void ang_operator_average(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    average = 0;
    #pragma omp parallel for collapse(3) reduction(+:average)
    for (unsigned int m = 0; m < state_symQ.dimension(0); m++) {
        for (unsigned int n = 0; n < state_symQ.dimension(1); n++) {
            for (unsigned int k = 0; k < state_symQ.dimension(2); k++) {
                average += Sv_Pfunc(n_qubits, qubitstate_size, v, m, n, k) * state_symQ(m, n, k);
            }
        }
    }
}

void cartesian_ang_operator_averages(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &state_symQ, double &S_x, double &S_y, double &S_z) {
    S_x = 0;
    S_y = 0;
    S_z = 0;
    #pragma omp parallel for collapse(3) reduction(+:average)
    for (unsigned int m = 0; m < state_symQ.dimension(0); m++) {
        for (unsigned int n = 0; n < state_symQ.dimension(1); n++) {
            for (unsigned int k = 0; k < state_symQ.dimension(2); k++) {
                S_x += Sv_Pfunc(n_qubits, qubitstate_size, Eigen::Vector3d(1,0,0), m, n, k) * state_symQ(m, n, k);
                S_y += Sv_Pfunc(n_qubits, qubitstate_size, Eigen::Vector3d(0,1,0), m, n, k) * state_symQ(m, n, k);
                S_z += Sv_Pfunc(n_qubits, qubitstate_size, Eigen::Vector3d(0,0,1), m, n, k) * state_symQ(m, n, k);
            }
        }
    }
}

// Missing the second moments
Eigen::Matrix3d get_correlation_matrix(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ) {
    double S_x, S_y, S_z;
    double sqrt3 = std::sqrt(3.0);
    cartesian_ang_operator_averages(n_qubits, qubitstate_size, symQ, S_x, S_y, S_z);

    Eigen::Matrix3d Lambda {
        {2.0 * n_qubits, sqrt3 * S_z, sqrt3 * S_y},
        {sqrt3 * S_z, 2.0 * n_qubits, sqrt3 * S_z},
        {sqrt3 * S_y, sqrt3 * S_x, 2.0 * n_qubits}
    };
}

template <typename T>
double Gfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ, const T &m, const T &n, const T &k) {
    static double coeff = (1 << (n_qubits + 1)) / ( M_PI * n_qubits * std::sqrt(M_PI * n_qubits) );
    static Eigen::Matrix3d correlation_matrix = get_correlation_matrix();
}
