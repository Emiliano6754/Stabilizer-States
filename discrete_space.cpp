#include "discrete_space.h"
#include<cmath>
#include<complex>

// ALL LOOPS WITH M, N, K NEED TO BE REVISED TO LIMIT THEMSELVES TO THE CORRECT REGION

constexpr double sqrt3 = 1.73205080756887;
constexpr std::complex<double> xi = std::complex<double>(0.5 * (sqrt3 - 1), 0.5 * (sqrt3 - 1));
constexpr std::complex<double> xi_conj = std::complex<double>(0.5 * (sqrt3 - 1), -0.5 * (sqrt3 - 1));
constexpr double xi_sum_inv = sqrt3; // (1 + abs(xi)^2)/(xi + xi_conj)
constexpr std::complex<double> xi_min_inv = std::complex<double>(0.0, sqrt3); // (1 + abs(xi)^2)/(xi_conj - xi)
constexpr double xi_norm_inv = sqrt3; // (1 + abs(xi)^2)/(1 - abs(xi)^2)

unsigned int fact(const unsigned int &n) {
    unsigned int res = n;
    for (unsigned int j = 2; j < n; j++) {
        res *= j;
    }
    return res;
}

template <typename... Ints>
Eigen::array<unsigned int, sizeof...(Ints)> indices_array(Ints... index) {
    return {static_cast<unsigned int>(index)...};
}

inline double Rmnk(const unsigned int &n_qubits, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return fact(n_qubits) / ( fact(n_qubits - (m+n+k)/2) * fact((-m+n+k)/2) * fact((m-n+k)/2) * fact((m+n-k)/2) );
}

Eigen::Tensor<double, 3> get_Rmnk(const unsigned int &n_qubits) {
    Eigen::Tensor<double, 3> R(n_qubits+1, n_qubits+1, n_qubits+1);
    R.setZero();
    for (int m = 0; m < R.dimension(0); m++) {
        for (int n = 0; n < R.dimension(1); n++) {
            for (int k = std::abs(m - n); k < std::min(m + n, static_cast<int>(n_qubits) - m - n); k++) {
                R(m,n,k) = Rmnk(n_qubits, m, n, k);
            }
        }
    }
    return R;
}

// Returns the P function of S•v, where v is defined by the unit vector v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
inline double Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return std::sqrt(3.0) * (n_qubits - 2.0 * (m * std::sin(theta) * std::cos(phi) + n * std::sin(theta) * std::sin(phi) + k * std::cos(theta) ) ) / qubitstate_size;
}

Eigen::Tensor<double, 3> get_Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi) {
    Eigen::Tensor<double, 3> P(n_qubits+1, n_qubits+1, n_qubits+1);
    #pragma omp parallel for
    for (unsigned int m = 0; m < P.dimension(0); m++) {
        for (unsigned int n = 0; n < P.dimension(1); n++) {
            for (unsigned int k = 0; k < P.dimension(2); k++) {
                P(m,n,k) = Sv_Pfunc(n_qubits, qubitstate_size, theta, phi, m, n, k);
            }
        }
    }
    return P;
}

// Returns the P function of S•v, assuming v is normalized
inline double Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return std::sqrt(3.0) * (n_qubits - 2.0 * (m * v(0) + n * v(1) + k * v(2) ) ) / qubitstate_size;
}

Eigen::Tensor<double, 3> get_Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v) {
    Eigen::Tensor<double, 3> P(n_qubits+1, n_qubits+1, n_qubits+1);
    #pragma omp parallel for
    for (unsigned int m = 0; m < P.dimension(0); m++) {
        for (unsigned int n = 0; n < P.dimension(1); n++) {
            for (unsigned int k = 0; k < P.dimension(2); k++) {
                P(m,n,k) = Sv_Pfunc(n_qubits, qubitstate_size, v, m, n, k);
            }
        }
    }
    return P;
}

// Returns the P function of Sx/Sy/Sz. They are all equal, with the only difference being which variable is spanned by the single dimension
Eigen::Tensor<double, 1> get_cartesian_S_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 1> P(n_qubits+1);
    for (unsigned int m = 0; m < P.dimension(0); m++) {
        P(m) = sqrt3 * (n_qubits - 2 * m) / qubitstate_size;
    }
    return P;
}

// Returns the P function of Sx^2/S_y^2/S_z^2. They are all equal, with the only difference being which variable is spanned by the single dimension
Eigen::Tensor<double, 1> get_cartesian_S2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 1> P(n_qubits+1);
    for (unsigned int m = 0; m < P.dimension(0); m++) {
        P(m) = ( n_qubits + xi_sum_inv*xi_sum_inv * ((n_qubits - 2*m)*(n_qubits - 2*m) - n_qubits) ) / qubitstate_size;
    }
    return P;
}

// Returns the full symmetric P function of {Sy,Sz}
Eigen::Tensor<double, 3> get_aSySz_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits+1, n_qubits+1, n_qubits+1);
    for (unsigned int m = 0; m < P.dimension(0); m++) {
        for (unsigned int n = 0; n < P.dimension(1); n++) {
            for (unsigned int k = 0; k < P.dimension(2); k++) {
                P(m,n,k) = 6 * ((n_qubits - 2*k) * (n_qubits - 2*n) - (n_qubits - 2*m)) / qubitstate_size;
            }
        }
    }
    return P;
}

// Returns the full symmetric P function of {Sz,Sx}
Eigen::Tensor<double, 3> get_aSzSx_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits+1, n_qubits+1, n_qubits+1);
    for (unsigned int m = 0; m < P.dimension(0); m++) {
        for (unsigned int n = 0; n < P.dimension(1); n++) {
            for (unsigned int k = 0; k < P.dimension(2); k++) {
                P(m,n,k) = 6 * ((n_qubits - 2*m) * (n_qubits - 2*n) - n_qubits) / qubitstate_size;
            }
        }
    }
    return P;
}

// Returns the full symmetric P function of {Sx,Sy}
Eigen::Tensor<double, 3> get_aSxSy_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits+1, n_qubits+1, n_qubits+1);
    for (unsigned int m = 0; m < P.dimension(0); m++) {
        for (unsigned int n = 0; n < P.dimension(1); n++) {
            for (unsigned int k = 0; k < P.dimension(2); k++) {
                P(m,n,k) = 6 * ((n_qubits - 2*m) * (n_qubits - 2*k) - (n_qubits - 2*n)) / qubitstate_size;
            }
        }
    }
    return P;
}

// Calculates the average of a symmetric operator. Accepts the operator_symP as a template to allow for tensor expressions to be passed
template <typename TensorExpr>
void sym_operator_average(const TensorExpr &operator_symP, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    Eigen::Tensor<double, 0> average_result;
    Eigen::Array<Eigen::IndexPair<int>, 3, 1> contraction_indices = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(2,2)};
    average_result = state_symQ.contract(operator_symP, contraction_indices);
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

// Calculates the averages of all quadratic operators, where the cross products are replaced by the anticommutator. Can be optimized by reducing the size of Sx/Sy/Sz to their respective variables
void cartesian_ang_operator_averages(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &state_symQ, double &Sx, double &Sy, double &Sz, double &Sx2, double &Sy2, double &Sz2, double &SySz, double &SzSx, double &SxSy) {
    static const Eigen::Tensor<double, 1> Sv_Pfunc = get_cartesian_S_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 1> Sv2_Pfunc = get_cartesian_S2_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SySz_Pfunc = get_aSySz_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SzSx_Pfunc = get_aSzSx_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SxSy_Pfunc = get_aSxSy_Pfunc(n_qubits, qubitstate_size);
    auto Sx_P = Sv_Pfunc.reshape(indices_array(n_qubits+1, 1, 1)).broadcast(indices_array(1, n_qubits+1, n_qubits+1));
    auto Sy_P = Sv_Pfunc.reshape(indices_array(1, 1, n_qubits+1)).broadcast(indices_array(n_qubits+1, n_qubits+1, 1));
    auto Sz_P = Sv_Pfunc.reshape(indices_array(1, n_qubits+1, 1)).broadcast(indices_array(n_qubits+1, 1, n_qubits+1));
    auto Sx2_P = Sv2_Pfunc.reshape(indices_array(n_qubits+1, 1, 1)).broadcast(indices_array(1, n_qubits+1, n_qubits+1));
    auto Sy2_P = Sv2_Pfunc.reshape(indices_array(1, 1, n_qubits+1)).broadcast(indices_array(n_qubits+1, n_qubits+1, 1));
    auto Sz2_P = Sv2_Pfunc.reshape(indices_array(1, n_qubits+1, 1)).broadcast(indices_array(n_qubits+1, 1, n_qubits+1));
    sym_operator_average(Sx_P, state_symQ, Sx);
    sym_operator_average(Sy_P, state_symQ, Sy);
    sym_operator_average(Sz_P, state_symQ, Sz);
    sym_operator_average(Sx2_P, state_symQ, Sx2);
    sym_operator_average(Sy2_P, state_symQ, Sy2);
    sym_operator_average(Sz2_P, state_symQ, Sz2);
    sym_operator_average(SySz_Pfunc, state_symQ, SySz);
    sym_operator_average(SzSx_Pfunc, state_symQ, SzSx);
    sym_operator_average(SxSy_Pfunc, state_symQ, SxSy);
}

// Returns the correlation matrix for a particular symmetric Q function. In order to calculate values of the Gaussian envelope, the average values of Sx, Sy, Sz are returned in their inputs as well
Eigen::Matrix3d get_correlation_matrix(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ, double &Sx, double &Sy, double &Sz) {
    double Sx2, Sy2, Sz2, SySz, SzSx, SxSy;
    cartesian_ang_operator_averages(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz, Sx2, Sy2, Sz2, SySz, SzSx, SxSy);

    Eigen::Matrix3d Gamma {
        {Sx2 - Sx*Sx, SxSy - Sx*Sy, SzSx - Sz*Sx},
        {SxSy - Sx*Sy, Sy2 - Sy*Sy, SySz - Sy*Sz},
        {SzSx - Sz*Sx, SySz - Sy*Sz, Sz2 - Sz*Sz}
    };

    Eigen::Matrix3d Lambda {
        {2.0 * n_qubits, sqrt3 * Sz, sqrt3 * Sy},
        {sqrt3 * Sz, 2.0 * n_qubits, sqrt3 * Sz},
        {sqrt3 * Sy, sqrt3 * Sx, 2.0 * n_qubits}
    };

    return (Gamma + Lambda) / (6 * n_qubits);
}

// Returns the Gaussian envelope of the state SymQ
Eigen::Tensor<double, 3> get_Gfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ) {
    Eigen::Tensor<double, 3> Gfunc(n_qubits+1, n_qubits+1, n_qubits+1);
    double Sx, Sy, Sz;
    Eigen::Matrix3d correlation_matrix = get_correlation_matrix(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz);
    Eigen::Vector3d x_bar = {0.5 - Sx/(2 * sqrt3 * n_qubits), 0.5 - Sy/(2 * sqrt3 * n_qubits), 0.5 - Sz/(2 * sqrt3 * n_qubits)};
    Eigen::Vector3d x;
    double coeff = (1 << (n_qubits + 1)) / ( EIGEN_PI * n_qubits * std::sqrt(EIGEN_PI * n_qubits) * correlation_matrix.determinant() );
    for (unsigned int m = 0; m < symQ.dimension(0); m++) {
        for (unsigned int n = 0; n < symQ.dimension(1); n++) {
            for (unsigned int k = 0; k < symQ.dimension(2); k++) {
                x = {m/n_qubits, n/n_qubits, k/n_qubits};
                Gfunc(m, n, k) = coeff * std::exp(- n_qubits * (x - x_bar).transpose() * correlation_matrix * (x - x_bar) );
            }
        }
    }
    return Gfunc;
}
