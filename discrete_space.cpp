#include "discrete_space.h"
#include<cmath>
#include<complex>

static constexpr double sqrt3 = 1.73205080756887;
static constexpr std::complex<double> xi = std::complex<double>(0.5 * (sqrt3 - 1), 0.5 * (sqrt3 - 1));
static constexpr std::complex<double> xi_conj = std::complex<double>(0.5 * (sqrt3 - 1), -0.5 * (sqrt3 - 1));
static constexpr double xi_sum_inv = sqrt3; // (1 + abs(xi)^2)/(xi + xi_conj)
static constexpr std::complex<double> xi_min_inv = std::complex<double>(0.0, sqrt3); // (1 + abs(xi)^2)/(xi_conj - xi)
static constexpr double xi_norm_inv = sqrt3; // (1 + abs(xi)^2)/(1 - abs(xi)^2)

static unsigned int fact(const unsigned int &n) {
    unsigned int res = n;
    for (unsigned int j = 2; j < n; j++) {
        res *= j;
    }
    return res;
}

template <typename... Ints>
static Eigen::array<unsigned int, sizeof...(Ints)> ind_arr(Ints... index) {
    return {static_cast<unsigned int>(index)...};
}

// Executes func(int m, int n, int k) over the whole valid triples (m, n, k) of the symmetric space
template <typename Func>
void sym_space_loop(const unsigned int &n_qubits, Func func) {
    int k_max;
    for (int m = 0; m < n_qubits + 1; m++) {
        for (int n = 0; n < n_qubits + 1; n++) {
            k_max = std::min(m + n, 2*n_qubits - m - n);
            for (int k = std::abs(m - n); k < k_max; k += 2) {
                func(m, n, k);
            }
        }
    }
}

// Returns a mask with 1s on valid triples (m, n, k) inside the symmetric space and 0s everywhere else
Eigen::Tensor<double, 3> sym_space_mask(const unsigned int &n_qubits) {
    Eigen::Tensor<double, 3> mask(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    mask.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        mask(m, n, k) = 1;
    });
    return mask;
}

// Returns a tensor filled with the values R_{m,n,k}
Eigen::Tensor<double, 3> get_Rmnk(const unsigned int &n_qubits) {
    Eigen::Tensor<double, 3> R(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    R.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        R(m, n, k) = Rmnk(n_qubits, m, n, k);
    });
    return R;
}

// Returns a tensor with the P function of S•v, assuming v is normalized
Eigen::Tensor<double, 3> get_Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = Sv_Pfunc(n_qubits, qubitstate_size, v, m, n, k);
    });
    return P;
}

// Returns the P function of S•v, evaluated in (m, n, k),  where v is defined by the unit vector v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
inline double Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return std::sqrt(3.0) * (n_qubits - 2.0 * (m * std::sin(theta) * std::cos(phi) + n * std::sin(theta) * std::sin(phi) + k * std::cos(theta) ) ) / qubitstate_size;
}

// Returns a tensor with the P function of S•v, assuming v is normalized,  where v is defined by the unit vector v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
Eigen::Tensor<double, 3> get_Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi) {
    return get_Sv_Pfunc(n_qubits, qubitstate_size, {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)});
}

// Returns the P function of Sx/Sy/Sz. They are all equal, with the only difference being which variable is spanned by the single dimension. Notice that on broadcasting only valid triples (m, n, k) should be distinct from zero. If this is used to calculate averages, it is enough if the state sym Q is zero in those places
Eigen::Tensor<double, 1> get_cartesian_S_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 1> P(n_qubits + 1);
    for (unsigned int m = 0; m < P.dimension(0); m++) {
        P(m) = sqrt3 * (n_qubits - 2 * m) / qubitstate_size;
    }
    return P;
}

// Returns the P function of Sx^2/S_y^2/S_z^2. They are all equal, with the only difference being which variable is spanned by the single dimension. Notice that on broadcasting only valid triples (m, n, k) should be distinct from zero. If this is used to calculate averages, it is enough if the state sym Q is zero in those places
Eigen::Tensor<double, 1> get_cartesian_S2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 1> P(n_qubits + 1);
    for (unsigned int m = 0; m < P.dimension(0); m++) {
        P(m) = ( n_qubits + xi_sum_inv*xi_sum_inv * ((n_qubits - 2*m)*(n_qubits - 2*m) - n_qubits) ) / qubitstate_size;
    }
    return P;
}

// Returns the full symmetric P function of {Sy,Sz}
Eigen::Tensor<double, 3> get_aSySz_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = 6 * ((n_qubits - 2*k) * (n_qubits - 2*n) - (n_qubits - 2*m)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of {Sz,Sx}
Eigen::Tensor<double, 3> get_aSzSx_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = 6 * ((n_qubits - 2*m) * (n_qubits - 2*n) - n_qubits) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of {Sx,Sy}
Eigen::Tensor<double, 3> get_aSxSy_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = 6 * ((n_qubits - 2*m) * (n_qubits - 2*k) - (n_qubits - 2*n)) / qubitstate_size;
    });
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

// Calculates the expected value of S•v, where v is given as an Eigen::Vector3d and is assummed to be normalized
void ang_operator_average(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    average = 0;
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        average += Sv_Pfunc(n_qubits, qubitstate_size, v, m, n, k) * state_symQ(m, n, k);
    });
}

// Calculates the expected value of S•n, where n is given by its angles
void ang_operator_average(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    ang_operator_average(n_qubits, qubitstate_size, {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)}, state_symQ, average);
}

// Calculates the averages of all quadratic operators, where the cross products are replaced by the anticommutator. Can be optimized by reducing the size of Sx/Sy/Sz to their respective variables
void cartesian_ang_operator_averages(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &state_symQ, double &Sx, double &Sy, double &Sz, double &Sx2, double &Sy2, double &Sz2, double &SySz, double &SzSx, double &SxSy) {
    static const Eigen::Tensor<double, 3> mask = sym_space_mask(n_qubits);
    static const Eigen::Tensor<double, 1> Sv_Pfunc = get_cartesian_S_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 1> Sv2_Pfunc = get_cartesian_S2_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SySz_Pfunc = get_aSySz_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SzSx_Pfunc = get_aSzSx_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SxSy_Pfunc = get_aSxSy_Pfunc(n_qubits, qubitstate_size);
    static const auto Sx_P = mask * Sv_Pfunc.reshape(ind_arr(n_qubits + 1, 1, 1)).broadcast(ind_arr(1, n_qubits + 1, n_qubits + 1));
    static const auto Sy_P = mask * Sv_Pfunc.reshape(ind_arr(1, 1, n_qubits + 1)).broadcast(ind_arr(n_qubits + 1, n_qubits + 1, 1));
    static const auto Sz_P = mask * Sv_Pfunc.reshape(ind_arr(1, n_qubits + 1, 1)).broadcast(ind_arr(n_qubits + 1, 1, n_qubits + 1));
    static const auto Sx2_P = mask * Sv2_Pfunc.reshape(ind_arr(n_qubits + 1, 1, 1)).broadcast(ind_arr(1, n_qubits + 1, n_qubits + 1));
    static const auto Sy2_P = mask * Sv2_Pfunc.reshape(ind_arr(1, 1, n_qubits + 1)).broadcast(ind_arr(n_qubits + 1, n_qubits + 1, 1));
    static const auto Sz2_P = mask * Sv2_Pfunc.reshape(ind_arr(1, n_qubits + 1, 1)).broadcast(ind_arr(n_qubits + 1, 1, n_qubits + 1));
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
    Eigen::Tensor<double, 3> Gfunc(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    double Sx, Sy, Sz;
    Eigen::Matrix3d correlation_matrix = get_correlation_matrix(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz);
    Eigen::Vector3d x_bar = {0.5 - Sx/(2 * sqrt3 * n_qubits), 0.5 - Sy/(2 * sqrt3 * n_qubits), 0.5 - Sz/(2 * sqrt3 * n_qubits)};
    Eigen::Vector3d x;
    double coeff = (1 << (n_qubits + 1)) / ( EIGEN_PI * n_qubits * std::sqrt(EIGEN_PI * n_qubits) * correlation_matrix.determinant() );
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        x = {m/n_qubits, n/n_qubits, k/n_qubits};
        Gfunc(m, n, k) = coeff * std::exp(- n_qubits * (x - x_bar).transpose() * correlation_matrix * (x - x_bar) );
    });
    return Gfunc;
}
