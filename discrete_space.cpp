#include "discrete_space.h"
#include<cmath>
#include<complex>


#include<iostream>

static constexpr double sqrt3 = 1.73205080756887;
static constexpr std::complex<double> xi = std::complex<double>(0.5 * (sqrt3 - 1), 0.5 * (sqrt3 - 1));
static constexpr std::complex<double> xi_conj = std::complex<double>(0.5 * (sqrt3 - 1), -0.5 * (sqrt3 - 1));
static constexpr double xi_sum_inv = sqrt3; // (1 + abs(xi)^2)/(xi + xi_conj)
static constexpr std::complex<double> xi_min_inv = std::complex<double>(0.0, sqrt3); // (1 + abs(xi)^2)/(xi_conj - xi)
static constexpr double xi_norm_inv = sqrt3; // (1 + abs(xi)^2)/(1 - abs(xi)^2)

static unsigned int fact(const unsigned int &n) {
    if (n == 0) { 
        return 1;
    }
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
            k_max = std::min(m + n, 2*static_cast<int>(n_qubits) - m - n) + 1;
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

// Returns the value of R_{m,n,k}
inline double Rmnk(const unsigned int &n_qubits, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return static_cast<double>(fact(n_qubits)) / static_cast<double>( fact(n_qubits - (m+n+k)/2) * fact((-m+n+k)/2) * fact((m-n+k)/2) * fact((m+n-k)/2) );
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

// Returns a tensor with the P function of S•v, assuming v is normalized,  where v is defined by the unit vector v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
Eigen::Tensor<double, 3> get_Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi) {
    return get_Sv_Pfunc(n_qubits, qubitstate_size, {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)});
}

// Returns the P function of Sx/Sy/Sz. They are all equal, with the only difference being which variable is spanned by the single dimension. Notice that on broadcasting only valid triples (m, n, k) should be distinct from zero. If this is used to calculate averages, it is enough if the state sym Q is zero in those places
Eigen::Tensor<double, 1> get_cartesian_S_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 1> P(n_qubits + 1);
    for (int m = 0; m < P.dimension(0); m++) {
        P(m) = sqrt3 * (static_cast<double>(n_qubits) - 2 * m) / qubitstate_size;
    }
    return P;
}

// Returns the P function of Sx^2/S_y^2/S_z^2. They are all equal, with the only difference being which variable is spanned by the single dimension. Notice that on broadcasting only valid triples (m, n, k) should be distinct from zero. If this is used to calculate averages, it is enough if the state sym Q is zero in those places
Eigen::Tensor<double, 1> get_cartesian_S2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 1> P(n_qubits + 1);
    for (int m = 0; m < P.dimension(0); m++) {
        P(m) = ( n_qubits + xi_sum_inv*xi_sum_inv * ((static_cast<double>(n_qubits) - 2.0 * m)*(static_cast<double>(n_qubits) - 2.0 * m) - n_qubits) ) / qubitstate_size;
    }
    return P;
}

// Returns the full symmetric P function of Sx
Eigen::Tensor<double, 3> get_Sx_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = sqrt3 * (static_cast<double>(n_qubits) - 2.0 * m) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sy
Eigen::Tensor<double, 3> get_Sy_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = sqrt3 * (static_cast<double>(n_qubits) - 2.0 * k) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sz
Eigen::Tensor<double, 3> get_Sz_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = sqrt3 * (static_cast<double>(n_qubits) - 2.0 * n) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sx^2
Eigen::Tensor<double, 3> get_Sx2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = (n_qubits + 3 * ((static_cast<double>(n_qubits) - 2.0 * m)*(static_cast<double>(n_qubits) - 2.0 * m) - n_qubits)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sy^2
Eigen::Tensor<double, 3> get_Sy2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = (n_qubits + 3 * ((static_cast<double>(n_qubits) - 2.0 * k)*(static_cast<double>(n_qubits) - 2.0 * k) - n_qubits)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sz^2
Eigen::Tensor<double, 3> get_Sz2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = (n_qubits + 3 * ((static_cast<double>(n_qubits) - 2.0 * n)*(static_cast<double>(n_qubits) - 2.0 * n) - n_qubits)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of {Sy,Sz}/2
Eigen::Tensor<double, 3> get_aSySz_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = 3.0 * ((static_cast<double>(n_qubits) - 2.0 * k) * (static_cast<double>(n_qubits) - 2.0 * n) - (static_cast<double>(n_qubits) - 2.0 * m)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of {Sz,Sx}/2
Eigen::Tensor<double, 3> get_aSzSx_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = 3.0 * ((static_cast<double>(n_qubits) - 2.0 * m) * (static_cast<double>(n_qubits) - 2.0 * n) - n_qubits) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of {Sx,Sy}/2
Eigen::Tensor<double, 3> get_aSxSy_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = 3.0 * ((static_cast<double>(n_qubits) - 2.0 * m) * (static_cast<double>(n_qubits) - 2.0 * k) - (static_cast<double>(n_qubits) - 2.0 * n)) / qubitstate_size;
    });
    return P;
}

// Calculates the average of a symmetric operator. Accepts the operator_symP as a template to allow for tensor expressions to be passed
template <typename TensorExpr>
void sym_operator_average(const TensorExpr &operator_symP, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    Eigen::Tensor<double, 0> average_result;
    // Eigen::Array<Eigen::IndexPair<int>, 3, 1> contraction_indices = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(2,2)};
    // average_result = state_symQ.contract(operator_symP, contraction_indices);
    average_result = (state_symQ * operator_symP).sum();
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
    static const Eigen::Tensor<double, 3> Sx_Pfunc = get_Sx_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sy_Pfunc = get_Sy_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sz_Pfunc = get_Sz_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sx2_Pfunc = get_Sx2_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sy2_Pfunc = get_Sy2_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sz2_Pfunc = get_Sz2_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SySz_Pfunc = get_aSySz_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SzSx_Pfunc = get_aSzSx_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SxSy_Pfunc = get_aSxSy_Pfunc(n_qubits, qubitstate_size);
    sym_operator_average(Sx_Pfunc, state_symQ, Sx);
    sym_operator_average(Sy_Pfunc, state_symQ, Sy);
    sym_operator_average(Sz_Pfunc, state_symQ, Sz);
    sym_operator_average(Sx2_Pfunc, state_symQ, Sx2);
    sym_operator_average(Sy2_Pfunc, state_symQ, Sy2);
    sym_operator_average(Sz2_Pfunc, state_symQ, Sz2);
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
        {sqrt3 * Sz, 2.0 * n_qubits, sqrt3 * Sx},
        {sqrt3 * Sy, sqrt3 * Sx, 2.0 * n_qubits}
    };
    return (Gamma + Lambda) / (6.0 * n_qubits);
}

// Returns the Gaussian envelope of the state SymQ. NEEDS CORRECTION TO MIMIC THE FUNCTION BELOW
Eigen::Tensor<double, 3> get_Gfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ) {
    Eigen::Tensor<double, 3> Gfunc(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    double Sx, Sy, Sz;
    Eigen::Matrix3d correlation_matrix = get_correlation_matrix(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz);
    Eigen::Vector3d x_bar = {0.5 - Sx/(2 * sqrt3 * n_qubits), 0.5 - Sy/(2 * sqrt3 * n_qubits), 0.5 - Sz/(2 * sqrt3 * n_qubits)};
    Eigen::Vector3d x;
    double coeff = (1 << (n_qubits + 1)) / ( EIGEN_PI * n_qubits * std::sqrt(EIGEN_PI * n_qubits) * correlation_matrix.determinant() );
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        x = {static_cast<double>(m)/n_qubits, static_cast<double>(n)/n_qubits, static_cast<double>(k)/n_qubits};
        Gfunc(m, n, k) = coeff * std::exp(- static_cast<double>(n_qubits) * (x - x_bar).transpose() * correlation_matrix * (x - x_bar) );
    });
    return Gfunc;
}

// Returns the Gaussian envelope of the state SymQ in Gfunc to avoid copying
void get_Gfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ, Eigen::Tensor<double, 3> &Gfunc) {
    double Sx, Sy, Sz;
    Eigen::Matrix3d correlation_matrix = get_correlation_matrix(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz);
    Eigen::Vector3d x_bar = {0.5 - Sx/(2 * sqrt3 * n_qubits), 0.5 - Sy/(2 * sqrt3 * n_qubits), 0.5 - Sz/(2 * sqrt3 * n_qubits)};
    Eigen::Vector3d x;
    double coeff = (1 << (n_qubits + 1)) / ( EIGEN_PI * n_qubits * std::sqrt(EIGEN_PI * n_qubits * correlation_matrix.determinant()) );
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        x = {static_cast<double>(m)/n_qubits, static_cast<double>(n)/n_qubits, static_cast<double>(k)/n_qubits};
        Gfunc(m, n, k) = coeff * std::exp(- static_cast<double>(n_qubits) * (x - x_bar).transpose() * correlation_matrix * (x - x_bar) );
    });
    // std::cout << Gfunc.sum() << std::endl;
    std::cout << coeff << std::endl;
}

// To be implemented
Eigen::Tensor<double, 3> get_symQ(const unsigned int &n_qubits, const unsigned int &qubitstate_size, Eigen::MatrixXd &Qfunc);