#include<iostream>
#include<Eigen/Dense>
#include<unsupported/Eigen/CXX11/Tensor>
#include<kravchuk.h>
#include<sym_space.h>

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

// Returns Binom(N,k)
static unsigned int binom(const unsigned int &N, const unsigned int &k) {
    unsigned int res = 1;
    for (unsigned int j = N; j > N - k; j--) {
        res *= j;
    }
    res /= fact(k);
    return res;
}

// Returns a tensor of doubles filled with all binomials (N,k) from k=0 to k=N
static Eigen::Tensor<double, 1> binom(const unsigned int &N) {
    Eigen::Tensor<double, 1> res(N+1);
    for (unsigned int k = 0; k < N+1; k++) {
        res(k) = binom(N, k);
    }
    return res;
}

// Returns a tensor of doubles filled with all binomials (N,k) from k=0 to k=N, squared
static Eigen::Tensor<double, 1> binom2(const unsigned int &N) {
    Eigen::Tensor<double, 1> res(N+1);
    for (unsigned int k = 0; k < N+1; k++) {
        res(k) = binom(N, k);
        res(k) *= res(k);
    }
    return res;
}

static void get_unsignedint(unsigned int &parsed_input) {
    std::string input = "";
    std::cin >> input;
    try {
        unsigned long u = std::stoul(input);
        if (u > std::numeric_limits<unsigned int>::max())
            throw std::out_of_range(input);

        parsed_input = u;
    } catch (const std::invalid_argument& e) {
        std::cout << "Input could not be parsed: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Input out of range: " << e.what() << std::endl;
    }
}

void check_Rmnk() {
    unsigned int max_n;
    std::cout << "Enter the maximum number of qubits to check" << std::endl;
    get_unsignedint(max_n);
    Eigen::Tensor<double, 3> exact_Rmnk, Kravchuk_Rmnk;
    Eigen::array<Eigen::Index, 3> dimensions;
    Eigen::Tensor<double, 1> binoms(max_n), binoms2(max_n);
    double norm_fact = 1;
    Eigen::Tensor<double, 0> squared_difference;
    for (int n_qubits = 1; n_qubits <= max_n; n_qubits++) {
        exact_Rmnk = get_Rmnk(n_qubits);
        dimensions = Eigen::array<Eigen::Index, 3>{n_qubits + 1, n_qubits + 1, n_qubits + 1};
        Kravchuk_Rmnk.resize(dimensions);
        Kravchuk_Rmnk.setZero();
        {
            std::vector<polynomial> Kravchuks = get_Kravchuk_pols(n_qubits, n_qubits);
            binoms = binom(n_qubits);
            binoms2 = binom2(n_qubits);
            norm_fact = 1.0 / (1 << n_qubits);
            sym_space_loop(n_qubits, 
            [&] (int const &m, int const &n, int const &k) {
                    for (int l = 0; l <= n_qubits; l++) {
                            Kravchuk_Rmnk(m, n, k) += Kravchuks[l](m) * Kravchuks[l](n) * Kravchuks[l](k) / binoms2[l];
                        }
                        Kravchuk_Rmnk(m, n, k) *= norm_fact * binoms[m] * binoms[n] * binoms[k];
                    }
                );
            squared_difference = (exact_Rmnk - Kravchuk_Rmnk).square().sum();
            std::cout << "N = " << n_qubits << ". Diff = " << squared_difference(0) << std::endl;
        }
    }
}

int main() {
    check_Rmnk();

    return 0;
}