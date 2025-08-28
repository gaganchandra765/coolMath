#include <iostream>
#include <random>
#include <vector>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    // Parameters
    int N = 5;          // Window size
    int n_max = 10;     // Steps
    int M = 10000;      // Realizations
    double var = 1.0;   // Variance

    // Random number generator for Gaussian(0,1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, std::sqrt(var));

    // Storage: X = M × (n_max+N)
    std::vector<std::vector<double>> X(M, std::vector<double>(n_max + N));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n_max + N; j++) {
            X[i][j] = dist(gen);
        }
    }

    // Storage for Y = M × n_max
    std::vector<std::vector<double>> Y(M, std::vector<double>(n_max, 0.0));

    // Compute Moving Average process
    for (int n = 0; n < n_max; n++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += X[i][n + k];
            }
            Y[i][n] = sum / N;
        }
    }

    // Estimate mean across realizations
    std::vector<double> mean_est(n_max, 0.0);
    for (int n = 0; n < n_max; n++) {
        double sum = 0.0;
        for (int i = 0; i < M; i++) {
            sum += Y[i][n];
        }
        mean_est[n] = sum / M;
    }

    // Visualization: plot mean estimate
    std::vector<int> n_vals(n_max);
    for (int i = 0; i < n_max; i++) n_vals[i] = i+1;

    plt::figure();
    plt::plot(n_vals, mean_est, "r-o");
    plt::xlabel("n");
    plt::ylabel("Estimated Mean");
    plt::title("Mean Estimate of MA Process (N=5, M=10000)");
    plt::grid(true);
    plt::show();

    // Visualization: a few sample realizations of MA process
    plt::figure();
    for (int i = 0; i < 5; i++) { // plot 5 sample paths
        std::vector<double> sample(Y[i].begin(), Y[i].end());
        plt::plot(n_vals, sample);
    }
    plt::xlabel("n");
    plt::ylabel("Y[n]");
    plt::title("Sample MA Process Realizations");
    plt::grid(true);
    plt::show();

    return 0;
}

