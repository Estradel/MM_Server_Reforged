//g++ -o benchmark benchmark.cpp -O3 -march=native -ffast-math -std=c++17
//./benchmark

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <immintrin.h>

// Version avec optimisations manuelles
void op_cpp_optimized(const std::vector<double>& x_list,
                      const std::vector<double>& y_list,
                      std::vector<double>& result) {
    const size_t n = x_list.size();
    const double* x_ptr = x_list.data();
    const double* y_ptr = y_list.data();
    double* res_ptr = result.data();

    // Boucle déroulée + accès directs aux pointeurs
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Traitement de 4 éléments à la fois
        double x0 = x_ptr[i], y0 = y_ptr[i];
        double x1 = x_ptr[i+1], y1 = y_ptr[i+1];
        double x2 = x_ptr[i+2], y2 = y_ptr[i+2];
        double x3 = x_ptr[i+3], y3 = y_ptr[i+3];

        double xy0 = x0 * y0;
        double xy1 = x1 * y1;
        double xy2 = x2 * y2;
        double xy3 = x3 * y3;

        res_ptr[i]   = std::sqrt(x0*x0 + y0*y0) / (1.0 + std::exp(xy0));
        res_ptr[i+1] = std::sqrt(x1*x1 + y1*y1) / (1.0 + std::exp(xy1));
        res_ptr[i+2] = std::sqrt(x2*x2 + y2*y2) / (1.0 + std::exp(xy2));
        res_ptr[i+3] = std::sqrt(x3*x3 + y3*y3) / (1.0 + std::exp(xy3));
    }

    // Reste des éléments
    for (; i < n; ++i) {
        double x = x_ptr[i];
        double y = y_ptr[i];
        res_ptr[i] = std::sqrt(x*x + y*y) / (1.0 + std::exp(x * y));
    }
}

int main() {
    const size_t NUMBER = 1000000;

    // Allocation alignée pour optimisation SIMD
    std::vector<double> x(NUMBER);
    std::vector<double> y(NUMBER);
    std::vector<double> result(NUMBER);

    // Génération de données avec seed fixe pour reproductibilité
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < NUMBER; ++i) {
        x[i] = dis(gen);
        y[i] = dis(gen);
    }

    // Warm-up pour éviter les effets de cache
    op_cpp_optimized(x, y, result);

    // Benchmark sur plusieurs itérations
    constexpr int ITERATIONS = 10;
    double total_time = 0.0;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        op_cpp_optimized(x, y, result);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time += elapsed.count();
    }

    std::cout << "Temps moyen d'exécution: " << total_time / ITERATIONS << " ms\n";

    // Affichage de quelques résultats pour vérification
    std::cout << "Premiers résultats: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
