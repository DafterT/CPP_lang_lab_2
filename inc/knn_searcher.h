#pragma once

#include <vector>
#include <utility>

// Структура для удобного возврата сгенерированных данных
struct KnnData {
    std::vector<std::vector<float>> dataset;
    std::vector<float> query;
};

class KnnSearcher {
public:
    // 1. Генератор данных
    // Создает dataset размером [num_vectors x dim] и один query вектор [dim]
    static KnnData generate_data(size_t num_vectors, size_t dim);

    // 2. Наивная реализация на чистом C++
    static std::vector<int> find_naive(
        const std::vector<std::vector<float>>& dataset, 
        const std::vector<float>& query, 
        int k
    );

    // 3. Реализация с использованием AVX-512
    static std::vector<int> find_simd(
        const std::vector<std::vector<float>>& dataset, 
        const std::vector<float>& query, 
        int k
    );

    static std::vector<int> find_simd_soa(
        const std::vector<std::vector<float>>& dataset,
        const std::vector<float>& query,
        int k
    );

private:
    // Вспомогательные функции для подсчета расстояний
    static float get_euclidean_distance_naive(const std::vector<float>& a, const std::vector<float>& b);
    static float get_euclidean_distance_avx512(const float* a, const float* b, size_t size);
};
