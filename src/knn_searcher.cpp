#include "knn_searcher.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <immintrin.h> // Библиотека для SIMD

// --------------------------------------------------------------------
// 1. ГЕНЕРАТОР ДАННЫХ
// --------------------------------------------------------------------
KnnData KnnSearcher::generate_data(size_t num_vectors, size_t dim) {
    KnnData data;
    
    // Инициализация генератора случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    // Генерация датасета
    data.dataset.resize(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        data.dataset[i].resize(dim);
        for (size_t j = 0; j < dim; ++j) {
            data.dataset[i][j] = dis(gen);
        }
    }

    // Генерация запроса
    data.query.resize(dim);
    for (size_t j = 0; j < dim; ++j) {
        data.query[j] = dis(gen);
    }

    return data;
}

// --------------------------------------------------------------------
// 2. НАИВНАЯ РЕАЛИЗАЦИЯ (Standard C++)
// --------------------------------------------------------------------

float KnnSearcher::get_euclidean_distance_naive(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<int> KnnSearcher::find_naive(
    const std::vector<std::vector<float>>& dataset, 
    const std::vector<float>& query, 
    int k
) {
    std::vector<std::pair<float, int>> distances;
    distances.reserve(dataset.size());

    for (size_t i = 0; i < dataset.size(); ++i) {
        float dist = get_euclidean_distance_naive(dataset[i], query);
        distances.push_back({dist, (int)i});
    }

    // Наивная полная сортировка
    std::sort(distances.begin(), distances.end());

    std::vector<int> result;
    result.reserve(k);
    for (int i = 0; i < k && i < (int)distances.size(); ++i) {
        result.push_back(distances[i].second);
    }
    return result;
}

// --------------------------------------------------------------------
// 3. SIMD РЕАЛИЗАЦИЯ (AVX-512)
// --------------------------------------------------------------------

float KnnSearcher::get_euclidean_distance_avx512(const float* a, const float* b, size_t size) {
    __m512 sum_vec = _mm512_setzero_ps();
    size_t i = 0;

    // Основной цикл по 16 float (512 бит)
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }

    float total_sum = _mm512_reduce_add_ps(sum_vec);

    // Обработка остатка через маски (AVX-512 feature)
    if (i < size) {
        unsigned int remaining = size - i;
        // Маска для оставшихся элементов (1 бит на элемент)
        __mmask16 mask = (__mmask16)((1U << remaining) - 1);

        __m512 va = _mm512_maskz_loadu_ps(mask, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(mask, b + i);
        
        __m512 diff = _mm512_sub_ps(va, vb);
        __m512 sq_diff = _mm512_mul_ps(diff, diff);
        
        total_sum += _mm512_reduce_add_ps(sq_diff);
    }

    return std::sqrt(total_sum);
}

std::vector<int> KnnSearcher::find_simd(
    const std::vector<std::vector<float>>& dataset, 
    const std::vector<float>& query, 
    int k
) {
    std::vector<std::pair<float, int>> distances;
    distances.reserve(dataset.size());

    // Указатель на данные запроса и размер берем один раз
    const float* query_ptr = query.data();
    size_t dim = query.size();

    for (size_t i = 0; i < dataset.size(); ++i) {
        // Передаем указатели на данные векторов
        float dist = get_euclidean_distance_avx512(dataset[i].data(), query_ptr, dim);
        distances.push_back({dist, (int)i});
    }

    // Сортировка идентична наивной
    std::sort(distances.begin(), distances.end());

    std::vector<int> result;
    result.reserve(k);
    for (int i = 0; i < k && i < (int)distances.size(); ++i) {
        result.push_back(distances[i].second);
    }
    return result;
}
