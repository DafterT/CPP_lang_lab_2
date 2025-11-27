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

// --------------------------------------------------------------------
// 4. SIMD РЕАЛИЗАЦИЯ (AVX-512) сразу несколько элементов в массиве
// --------------------------------------------------------------------

inline __m512 kernel_16_vectors(
    const float* const* ptrs, 
    const float* qptr,        
    size_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    alignas(64) float slice[16];

    for (size_t d = 0; d < dim; ++d) {
        for (int lane = 0; lane < 16; ++lane) {
            slice[lane] = ptrs[lane][d];
        }

        __m512 v_dim = _mm512_load_ps(slice);
        
        __m512 v_query = _mm512_set1_ps(qptr[d]);
        __m512 diff = _mm512_sub_ps(v_dim, v_query);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    return _mm512_sqrt_ps(sum);
}

std::vector<int> KnnSearcher::find_simd_soa(
    const std::vector<std::vector<float>>& dataset,
    const std::vector<float>& query,
    int k
) {
    const size_t N = dataset.size();
    const size_t dim = query.size();
    std::vector<int> result;
    
    if (N == 0 || dim == 0 || k <= 0) return result;

    const float* qptr = query.data();
    
    std::vector<std::pair<float, int>> distances;
    distances.reserve(N);

    alignas(64) float block_dists[16];
    const float* ptrs[16];

    size_t i = 0;
    // Главный цикл: шаг 16 векторов
    for (; i + 16 <= N; i += 16) {
        for (int j = 0; j < 16; ++j) {
            ptrs[j] = dataset[i + j].data();
        }

        __m512 res_vec = kernel_16_vectors(ptrs, qptr, dim);
        _mm512_store_ps(block_dists, res_vec);

        for (int j = 0; j < 16; ++j) {
            distances.emplace_back(block_dists[j], (int)(i + j));
        }
    }

    // Обработка остатка (tail case)
    if (i < N) {
        size_t remaining = N - i;
        for (size_t j = 0; j < remaining; ++j) {
            ptrs[j] = dataset[i + j].data();
        }
        for (size_t j = remaining; j < 16; ++j) {
            ptrs[j] = ptrs[0]; 
        }

        __m512 res_vec = kernel_16_vectors(ptrs, qptr, dim);
        _mm512_store_ps(block_dists, res_vec);

        for (size_t j = 0; j < remaining; ++j) {
            distances.emplace_back(block_dists[j], (int)(i + j));
        }
    }

    std::sort(distances.begin(), distances.end());

    result.reserve(k);
    for (int j = 0; j < k && j < (int)distances.size(); ++j) {
        result.push_back(distances[j].second);
    }

    return result;
}