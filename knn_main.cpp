#include <benchmark/benchmark.h>
#include <vector>
#include <iostream>

// Подключаем твой заголовочный файл
#include "knn_searcher.h"

// Класс-фикстура для подготовки данных
// Google Benchmark создает новый экземпляр фикстуры для каждого теста,
// поэтому данные генерируются заново для каждой комбинации параметров.
class KnnFixture : public benchmark::Fixture {
public:
    KnnData data;
    int k;
    int num_vectors;
    int dim;

    void SetUp(const ::benchmark::State& state) {
        // Параметры приходят из state.range(i)
        // range(0) -> Количество векторов (N)
        // range(1) -> Размерность вектора (Dim)
        // range(2) -> Количество соседей (K)
        
        num_vectors = state.range(0);
        dim = state.range(1);
        k = state.range(2);

        // Генерируем данные
        // Примечание: для больших N и Dim генерация может занять время,
        // но она не входит в замер времени самого цикла while(state.KeepRunning())
        data = KnnSearcher::generate_data(num_vectors, dim);
    }

    void TearDown(const ::benchmark::State& state) {
        // Очистка памяти (не обязательна для вектора, но для порядка)
        data.dataset.clear();
        data.query.clear();
    }
};

// 1. Бенчмарк для NAIVE реализации
BENCHMARK_DEFINE_F(KnnFixture, BM_FindNaive)(benchmark::State& state) {
    for (auto _ : state) {
        // Код, который замеряем
        std::vector<int> result = KnnSearcher::find_naive(data.dataset, data.query, k);
        
        // clobber memory, чтобы компилятор не выкинул "бесполезный" код
        benchmark::DoNotOptimize(result.data());
    }

    // Метрика: Обработано байт данных (чтение датасета)
    // N векторов * Dim элементов * 4 байта (float)
    int64_t bytes_processed = int64_t(state.iterations()) * 
                              int64_t(num_vectors) * 
                              int64_t(dim) * 
                              sizeof(float);
    state.SetBytesProcessed(bytes_processed);
}

// 2. Бенчмарк для SIMD (AVX-512) реализации
BENCHMARK_DEFINE_F(KnnFixture, BM_FindSIMD)(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<int> result = KnnSearcher::find_simd(data.dataset, data.query, k);
        benchmark::DoNotOptimize(result.data());
    }

    int64_t bytes_processed = int64_t(state.iterations()) * 
                              int64_t(num_vectors) * 
                              int64_t(dim) * 
                              sizeof(float);
    state.SetBytesProcessed(bytes_processed);
}

// Функция генерации аргументов
// Создает декартово произведение (N x Dim x K)
static void CustomArguments(benchmark::internal::Benchmark* b) {
    // 1. Размер датасета: от 32 до 131072 (степени двойки)
    std::vector<int> sizes;
    for (int n = 32; n <= 131072; n *= 4) {
        sizes.push_back(n);
    }

    // 2. Длина вектора: от 2 до 128 (степени двойки)
    std::vector<int> dims;
    for (int d = 2; d <= 128; d *= 2) {
        dims.push_back(d);
    }

    // 3. K ближайших: от 1 до 32 (используем степени двойки для покрытия диапазона)
    std::vector<int> ks = {1, 2, 4, 8, 16, 32};

    for (int s : sizes) {
        for (int d : dims) {
            for (int k : ks) {
                // Передаем тройку аргументов
                b->Args({s, d, k});
            }
        }
    }
}

// Регистрация бенчмарков с применением аргументов
BENCHMARK_REGISTER_F(KnnFixture, BM_FindNaive)
    ->Apply(CustomArguments)
    ->Unit(benchmark::kMicrosecond); // kMicrosecond обычно удобнее для малых N

BENCHMARK_REGISTER_F(KnnFixture, BM_FindSIMD)
    ->Apply(CustomArguments)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
