cmake_minimum_required(VERSION 3.14)
project(GaussianBlurProject)

set(CMAKE_CXX_STANDARD 17)

# ==========================================
# 1. Подключаем зависимости (STB + Benchmark)
# ==========================================
include(FetchContent)

# STB
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

# Google Benchmark
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(googlebenchmark)

# ==========================================
# 2. Собираем исходники библиотек
# ==========================================
# Находим все .cpp в папке src (ImageConvolver.cpp, KnnSearcher.cpp и т.д.)
file(GLOB SOURCES "src/*.cpp")

# ==========================================
# 3. Настройка компиляции (Флаги)
# ==========================================
# Выносим флаги в переменную, чтобы применить к обоим таргетам
if(MSVC)
    set(MY_COMPILE_FLAGS /O2 /arch:AVX512)
else()
    # -march=native автоматически включит avx512f, avx512dq и т.д., если процессор их поддерживает
    set(MY_COMPILE_FLAGS -O3 -march=native)
endif()

# ==========================================
# 4. Таргет 1: Бенчмарк Свертки (Image)
# ==========================================
add_executable(run_image_benchmark image_main.cpp ${SOURCES})

target_compile_options(run_image_benchmark PRIVATE ${MY_COMPILE_FLAGS})

target_include_directories(run_image_benchmark PRIVATE 
    inc
    ${stb_SOURCE_DIR}
)

target_link_libraries(run_image_benchmark PRIVATE benchmark::benchmark)

# ==========================================
# 5. Таргет 2: Бенчмарк KNN
# ==========================================
# Важно: убедись, что файл с тестом KNN называется knn_main.cpp
add_executable(run_knn_benchmark knn_main.cpp ${SOURCES})

target_compile_options(run_knn_benchmark PRIVATE ${MY_COMPILE_FLAGS})

target_include_directories(run_knn_benchmark PRIVATE 
    inc
    # STB тут может быть не нужен, но оставим для единообразия путей
    ${stb_SOURCE_DIR} 
)

target_link_libraries(run_knn_benchmark PRIVATE benchmark::benchmark)
