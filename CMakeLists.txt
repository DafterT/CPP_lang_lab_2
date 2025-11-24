cmake_minimum_required(VERSION 3.14)
project(GaussianBlurProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Принудительно Release конфигурация для этой сборки ===
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_USE_LIBRARIES OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

# Включаем оптимизацию сборки для benchmark через временную переменную:
# CMake по умолчанию использует CMAKE_BUILD_TYPE снаружи,
# но если benchmark уже инициализирован, явное переопределение не очень-то помогает.
# Просто устанавливаем CMAKE_BUILD_TYPE до FetchContent_MakeAvailable.
set(CMAKE_BUILD_TYPE Release CACHE STRING "Release" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
FetchContent_MakeAvailable(googlebenchmark)

# ==========================================
# 2. Собираем исходники библиотек
# ==========================================
file(GLOB SOURCES "src/*.cpp")

# ==========================================
# 3. Настройка компиляции (Флаги)
# ==========================================
if(MSVC)
    set(MY_COMPILE_FLAGS /O2 /arch:AVX512)
else()
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
add_executable(run_knn_benchmark knn_main.cpp ${SOURCES})

target_compile_options(run_knn_benchmark PRIVATE ${MY_COMPILE_FLAGS})

target_include_directories(run_knn_benchmark PRIVATE 
    inc
    ${stb_SOURCE_DIR} 
)

target_link_libraries(run_knn_benchmark PRIVATE benchmark::benchmark)

# ==========================================
# 6. Вывод команд по сборке и запуску
# ==========================================
message(STATUS "Configure done. Build with:")
message(STATUS "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release")
message(STATUS "  cmake --build build -j")
message(STATUS "Run Image benchmark:")
message(STATUS "  ./build/run_image_benchmark")
message(STATUS "Run KNN benchmark:")
message(STATUS "  ./build/run_knn_benchmark")
