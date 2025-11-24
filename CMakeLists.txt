cmake_minimum_required(VERSION 3.14)
project(GaussianBlurProject)

set(CMAKE_CXX_STANDARD 17)

# 1. Подключаем STB
include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

# 2. Подключаем Google Benchmark
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(googlebenchmark)

# Находим исходники библиотеки (в папке src)
file(GLOB SOURCES "src/*.cpp")

# 3. Создаем ЕДИНСТВЕННЫЙ исполняемый файл для тестов
# Я использую main.cpp, так как именно там у тебя лежит код тестов (судя по ошибке)
add_executable(run_benchmarks main.cpp ${SOURCES})

# 4. Флаги оптимизации (AVX-512 + O3)
if(MSVC)
    target_compile_options(run_benchmarks PRIVATE /O2 /arch:AVX512)
else()
    target_compile_options(run_benchmarks PRIVATE -O3 -march=native)
endif()

# 5. Пути к заголовкам
target_include_directories(run_benchmarks PRIVATE 
    inc
    ${stb_SOURCE_DIR}
)

# 6. Линкуем библиотеку бенчмарка
target_link_libraries(run_benchmarks PRIVATE benchmark::benchmark)
