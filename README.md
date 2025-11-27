Компиляция:
```
mkdir build
cd build
cmake ..
cmake --build .
```
Запуск теста картинок
```
./run_image_benchmark --benchmark_out=results_image.json --benchmark_out_format=json
```
Запуск теста К-ближайших
```
./run_knn_benchmark --benchmark_out=results_knn.json --benchmark_out_format=json
```

```
lscpu | grep -i avx
```
```
grep "model name" /proc/cpuinfo | head -1
```