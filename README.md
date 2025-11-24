Компиляция:
```
mkdir build
cd build
cmake ..
cmake --build .
```
Запуск теста картинок
```
./run_image_benchmark --benchmark_format=json --benchmark_out=results_image.json
```
Запуск теста К-ближайших
```
./run_knn_benchmark --benchmark_format=json --benchmark_out=results_knn.json
```