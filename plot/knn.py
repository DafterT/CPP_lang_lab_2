import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# ==========================================
# Настройки
# ==========================================
sns.set_theme(style="whitegrid")
# Укажи здесь путь к твоему файлу json
FILENAME = '../build/results_knn.json' 

if not os.path.exists(FILENAME):
    # Фолбек для запуска скрипта рядом с файлом
    FILENAME = 'knn_results.json'
    
if not os.path.exists(FILENAME):
    print(f"Ошибка: Файл {FILENAME} не найден. Сначала запусти бенчмарк: ./run_knn_benchmark --benchmark_format=json > knn_results.json")
    exit()

print(f"Читаю файл {FILENAME}...")
with open(FILENAME, 'r') as f:
    data = json.load(f)

# ==========================================
# 1. Парсинг данных
# ==========================================
records = []

for bench in data['benchmarks']:
    # Формат имени: "KnnFixture/BM_FindNaive/32/2/1"
    # Где: Fixture / Method / N / Dim / K
    name_parts = bench['name'].split('/')
    
    if len(name_parts) < 5:
        continue

    method_raw = name_parts[1]
    num_vectors = int(name_parts[2]) # N
    dim = int(name_parts[3])         # D
    k_neighbors = int(name_parts[4]) # K
    
    # Переводим байты/сек в ГБ/сек
    gb_per_sec = bench['bytes_per_second'] / 1e9
    
    records.append({
        'Method': 'SIMD (AVX-512)' if 'SIMD' in method_raw else 'Default (C++)',
        'Dataset Size (N)': num_vectors,
        'Dimension': dim,
        'K': k_neighbors,
        'Speed (GB/s)': gb_per_sec
    })

df = pd.DataFrame(records)

# ==========================================
# 2. Генерация графиков
# ==========================================
# Мы будем создавать отдельный график для каждой РАЗМЕРНОСТИ (Dimension),
# так как поведение кэша сильно зависит от длины вектора.

unique_dims = sorted(df['Dimension'].unique())

for dim in unique_dims:
    plt.figure(figsize=(14, 9)) # Чуть увеличил высоту для подписей
    
    subset = df[df['Dimension'] == dim]
    
    # Рисуем график
    # Можно добавить errorbar=None, если хочешь вообще убрать "палки"
    ax = sns.barplot(
        data=subset,
        x='Dataset Size (N)',
        y='Speed (GB/s)',
        hue='Method',
        palette="viridis",
        errorbar='sd' # 'sd' - стандартное отклонение (или 'ci' для доверительного интервала)
    )
    
    plt.title(f'Производительность KNN (Vector Dim: {dim})', fontsize=16, pad=20)
    plt.ylabel('Скорость (GB/s)', fontsize=14)
    plt.xlabel('Количество векторов в базе (N)', fontsize=14)
    plt.legend(title='Реализация', fontsize=12)
    
    # === ИСПРАВЛЕНИЕ ПОДПИСЕЙ ===
    for container in ax.containers:
        ax.bar_label(
            container, 
            fmt='%.2f', 
            padding=5,           # Отступ чуть больше
            rotation=0,         # Вертикальный текст
            fontsize=14,
            # Добавляем белую подложку (bbox), чтобы текст не сливался с "палкой"
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.7) 
        )

    # Увеличим верхнюю границу графика на 15%, чтобы подписи длинных столбцов не обрезались
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)

    # Сохраняем
    filename = f'knn_benchmark_dim_{dim}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранен график: {filename}")

print("\nВсе графики построены!")
