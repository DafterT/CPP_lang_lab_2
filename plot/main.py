import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Настройки стиля
sns.set_theme(style="whitegrid")

# Путь к файлу с результатами
FILENAME = '../build/results.json' 

if not os.path.exists(FILENAME):
    print(f"Ошибка: Файл {FILENAME} не найден. Запустите бенчмарк сначала.")
    exit()

print(f"Читаю файл {FILENAME}...")
with open(FILENAME, 'r') as f:
    data = json.load(f)

# 1. Парсинг данных
records = []
for bench in data['benchmarks']:
    # Разбиваем строку вида: "BlurFixture/BM_ProcessDefault/32/3"
    name_parts = bench['name'].split('/')
    
    # Индексы:
    # 0: BlurFixture
    # 1: Имя метода (BM_ProcessDefault / BM_ProcessSIMD)
    # 2: Размер картинки
    # 3: Размер ядра
    
    method_raw = name_parts[1]
    img_size = int(name_parts[2])
    kernel_size = int(name_parts[3])
    
    # Переводим байты/сек в ГБ/сек (10^9)
    gb_per_sec = bench['bytes_per_second'] / 1e9
    
    records.append({
        'Method': 'SIMD (AVX-512)' if 'SIMD' in method_raw else 'Default (C++)',
        'Image Size': img_size,
        'Kernel Size': kernel_size,
        'Speed (GB/s)': gb_per_sec
    })

df = pd.DataFrame(records)

# 2. Генерация отдельных графиков
kernel_sizes = sorted(df['Kernel Size'].unique())

for k_size in kernel_sizes:
    # Создаем новую фигуру для каждого ядра
    plt.figure(figsize=(12, 8))
    
    # Фильтруем данные только для текущего ядра
    subset = df[df['Kernel Size'] == k_size]
    
    # Рисуем график
    ax = sns.barplot(
        data=subset, 
        x='Image Size', 
        y='Speed (GB/s)', 
        hue='Method',
        palette="viridis"
    )
    
    # Настройки заголовков
    plt.title(f'Производительность (Kernel {k_size}x{k_size})', fontsize=16, pad=20)
    plt.ylabel('Скорость (GB/s)', fontsize=14)
    plt.xlabel('Размер изображения (NxN)', fontsize=14)
    plt.legend(title='Метод', fontsize=12, title_fontsize=12)
    
    # Добавляем точные подписи значений (3 знака после запятой)
    for container in ax.containers:
        # fmt='%.3f' покажет 0.019 вместо 0.0
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)
    
    # Сохраняем в отдельный файл
    filename = f'benchmark_kernel_{k_size}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150) # dpi=150 для четкости
    print(f"Сохранено: {filename}")
    
    # Закрываем фигуру, чтобы освободить память
    plt.close()

print("\nГотово! Создано 4 изображения.")
