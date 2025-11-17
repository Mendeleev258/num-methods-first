# plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Убедитесь, что папка results существует
RESULTS_DIR = 'results'
OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Список файлов и их типы
files_info = {
    'results_rand.csv': 'random',
    'results_dominant.csv': 'dominant',
    'results_ill_conditioned.csv': 'ill_conditioned'
}

# Загрузка и объединение всех данных
all_data = []
for filename, cond_type in files_info.items():
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Извлекаем low/high из строки вида "[-1.0, 1.0]"
        df['low'] = df['filling range'].apply(lambda x: float(x.strip('[]').split(',')[0]))
        df['high'] = df['filling range'].apply(lambda x: float(x.strip('[]').split(',')[1]))
        df['condition_type'] = cond_type
        all_data.append(df)
    else:
        print(f"Файл {path} не найден. Пропускаем.")

if not all_data:
    raise FileNotFoundError("Ни один CSV-файл не найден в папке 'results'")

df_all = pd.concat(all_data, ignore_index=True)

# Преобразуем диапазон в масштаб (например, max(|low|, |high|))
df_all['scale'] = df_all[['low', 'high']].abs().max(axis=1)

# Настройка стиля
plt.rcParams.update({'font.size': 10})

# Заданные размеры систем для отметок на оси X
system_sizes_ticks = np.logspace(1, 6, 6, base=2).astype(int)
print(f"Отметки на оси X: {system_sizes_ticks}")


# Функция для настройки осей X с заданными отметками размеров систем
def setup_xaxis(ax, x_data=None, xlabel='Размер системы (n)'):
    """Настраивает ось X с заданными отметками размеров систем"""
    ax.set_xlabel(xlabel)

    # Используем заданные отметки, но фильтруем те, что попадают в диапазон данных
    if x_data is not None and len(x_data) > 0:
        x_min, x_max = min(x_data), max(x_data)
        # Фильтруем отметки, чтобы они попадали в диапазон данных
        valid_ticks = [x for x in system_sizes_ticks if x_min <= x <= x_max]

        if valid_ticks:
            ax.set_xticks(valid_ticks)
            ax.set_xticklabels([str(int(x)) for x in valid_ticks])
    else:
        # Если данных нет, используем все заданные отметки
        ax.set_xticks(system_sizes_ticks)
        ax.set_xticklabels([str(int(x)) for x in system_sizes_ticks])


# 1. Влияние размера системы на точность (для каждого типа и метода)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Влияние размера системы на точность решения', fontsize=14)

for ax, (err_type, title) in zip(axes.flat, [
    ('absolute error (A)', 'Абсолютная погрешность (метод A)'),
    ('relative error (A)', 'Относительная погрешность (метод A)'),
    ('absolute error (B)', 'Абсолютная погрешность (метод B)'),
    ('relative error (B)', 'Относительная погрешность (метод B)')
]):
    system_sizes = []
    for cond in df_all['condition_type'].unique():
        subset = df_all[df_all['condition_type'] == cond]
        # Усредняем по экспериментам одного размера
        grouped = subset.groupby('system size')[err_type].mean()
        system_sizes.extend(grouped.index)
        ax.loglog(grouped.index, grouped.values, 'o-', label=f'{cond}')

    setup_xaxis(ax, system_sizes)
    ax.set_ylabel('Погрешность')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    ax.set_title(title)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'size_vs_error.png'), dpi=150)
plt.close()

# 2. Влияние обусловленности (сравнение типов при фиксированном масштабе, например scale=1)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Влияние обусловленности системы на точность (масштаб = 1)', fontsize=14)

# Берём только данные с масштабом ~1 (например, low=-1, high=1)
df_scale1 = df_all[np.isclose(df_all['scale'], 1.0, atol=0.1)]

for ax, (err_type, title) in zip(axes.flat, [
    ('absolute error (A)', 'Абсолютная погрешность (метод A)'),
    ('relative error (A)', 'Относительная погрешность (метод A)'),
    ('absolute error (B)', 'Абсолютная погрешность (метод B)'),
    ('relative error (B)', 'Относительная погрешность (метод B)')
]):
    system_sizes = []
    for cond in df_scale1['condition_type'].unique():
        subset = df_scale1[df_scale1['condition_type'] == cond]
        grouped = subset.groupby('system size')[err_type].mean()
        system_sizes.extend(grouped.index)
        ax.loglog(grouped.index, grouped.values, 'o-', label=f'{cond}')

    setup_xaxis(ax, system_sizes)
    ax.set_ylabel('Погрешность')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    ax.set_title(title)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'condition_vs_error.png'), dpi=150)
plt.close()

# 3. Влияние диапазона заполнения (scale) на точность (при фиксированном размере, например n=32)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Влияние диапазона заполнения на точность (n ≈ 32)', fontsize=14)

# Выбираем размер, близкий к 32
sizes = df_all['system size'].unique()
n_target = 32
n_closest = min(sizes, key=lambda x: abs(x - n_target))
df_n = df_all[df_all['system size'] == n_closest]

for ax, (err_type, title) in zip(axes.flat, [
    ('absolute error (A)', 'Абсолютная погрешность (метод A)'),
    ('relative error (A)', 'Относительная погрешность (метод A)'),
    ('absolute error (B)', 'Абсолютная погрешность (метод B)'),
    ('relative error (B)', 'Относительная погрешность (метод B)')
]):
    scales = []
    for cond in df_n['condition_type'].unique():
        subset = df_n[df_n['condition_type'] == cond]
        # Группируем по масштабу
        grouped = subset.groupby('scale')[err_type].mean()
        scales.extend(grouped.index)
        ax.loglog(grouped.index, grouped.values, 'o-', label=f'{cond}')

    # Для оси масштаба используем специальную настройку
    if scales:
        min_scale, max_scale = min(scales), max(scales)
        # Создаем логарифмически распределенные отметки для масштаба
        scale_ticks = np.logspace(np.log10(min_scale), np.log10(max_scale), num=8)
        ax.set_xticks(scale_ticks)
        ax.set_xticklabels([f'{x:.1f}' for x in scale_ticks])

    ax.set_xlabel('Масштаб заполнения (max(|low|, |high|))')
    ax.set_ylabel('Погрешность')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    ax.set_title(title)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'scale_vs_error.png'), dpi=150)
plt.close()

# 4. Сравнение алгоритмов (A vs B) на одном графике
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Сравнение методов A (прогонка) и B (неустойчивый)', fontsize=14)

# Используем данные с масштабом=1 и случайной обусловленностью
df_compare = df_all[
    (df_all['condition_type'] == 'random') &
    (np.isclose(df_all['scale'], 1.0, atol=0.1))
    ]

for ax, err_type in zip(axes, ['absolute', 'relative']):
    system_sizes = []
    for method in ['A', 'B']:
        col = f'{err_type} error ({method})'
        grouped = df_compare.groupby('system size')[col].mean()
        system_sizes.extend(grouped.index)
        ax.loglog(grouped.index, grouped.values, 'o-', label=f'Метод {method}')

    setup_xaxis(ax, system_sizes)
    ax.set_ylabel(f'{err_type.capitalize()} погрешность')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    ax.set_title(f'{err_type.capitalize()} погрешность: A vs B')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'method_comparison.png'), dpi=150)
plt.close()

print(f"Диаграммы сохранены в папку '{OUTPUT_DIR}'")
print(f"Использованные отметки на оси X: {system_sizes_ticks}")