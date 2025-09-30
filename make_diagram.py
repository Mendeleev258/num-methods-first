import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Загружаем данные из CSV файла
df = pd.read_csv('results/results.csv')

print("Исходный DataFrame:")
print(df.head())
print(f"\nРазмер исходных данных: {df.shape}")

# ============================================================================
# АГРЕГАЦИЯ ДАННЫХ: СРЕДНЕЕ ЗНАЧЕНИЕ ДЛЯ ОДИНАКОВЫХ РАЗМЕРОВ (10 ЭКСПЕРИМЕНТОВ)
# ============================================================================

# Группируем по размеру системы и вычисляем средние значения
aggregated_df = df.groupby('system size').agg({
    'absolute error (A)': 'mean',
    'relative error (A)': 'mean',
    'absolute error (B)': 'mean',
    'relative error (B)': 'mean',
    'conditioning': 'mean'
}).reset_index()

print("\nАгрегированный DataFrame (средние значения по 10 экспериментам):")
print(aggregated_df)
print(f"\nРазмер агрегированных данных: {aggregated_df.shape}")

# ============================================================================
# ДИАГРАММЫ ДЛЯ МЕТОДА A (СРЕДНИЕ ЗНАЧЕНИЯ ПО 10 ЭКСПЕРИМЕНТАМ)
# ============================================================================

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('МЕТОД A: Зависимости ошибок от размера системы и обусловленности\n(средние значения по 10 экспериментам)',
              fontsize=14, fontweight='bold')

# Абсолютная ошибка метода A vs размер системы
ax1.semilogy(aggregated_df['system size'], aggregated_df['absolute error (A)'], 'bo-',
           linewidth=3, markersize=10, alpha=0.8, markerfacecolor='blue')
ax1.set_xlabel('Размер системы')
ax1.set_ylabel('Абсолютная ошибка (A)')
ax1.set_title('Абсолютная ошибка vs Размер системы\n(средние по 10 экспериментам)')
ax1.grid(True, alpha=0.3)

# Устанавливаем обычные целочисленные подписи на оси X
ax1.set_xticks(aggregated_df['system size'])
ax1.set_xticklabels([f'{int(x)}' for x in aggregated_df['system size']], rotation=45)

# Добавляем тренд
if len(aggregated_df) > 1:
    z = np.polyfit(aggregated_df['system size'], np.log(aggregated_df['absolute error (A)']), 1)
    p = np.poly1d(z)
    ax1.semilogy(aggregated_df['system size'], np.exp(p(aggregated_df['system size'])), 'r--',
              alpha=0.7, linewidth=2, label=f'тренд (наклон: {z[0]:.3f})')
    ax1.legend()

# Относительная ошибка метода A vs размер системы
ax2.semilogy(aggregated_df['system size'], aggregated_df['relative error (A)'], 'ro-',
           linewidth=3, markersize=10, alpha=0.8, markerfacecolor='red')
ax2.set_xlabel('Размер системы')
ax2.set_ylabel('Относительная ошибка (A)')
ax2.set_title('Относительная ошибка vs Размер системы\n(средние по 10 экспериментам)')
ax2.grid(True, alpha=0.3)

# Устанавливаем обычные целочисленные подписи на оси X
ax2.set_xticks(aggregated_df['system size'])
ax2.set_xticklabels([f'{int(x)}' for x in aggregated_df['system size']], rotation=45)

# Добавляем тренд
if len(aggregated_df) > 1:
    z = np.polyfit(aggregated_df['system size'], np.log(aggregated_df['relative error (A)']), 1)
    p = np.poly1d(z)
    ax2.semilogy(aggregated_df['system size'], np.exp(p(aggregated_df['system size'])), 'b--',
              alpha=0.7, linewidth=2, label=f'тренд (наклон: {z[0]:.3f})')
    ax2.legend()

# Абсолютная ошибка метода A vs обусловленность
ax3.loglog(aggregated_df['conditioning'], aggregated_df['absolute error (A)'], 'go-',
           linewidth=3, markersize=10, alpha=0.8, markerfacecolor='green')
ax3.set_xlabel('Число обусловленности')
ax3.set_ylabel('Абсолютная ошибка (A)')
ax3.set_title('Абсолютная ошибка vs Обусловленность\n(средние по 10 экспериментам)')
ax3.grid(True, alpha=0.3)

# Добавляем тренд
if len(aggregated_df) > 1:
    z = np.polyfit(np.log(aggregated_df['conditioning']), np.log(aggregated_df['absolute error (A)']), 1)
    p = np.poly1d(z)
    ax3.loglog(aggregated_df['conditioning'], np.exp(p(np.log(aggregated_df['conditioning']))), 'm--',
              alpha=0.7, linewidth=2, label=f'тренд (наклон: {z[0]:.3f})')
    ax3.legend()

# Относительная ошибка метода A vs обусловленность
ax4.loglog(aggregated_df['conditioning'], aggregated_df['relative error (A)'], 'mo-',
           linewidth=3, markersize=10, alpha=0.8, markerfacecolor='magenta')
ax4.set_xlabel('Число обусловленности')
ax4.set_ylabel('Относительная ошибка (A)')
ax4.set_title('Относительная ошибка vs Обусловленность\n(средние по 10 экспериментам)')
ax4.grid(True, alpha=0.3)

# Добавляем тренд
if len(aggregated_df) > 1:
    z = np.polyfit(np.log(aggregated_df['conditioning']), np.log(aggregated_df['relative error (A)']), 1)
    p = np.poly1d(z)
    ax4.loglog(aggregated_df['conditioning'], np.exp(p(np.log(aggregated_df['conditioning']))), 'g--',
              alpha=0.7, linewidth=2, label=f'тренд (наклон: {z[0]:.3f})')
    ax4.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# ДИАГРАММЫ ДЛЯ МЕТОДА B (СРЕДНИЕ ЗНАЧЕНИЯ ПО 10 ЭКСПЕРИМЕНТАМ)
# ============================================================================

fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('МЕТОД B: Зависимости ошибок от размера системы и обусловленности\n(средние значения по 10 экспериментам)',
              fontsize=14, fontweight='bold')

# Абсолютная ошибка метода B vs размер системы
ax5.semilogy(aggregated_df['system size'], aggregated_df['absolute error (B)'], 'co-',
           linewidth=3, markersize=10, alpha=0.8, markerfacecolor='cyan')
ax5.set_xlabel('Размер системы')
ax5.set_ylabel('Абсолютная ошибка (B)')
ax5.set_title('Абсолютная ошибка vs Размер системы\n(средние по 10 экспериментам)')
ax5.grid(True, alpha=0.3)

# Устанавливаем обычные целочисленные подписи на оси X
ax5.set_xticks(aggregated_df['system size'])
ax5.set_xticklabels([f'{int(x)}' for x in aggregated_df['system size']], rotation=45)

# Добавляем тренд
if len(aggregated_df) > 1:
    z = np.polyfit(aggregated_df['system size'], np.log(aggregated_df['absolute error (B)']), 1)
    p = np.poly1d(z)
    ax5.semilogy(aggregated_df['system size'], np.exp(p(aggregated_df['system size'])), 'r--',
              alpha=0.7, linewidth=2, label=f'тренд (наклон: {z[0]:.3f})')
    ax5.legend()

# Относительная ошибка метода B vs размер системы
ax6.semilogy(aggregated_df['system size'], aggregated_df['relative error (B)'], 'orange', marker='o',
           linewidth=3, markersize=10, alpha=0.8, markerfacecolor='orange')
ax6.set_xlabel('Размер системы')
ax6.set_ylabel('Относительная ошибка (B)')
ax6.set_title('Относительная ошибка vs Размер системы\n(средние по 10 экспериментам)')
ax6.grid(True, alpha=0.3)

# Устанавливаем обычные целочисленные подписи на оси X
ax6.set_xticks(aggregated_df['system size'])
ax6.set_xticklabels([f'{int(x)}' for x in aggregated_df['system size']], rotation=45)

# Добавляем тренд
if len(aggregated_df) > 1:
    z = np.polyfit(aggregated_df['system size'], np.log(aggregated_df['relative error (B)']), 1)
    p = np.poly1d(z)
    ax6.semilogy(aggregated_df['system size'], np.exp(p(aggregated_df['system size'])), 'b--',
              alpha=0.7, linewidth=2, label=f'тренд (наклон: {z[0]:.3f})')
    ax6.legend()

# Абсолютная ошибка метода B vs обусловленность
ax7.loglog(aggregated_df['conditioning'], aggregated_df['absolute error (B)'], 'yo-',
           linewidth=3, markersize=10, alpha=0.8, markerfacecolor='yellow')
ax7.set_xlabel('Число обусловленности')
ax7.set_ylabel('Абсолютная ошибка (B)')
ax7.set_title('Абсолютная ошибка vs Обусловленность\n(средние по 10 экспериментам)')
ax7.grid(True, alpha=0.3)

# Добавляем тренд
if len(aggregated_df) > 1:
    z = np.polyfit(np.log(aggregated_df['conditioning']), np.log(aggregated_df['absolute error (B)']), 1)
    p = np.poly1d(z)
    ax7.loglog(aggregated_df['conditioning'], np.exp(p(np.log(aggregated_df['conditioning']))), 'm--',
              alpha=0.7, linewidth=2, label=f'тренд (наклон: {z[0]:.3f})')
    ax7.legend()

# Относительная ошибка метода B vs обусловленность
ax8.loglog(aggregated_df['conditioning'], aggregated_df['relative error (B)'], 'ko-',
           linewidth=3, markersize=10, alpha=0.8, markerfacecolor='black')
ax8.set_xlabel('Число обусловленности')
ax8.set_ylabel('Относительная ошибка (B)')
ax8.set_title('Относительная ошибка vs Обусловленность\n(средние по 10 экспериментам)')
ax8.grid(True, alpha=0.3)

# Добавляем тренд
if len(aggregated_df) > 1:
    z = np.polyfit(np.log(aggregated_df['conditioning']), np.log(aggregated_df['relative error (B)']), 1)
    p = np.poly1d(z)
    ax8.loglog(aggregated_df['conditioning'], np.exp(p(np.log(aggregated_df['conditioning']))), 'g--',
              alpha=0.7, linewidth=2, label=f'тренд (наклон: {z[0]:.3f})')
    ax8.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# СРАВНИТЕЛЬНЫЕ ДИАГРАММЫ (СРЕДНИЕ ЗНАЧЕНИЯ ПО 10 ЭКСПЕРИМЕНТАМ)
# ============================================================================

fig3, (ax9, ax10) = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle('СРАВНЕНИЕ МЕТОДОВ A и B (средние значения по 10 экспериментам)', fontsize=14, fontweight='bold')

# Сравнение абсолютных ошибок vs размер системы
ax9.semilogy(aggregated_df['system size'], aggregated_df['absolute error (A)'], 'bo-',
           linewidth=3, markersize=8, label='Метод A', alpha=0.8)
ax9.semilogy(aggregated_df['system size'], aggregated_df['absolute error (B)'], 'ro-',
           linewidth=3, markersize=8, label='Метод B', alpha=0.8)
ax9.set_xlabel('Размер системы')
ax9.set_ylabel('Абсолютная ошибка')
ax9.set_title('Сравнение абсолютных ошибок\n(средние по 10 экспериментам)')
ax9.grid(True, alpha=0.3)
ax9.legend()

# Устанавливаем обычные целочисленные подписи на оси X
ax9.set_xticks(aggregated_df['system size'])
ax9.set_xticklabels([f'{int(x)}' for x in aggregated_df['system size']], rotation=45)

# Сравнение относительных ошибок vs обусловленность
ax10.loglog(aggregated_df['conditioning'], aggregated_df['relative error (A)'], 'go-',
           linewidth=3, markersize=8, label='Метод A', alpha=0.8)
ax10.loglog(aggregated_df['conditioning'], aggregated_df['relative error (B)'], 'mo-',
           linewidth=3, markersize=8, label='Метод B', alpha=0.8)
ax10.set_xlabel('Число обусловленности')
ax10.set_ylabel('Относительная ошибка')
ax10.set_title('Сравнение относительных ошибок\n(средние по 10 экспериментам)')
ax10.grid(True, alpha=0.3)
ax10.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# СТАТИСТИЧЕСКАЯ ИНФОРМАЦИЯ (ПО СРЕДНИМ ЗНАЧЕНИЯМ ИЗ 10 ЭКСПЕРИМЕНТОВ)
# ============================================================================

print("\n" + "="*80)
print("СТАТИСТИЧЕСКАЯ ИНФОРМАЦИЯ О МЕТОДАХ (СРЕДНИЕ ПО 10 ЭКСПЕРИМЕНТАМ)")
print("="*80)

print(f"\nКоличество уникальных размеров систем: {len(aggregated_df)}")
print(f"Диапазон размеров: от {aggregated_df['system size'].min()} до {aggregated_df['system size'].max()}")

print("\nМЕТОД A:")
print(f"  Абсолютная ошибка - Среднее: {aggregated_df['absolute error (A)'].mean():.2e}")
print(f"  Абсолютная ошибка - Медиана: {aggregated_df['absolute error (A)'].median():.2e}")
print(f"  Относительная ошибка - Среднее: {aggregated_df['relative error (A)'].mean():.2e}")

print("\nМЕТОД B:")
print(f"  Абсолютная ошибка - Среднее: {aggregated_df['absolute error (B)'].mean():.2e}")
print(f"  Абсолютная ошибка - Медиана: {aggregated_df['absolute error (B)'].median():.2e}")
print(f"  Относительная ошибка - Среднее: {aggregated_df['relative error (B)'].mean():.2e}")

# Сравнение методов
improvement_abs = (aggregated_df['absolute error (A)'] - aggregated_df['absolute error (B)']) / aggregated_df['absolute error (A)'] * 100
improvement_rel = (aggregated_df['relative error (A)'] - aggregated_df['relative error (B)']) / aggregated_df['relative error (A)'] * 100

print(f"\nУЛУЧШЕНИЕ МЕТОДА B ОТНОСИТЕЛЬНО A:")
print(f"  По абсолютной ошибке: {improvement_abs.mean():.1f}% в среднем")
print(f"  По относительной ошибке: {improvement_rel.mean():.1f}% в среднем")
print(f"  Минимальное улучшение (абс.): {improvement_abs.min():.1f}%")
print(f"  Максимальное улучшение (абс.): {improvement_abs.max():.1f}%")

# Сохраняем агрегированные данные
aggregated_df.to_csv('results/results_aggregated_10.csv', index=False)
print(f"\nАгрегированные данные сохранены в 'results/results_aggregated_10.csv'")

print("\nДиаграммы успешно построены по средним значениям из 10 экспериментов! ✅")