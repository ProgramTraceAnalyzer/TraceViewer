import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw


def safe_euclidean(u, v):
    """Безопасная версия евклидова расстояния — работает с массивами и скалярами."""
    u_val = float(np.asarray(u).item())
    v_val = float(np.asarray(v).item())
    return np.sqrt((u_val - v_val) ** 2)


def plot_with_dtw(a, b):
    # Гарантируем 1D float-массивы
    a_np = np.array(a, dtype=float).flatten()
    b_np = np.array(b, dtype=float).flatten()

    # Проверка
    if a_np.ndim != 1:
        raise ValueError(f"a must be 1D, got shape {a_np.shape}")
    if b_np.ndim != 1:
        raise ValueError(f"b must be 1D, got shape {b_np.shape}")

    # Используем safe_euclidean вместо scipy.spatial.distance.euclidean
    distance, path = fastdtw(a_np, b_np, dist=safe_euclidean)

    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(a_np, label='Ряд a', color='blue', linewidth=4, marker='o', markersize=5)
    ax.plot(b_np, label='Ряд b', color='red', linewidth=4, marker='s', markersize=5)

    for i, j in path:
        ax.plot([i, j], [a_np[i], b_np[j]], color='green', linestyle='--', alpha=0.5, linewidth=1.8)

    ax.set_title(f'FastDTW: расстояние = {distance:.6f}', fontsize=16)
    ax.set_xlabel('Индекс')
    ax.set_ylabel('Значение')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"FastDTW расстояние: {distance:.6f}")
    print(f"Количество пар соответствий: {len(path)}")


# 🚀 ТЕСТ — СКОПИРУЙТЕ И ЗАПУСТИТЕ ЭТОТ БЛОК
if __name__ == "__main__":
    a = [1, 3, 5, 6, 9, 7, 6, 4, 1, 0, 1, 2]
    b = [0, 1, 2, 4, 6, 8, 7, 5, 3, 1, 0, 1]

    plot_with_dtw(a, b)
