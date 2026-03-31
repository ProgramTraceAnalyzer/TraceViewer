import numpy as np
from typing import Union, List

def remove_stutter_steps(l: Union[np.ndarray, List[int]]) -> np.ndarray:
    if len(l) == 0:
        return np.array([])

    # Конвертация в numpy.array, если входной массив не является им
    arr = np.asarray(l)

    # Находим индексы, где элемент не равен предыдущему
    mask = np.concatenate(([True], arr[1:] != arr[:-1]))

    # Возвращаем отфильтрованный массив
    return arr[mask]

def reverse_sequence(l: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Переворачивает последовательность целых чисел.

    Args:
        l: Входная последовательность (numpy.ndarray или List[int]).

    Returns:
        np.ndarray: Перевернутая последовательность.
    """
    if isinstance(l, list):
        l = np.array(l, dtype=int)  # Преобразуем список в numpy.ndarray, если нужно
    return l[::-1]  # Переворачиваем массив с помощью среза