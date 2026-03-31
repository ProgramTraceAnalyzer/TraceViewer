import ast
import subprocess
from collections import Mapping
from typing import List, Tuple, Set, Optional, Union
import pandas as pd
import numpy as np
from itertools import permutations, combinations
from fastdtw import fastdtw
from numpy.core.multiarray import ndarray
from scipy.spatial.distance import euclidean, chebyshev
from sklearn.preprocessing import StandardScaler
import Levenshtein
import matplotlib.pyplot as plt

import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray


def smith_waterman(
        seq1: List[int],
        seq2: List[int],
        match_score: int = 2,
        mismatch_penalty: int = -1,
        gap_penalty: int = -1,
) -> tuple[NDArray, list[tuple[int, int]], float]:
    """
    Perform local sequence alignment using the Smith–Waterman algorithm.

    Parameters
    ----------
    seq1 : List[int]
        First input sequence (list of integers).
    seq2 : List[int]
        Second input sequence (list of integers).
    match_score : int, optional
        Score for a match (default 2).
    mismatch_penalty : int, optional
        Penalty for a mismatch (default -1).
    gap_penalty : int, optional
        Penalty for a gap (default -1).

    Returns
    -------
    NDArray
        Score matrix as a numpy array.
    List[Tuple[int, int]]
        List of index pairs (i, j) representing the optimal local alignment.
        The list is ordered from start to end of the alignment.
    float
        Percentage similarity between the sequences (based on aligned positions).

    Example
    -------
    a = [1, 2, 3, 4, 5]
    b = [0, 1, 2, 3, 4, 5]
    matrix, alignment, similarity = smith_waterman(a, b)
    """
    n, m = len(seq1), len(seq2)
    global_similarity = 0

    # Initialize DP matrix and traceback pointers
    score_matrix = [[0] * (m + 1) for _ in range(n + 1)]
    traceback = [[0] * (m + 1) for _ in range(n + 1)]

    max_i, max_j = 0, 0
    max_score = 0

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = score_matrix[i - 1][j - 1] + match_score
            else:
                diag = score_matrix[i - 1][j - 1] + mismatch_penalty

            up = score_matrix[i - 1][j] + gap_penalty
            left = score_matrix[i][j - 1] + gap_penalty

            best = max(0, diag, up, left)
            score_matrix[i][j] = best

            # Record traceback direction
            if best == 0:
                traceback[i][j] = 0
            elif best == diag:
                traceback[i][j] = 1
            elif best == up:
                traceback[i][j] = 2
            else:
                traceback[i][j] = 3

            if best > max_score:
                max_score = best
                max_i, max_j = i, j

    # Backtrack from the cell with the highest score
    alignment = []
    i, j = max_i, max_j
    while i > 0 and j > 0 and score_matrix[i][j] > 0:
        if traceback[i][j] == 1:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif traceback[i][j] == 2:
            alignment.append((i - 1, j))
            i -= 1
        elif traceback[i][j] == 3:
            alignment.append((i, j - 1))
            j -= 1
        else:
            break

    alignment.reverse()

    # Calculate percentage similarity - FIXED VERSION
    if not alignment:
        similarity = 0.0
    else:
        # Count aligned positions (excluding gaps)
        aligned_positions = len(alignment)
        matches = 0

        for (i, j) in alignment:
            # Check if both indices are valid (not gaps)
            if i is not None and j is not None:
                if seq1[i] == seq2[j]:
                    matches += 1

        # Calculate similarity based on aligned positions
        similarity = (matches / aligned_positions) * 100 if aligned_positions > 0 else 0.0

        # Альтернативный вариант расчета (глобальная схожесть)
        total_positions = max(len(seq1), len(seq2))
        global_similarity = (matches / total_positions) * 100

    m_array = np.array(score_matrix, dtype=float)
    return m_array, alignment, global_similarity

def lcs(
    seq1: List[int],
    seq2: List[int],
) -> tuple[NDArray, list[tuple[int, int]], float]:
    """
    Compute the Longest Common Subsequence (LCS) of two integer sequences.

    Parameters
    ----------
    seq1 : List[int]
        First input sequence.
    seq2 : List[int]
        Second input sequence.

    Returns
    -------
    List[Tuple[int, int]]
        List of index pairs (i, j) representing the LCS.
        The list is ordered from start to end of the subsequence.

    Example
    -------
    >>> a = [1, 2, 3, 4, 5]
    >>> b = [0, 1, 2, 3, 4, 5]
    >>> lcs(a, b)
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    """
    n, m = len(seq1), len(seq2)

    # DP matrix: dp[i][j] = length of LCS for seq1[:i] and seq2[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to retrieve the index pairs
    alignment: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            # Match found – add to alignment and move diagonally
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            # Move up
            i -= 1
        else:
            # Move left
            j -= 1

    alignment.reverse()
    m_array = np.array(dp, dtype=float)

    matches = 0
    for (i, j) in alignment:
        # Check if both indices are valid (not gaps)
        if i is not None and j is not None:
            if seq1[i] == seq2[j]:
                matches += 1

    # Альтернативный вариант расчета (глобальная схожесть)
    total_positions = max(len(seq1), len(seq2))
    global_similarity = (matches / total_positions) * 100

    return m_array, alignment, global_similarity


def dtw(seq1: Union[List[int], NDArray], seq2: Union[List[int], NDArray]) -> tuple[
    NDArray, list[tuple[int, int]], float]:
    """
    Вычисляет динамическую трансформацию времени (DTW) между двумя последовательностями.

    Args:
        seq1: Первая последовательность (целые числа или numpy array)
        seq2: Вторая последовательность (целые числа или numpy array)

    Returns:
        dtw_matrix: Матрица трансформаций размером (n+1) x (m+1)
        path: Оптимальный путь выравнивания
        similarity: Схожесть последовательностей в процентах (0-100%)
    """
    # Преобразуем входные данные к целым числам, если это необходимо
    if isinstance(seq1, np.ndarray):
        seq1 = seq1.astype(int).tolist()
    if isinstance(seq2, np.ndarray):
        seq2 = seq2.astype(int).tolist()

    n = len(seq1)
    m = len(seq2)

    # Создаем матрицу DTW размером (n+1) x (m+1) как в LCS
    dtw_matrix = np.zeros((n + 1, m + 1), dtype=float)

    # Инициализируем граничные значения
    for i in range(1, n + 1):
        dtw_matrix[i, 0] = float('inf')
    for j in range(1, m + 1):
        dtw_matrix[0, j] = float('inf')
    dtw_matrix[0, 0] = 0

    # Заполняем матрицу DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],  # вставка
                                          dtw_matrix[i, j - 1],  # удаление
                                          dtw_matrix[i - 1, j - 1])  # совпадение

    # Восстанавливаем оптимальный путь
    path = []
    i, j = n, m

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))  # переходим к индексам оригинальных последовательностей

        # Находим направление минимальной стоимости
        min_val = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

        if min_val == dtw_matrix[i - 1, j - 1]:
            i -= 1
            j -= 1
        elif min_val == dtw_matrix[i - 1, j]:
            i -= 1
        else:
            j -= 1

    path.reverse()  # переворачиваем путь для правильного порядка

    # Вычисляем схожесть на основе нормализованной стоимости DTW
    total_cost = dtw_matrix[n, m]

    # Максимально возможная стоимость - если бы все элементы были максимально различны
    max_value1 = max(seq1) if seq1 else 0
    min_value1 = min(seq1) if seq1 else 0
    max_value2 = max(seq2) if seq2 else 0
    min_value2 = min(seq2) if seq2 else 0

    max_possible_diff = max(abs(max_value1 - min_value2),
                            abs(max_value2 - min_value1),
                            1)  # минимум 1 чтобы избежать деления на 0

    max_possible_cost = (n + m) * max_possible_diff

    # Нормализуем и преобразуем в процент схожести
    if max_possible_cost > 0:
        normalized_cost = total_cost / max_possible_cost
        similarity = max(0, 100 - (normalized_cost * 100))
    else:
        similarity = 100.0  # обе последовательности пусты

    return dtw_matrix, path, similarity

seq1 = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
seq2 = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]

matrix, path, sim = dtw(seq1, seq2)
print("DTW Matrix shape:", matrix.shape)
print("Path:", path)
print(f"Similarity: {sim:.2f}%")
