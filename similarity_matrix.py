import ast
import subprocess
from collections import Mapping

from typing import List, Tuple, Set, Optional
import pandas as pd
import numpy as np
import json
from itertools import permutations, combinations

from PyQt5.QtGui import QColor
from fastdtw import fastdtw
from numpy.core.multiarray import ndarray
from scipy.spatial.distance import euclidean, chebyshev
from sklearn.preprocessing import StandardScaler
import Levenshtein
import matplotlib.pyplot as plt

from dot_processer import dot_file_to_adjacency_dict
from dtw_plot import safe_euclidean
import os
from pathlib import Path

from sequence_view_splitter import SequenceViewSplitter
from metrics import smith_waterman, lcs, dtw
from sequence_preprocessing import remove_stutter_steps, reverse_sequence, preprocess_sequence
from variable_action_history_splitter import VariableActionHistorySplitter

def read_json_file(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def calculate_var_distances_by_dfs(variables_dfs1, variables_dfs2, allignment_function, non_stutter = False, reverse = False, action_seq_json1=None, action_seq_json2=None, pidg_dot1 = None, pidg_dot2 = None, remove_unused = False):
    dtws = {}
    for var1 in variables_dfs1:
        dtws[var1] = {}
        list_values1 = np.array(variables_dfs1[var1].tolist(), dtype=float).flatten()
        action_seq1 = read_json_file(action_seq_json1)
        pidg_adj_matrix1 = dot_file_to_adjacency_dict(pidg_dot1)
        list_values1 = preprocess_sequence(list_values1, non_stutter, reverse, var1, action_seq1, pidg_adj_matrix1, remove_unused)
        for var2 in variables_dfs2:
            list_values2 = np.array(variables_dfs2[var2].tolist(), dtype=float).flatten()
            action_seq2 = read_json_file(action_seq_json2)
            pidg_adj_matrix2 = dot_file_to_adjacency_dict(pidg_dot2)
            list_values2 = preprocess_sequence(list_values2, non_stutter, reverse, var2, action_seq2, pidg_adj_matrix2, remove_unused)
            if len(list_values1)> 0 and len(list_values2)>0:
                #dtw_dist, dtw_path = fastdtw(list_values1, list_values2, dist=safe_euclidean)
                print("CALL ALLIGNMENT FUNCTION", allignment_function)
                matrix, path, similarity = allignment_function(list_values1, list_values2)


                dtws[var1][var2] = {}
                dtws[var1][var2]["similarity"] = similarity
                dtws[var1][var2]["path"] = path
                dtws[var1][var2]["matrix"] = matrix
                dtws[var1][var2]["row_list_values"] = list_values1
                dtws[var1][var2]["col_list_values"] = list_values2
                # print(dtw)
            else:
                dtws[var1][var2] = {}
                dtws[var1][var2]["similarity"] = 0
                dtws[var1][var2]["path"] = None
                dtws[var1][var2]["matrix"] = None
                dtws[var1][var2]["row_list_values"] = list_values1
                dtws[var1][var2]["col_list_values"] = list_values2
    return dtws

def calculate_var_distances(trace_json1, trace_json2, allignment_function, fillna=None, non_stutter = False, reverse = False, action_seq_json1=None, action_seq_json2=None, pidg_dot1 = None, pidg_dot2 = None, remove_unused = False):
    variables_dfs1 = get_variable_dfs(trace_json1, fillna)
    variables_dfs2 = get_variable_dfs(trace_json2, fillna)
    return calculate_var_distances_by_dfs(variables_dfs1,variables_dfs2, allignment_function, non_stutter, reverse, action_seq_json1, action_seq_json2, pidg_dot1, pidg_dot2, remove_unused)

def calculate_read_var_distances(read_seq_json1, read_seq_json2, allignment_function, reverse = False):
    variables_dfs1 = get_read_variables_dfs(read_seq_json1)
    variables_dfs2 = get_read_variables_dfs(read_seq_json2)
    return calculate_var_distances_by_dfs(variables_dfs1, variables_dfs2, allignment_function, reverse)

def calculate_similary_two_json_traces(file1, file2, common_variables):
    # Ваши данные трасс
    trace1 = []  # первая трасса
    trace2 = []  # вторая трасса

    # Чтение содержимого файла state_seq.json в переменную trace
    with open(file1, 'r', encoding='utf-8') as file:
        trace1 = json.load(file)

    with open(file2, 'r', encoding='utf-8') as file:
        trace2 = json.load(file)

    # Если есть известные общие переменные
    # common_variables = ['side_A', 'side_B']  # пример

    # Вычисляем схожесть
    result = calculate_similarity_percentage(trace1, trace2, common_variables)
    print("RESULT", result)

    return result


def get_variable_dfs(trace_json, fillna=None):
    scalar_memory_df = generate_scalar_memory_df(trace_json)
    variables = scalar_memory_df.columns.to_list()
    variable_dfs = {}
    for var in variables:
        var_df = generate_variable_df(scalar_memory_df, var)
        if fillna != None:
            var_df = var_df.fillna(fillna)
        variable_dfs[var] = var_df
    return variable_dfs


def collect_all_stings_from_2D_string_array(string_array_2D):
    names = set()
    for el in string_array_2D:
        for name in el:
            names.add(name)
    return names


def get_read_variables_dfs(read_sequence_json):
    read_sequence = []
    with open(read_sequence_json, 'r') as f:
        read_sequence = json.load(f)
    # Собрать имена всех переменных
    names = collect_all_stings_from_2D_string_array(read_sequence)
    len_read_sequence = len(read_sequence)
    # Создать датафрейм для всех переменных
    df = pd.DataFrame(0, index=range(len_read_sequence), columns=list(names))
    index = 0
    for el in read_sequence:
        for name in el:
            df.loc[index, name] = 1
        index += 1
    dfs = {}
    for name in names:
        dfs[name] = df[name]
    return dfs


def normalize_dict_cols_in_df(array_df: pd.DataFrame) -> pd.DataFrame:
    """
    Находит в датафрейме столбцы, содержащие словари (dict/Mapping) или строки, которые можно
    безопасно распарсить в словари, и разворачивает их в набор скалярных столбцов вида
    "<исходное_имя>[<ключ>]". Значения приводятся к числовым, где возможно (иначе NaN).
    Исходные словарные столбцы удаляются. Порядок столбцов сохраняется.

    Пример:
        col = {'0': 1, '1': 2, '2': -3}  ->  col[0], col[1], col[2]
    """
    df = array_df.copy()

    def try_to_mapping(val) -> Optional[Mapping]:
        if pd.isna(val):
            return None
        if isinstance(val, Mapping):
            return val
        if isinstance(val, str):
            s = val.strip()
            if not (s.startswith("{") and s.endswith("}")):
                return None
            # Сначала пробуем безопасный ast.literal_eval (поддерживает одинарные кавычки)
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, Mapping):
                    return obj
            except Exception:
                pass
            # Потом JSON (на случай валидного JSON)
            try:
                obj = json.loads(s)
                if isinstance(obj, Mapping):
                    return obj
            except Exception:
                pass
        return None

    # Определяем, какие столбцы разворачивать, и заранее готовим развёрнутые фреймы
    expanded_by_col = {}
    dict_like_cols = []

    for col in df.columns:
        s = df[col]
        # Берём первое ненулевое значение для быстрой проверки
        first_nonnull = next((v for v in s if pd.notna(v)), None)
        first_map = try_to_mapping(first_nonnull)
        if first_map is None:
            continue  # не похоже на словарный столбец

        # Конвертируем все значения столбца в mapping или None
        converted = s.apply(try_to_mapping)

        # Формируем DataFrame из списка dict'ов; None -> строка с NaN
        expanded = pd.DataFrame(converted.tolist(), index=df.index)

        if expanded.shape[1] == 0:
            # Нет ключей — пропускаем
            continue

        # Сортировка столбцов: если все ключи числовые — по числовому значению
        cols = list(expanded.columns)

        def is_intable(x):
            try:
                int(str(x))
                return True
            except Exception:
                return False

        if all(is_intable(c) for c in cols):
            expanded = expanded.reindex(sorted(cols, key=lambda c: int(str(c))), axis=1)

        # Приводим значения к числовому типу, где возможно
        expanded = expanded.apply(pd.to_numeric, errors='coerce')

        # Переименовываем столбцы в формат "<col>[<key>]"
        expanded.columns = [f"{col}[{c}]" for c in expanded.columns]

        expanded_by_col[col] = expanded
        dict_like_cols.append(col)

    if not dict_like_cols:
        return df  # нечего разворачивать

    # Собираем итоговый датафрейм, сохраняя порядок столбцов
    parts = []
    for col in df.columns:
        if col in expanded_by_col:
            parts.append(expanded_by_col[col])
        else:
            parts.append(df[[col]])

    result = pd.concat(parts, axis=1)
    return result


def generate_array_scalars(array_seq_df: pd.Series):
    array_memory_df = pd.DataFrame(array_seq_df.tolist())
    normalized_array_df = normalize_dict_cols_in_df(array_memory_df)
    return normalized_array_df


def generate_scalar_memory_df(seq_file) -> pd.DataFrame:
    seq_df = pd.read_json(open(seq_file, 'r', encoding='utf-8').read())
    scalar_memory_series = seq_df['memory'].apply(lambda x: x.get('scalar_memory') if isinstance(x, dict) else {})
    array_memory_df = generate_array_scalars(
        seq_df['memory'].apply(lambda x: x.get('array_memory') if isinstance(x, dict) else {}))
    scalar_memory_df = pd.DataFrame(scalar_memory_series.tolist())
    scalar_memory_df = pd.concat([scalar_memory_df.reset_index(drop=True), array_memory_df.reset_index(drop=True)],
                                 axis=1)
    return scalar_memory_df


def generate_variable_df(scalar_memory_df, var_name):
    df_only_variable = scalar_memory_df[var_name]
    return df_only_variable


def generate_dataframes(seq_file, variables):
    seq_df = pd.read_json(open(seq_file).read())
    scalar_memory_series = seq_df['memory'].apply(lambda x: x.get('scalar_memory') if isinstance(x, dict) else {})
    array_memory_df = generate_array_scalars(
        seq_df['memory'].apply(lambda x: x.get('array_memory') if isinstance(x, dict) else {}))
    scalar_memory_df = pd.DataFrame(scalar_memory_series.tolist())
    scalar_memory_df = pd.concat([scalar_memory_df.reset_index(drop=True), array_memory_df.reset_index(drop=True)],
                                 axis=1)

    variables = (list(scalar_memory_df.columns))
    existing_columns = [col for col in variables if col in scalar_memory_df.columns]
    df_only_variables = scalar_memory_df[existing_columns]
    df_deduplicated = remove_stutter_steps(df_only_variables)
    df_deduplicated.index = pd.RangeIndex(start=0, stop=len(df_deduplicated), step=1)
    df_by_variables = {}
    df_by_variables_not_stutter = {}
    for v in variables:
        if v in scalar_memory_df.columns:
            df_by_variables[v] = scalar_memory_df[v]
            df_by_variables_not_stutter[v] = remove_stutter_steps(pd.DataFrame({v: df_by_variables[v].tolist()}))
            df_by_variables_not_stutter[v].index = pd.RangeIndex(start=0, stop=len(df_by_variables_not_stutter[v]),
                                                                 step=1)
            print("NON Stutter")
            print(df_by_variables_not_stutter[v])
        else:
            df_by_variables[v] = pd.Series([np.nan] * len(seq_df))
            df_by_variables_not_stutter[v] = pd.DataFrame(columns=[v])

    return seq_df, scalar_memory_df, df_only_variables, df_deduplicated, df_by_variables, df_by_variables_not_stutter


def find_optimal_variable_mapping(trace1, trace2, common_vars=None):
    """
    Находит оптимальное сопоставление переменных между двумя трассами
    """
    # Извлекаем все уникальные переменные из обеих трасс
    vars1 = set()
    vars2 = set()

    for state in trace1:
        vars1.update(state['memory']['scalar_memory'].keys())
    for state in trace2:
        vars2.update(state['memory']['scalar_memory'].keys())

    vars1 = sorted(vars1)
    vars2 = sorted(vars2)

    # Убираем общие переменные (если заданы)
    if common_vars:
        for var in common_vars:
            if var in vars1: vars1.remove(var)
            if var in vars2: vars2.remove(var)

    # Если количество переменных разное, берем минимум
    min_vars_count = min(len(vars1), len(vars2))
    vars1 = vars1[:min_vars_count]
    vars2 = vars2[:min_vars_count]

    # Генерируем все возможные перестановки сопоставления
    best_distance = float('inf')
    best_similarity = float(0)
    best_mapping = None
    best_path = None
    best_distance_dict = None

    # Все возможные сопоставления переменных
    for perm in permutations(vars2):
        mapping = dict(zip(vars1, perm))

        # Добавляем общие переменные (если есть)
        if common_vars:
            for var in common_vars:
                if var in vars1 and var in vars2:
                    mapping[var] = var

        # Вычисляем DTW с этим сопоставлением
        distance_dict = calculate_dtw_with_mapping(trace1, trace2, mapping)
        similarities = extract_similarities_from_distance_dict(distance_dict)
        avg_similarity = np.average(similarities)

        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_mapping = mapping
            best_distance_dict = distance_dict

    return best_mapping, best_distance_dict


def extract_similarities_from_distance_dict(distance_dict):
    similarities = []
    for key in distance_dict.keys():
        similarities.append(distance_dict[key]['similarity'])
    return similarities


def extract_distances_from_distance_dict(distance_dict):
    distances = []
    for key in distance_dict.keys():
        distances.append(distance_dict[key]['dist'])
    return distances


def trace_to_vectors(trace, var_order):
    """Преобразует трассу в числовые векторы"""
    vectors = []
    for state in trace:
        scalar_mem = state['memory']['scalar_memory']
        vector = []
        for var in var_order:
            if var in scalar_mem:
                vector.append(scalar_mem[var])
            else:
                vector.append(0)  # или другое значение по умолчанию
        vectors.append(vector)
    return np.array(vectors)


def calculate_dtw_with_mapping(trace1, trace2, variable_mapping):
    """
    Вычисляет DTW с заданным сопоставлением переменных
    """

    # Определяем порядок переменных для согласованного представления
    var_order = sorted(variable_mapping.keys())

    # Преобразуем трассы с учетом mapping
    vectors1 = trace_to_vectors(trace1, var_order)
    vectors2 = trace_to_vectors(trace2, [variable_mapping[var] for var in var_order])

    # Нормализация
    scaler = StandardScaler()
    all_vectors = np.vstack([vectors1, vectors2])
    scaler.fit(all_vectors)
    vectors1_normalized = scaler.transform(vectors1)
    vectors2_normalized = scaler.transform(vectors2)

    print("VECTOR1: ", vectors1)

    # Вычисляем DTW
    euclidean_distance, euclidean_path = fastdtw(vectors1, vectors2, dist=euclidean)

    # Вычисляем DTW
    chebyshev_distance, chebyshev_path = fastdtw(vectors1, vectors2, dist=chebyshev)

    levenshtein_dist = levenshtein_distance_between_vector_sequences(vectors1, vectors2)

    max_possible_distance = calculate_max_possible_distance(trace1, trace2)

    nonstutter_vectors1 = remove_stutter_steps(vectors1)
    nonstutter_vectors2 = remove_stutter_steps(vectors2)

    max_possible_nonstutter_distance = max(nonstutter_vectors1.size, nonstutter_vectors2.size) * 10

    print("NONSTUTTER VECTOR1: ", nonstutter_vectors1)

    # Вычисляем DTW
    nonstutter_euclidean_distance, nonstutter_euclidean_path = fastdtw(nonstutter_vectors1, nonstutter_vectors2,
                                                                       dist=euclidean)
    nonstutter_chebyshev_distance, nonstutter_chebyshev_path = fastdtw(nonstutter_vectors1, nonstutter_vectors2,
                                                                       dist=chebyshev)
    nonstutter_levenshtein_dist = levenshtein_distance_between_vector_sequences(nonstutter_vectors1,
                                                                                nonstutter_vectors2)

    dist_dict = {
        "euclidean": {"dist": euclidean_distance, "path": euclidean_path,
                      'similarity': calculate_similarity_by_distance(euclidean_distance, max_possible_distance)},
        "chebyshev": {"dist": chebyshev_distance, "path": chebyshev_path,
                      'similarity': calculate_similarity_by_distance(chebyshev_distance, max_possible_distance)},
        "nonstutter_euclidean": {"dist": nonstutter_euclidean_distance, "path": nonstutter_euclidean_path,
                                 'similarity': calculate_similarity_by_distance(nonstutter_euclidean_distance,
                                                                                max_possible_nonstutter_distance)},
        "nonstutter_chebyshev": {"dist": nonstutter_chebyshev_distance, "path": nonstutter_chebyshev_path,
                                 'similarity': calculate_similarity_by_distance(nonstutter_chebyshev_distance,
                                                                                max_possible_nonstutter_distance)},
        "Levenshtein": {"dist": levenshtein_dist,
                        'similarity': calculate_similarity_by_distance(levenshtein_dist, max_possible_distance)},
        "nonstutter_Levenshtein": {"dist": nonstutter_levenshtein_dist,
                                   'similarity': calculate_similarity_by_distance(nonstutter_levenshtein_dist,
                                                                                  max_possible_nonstutter_distance)},
        "2-gram": {'similarity': calculate_ngram_jaccard_similarity(vectors1, vectors2, 2) * 100},
        "3-gram": {'similarity': calculate_ngram_jaccard_similarity(vectors1, vectors2, 3) * 100},
        "5-gram": {'similarity': calculate_ngram_jaccard_similarity(vectors1, vectors2, 5) * 100}
    }
    return (dist_dict)
    # return euclidean_distance, euclidean_path




def calculate_similarity_by_distance(dist, max_possible_distance):
    return max(0, 100 - (dist / max_possible_distance * 100))


def generate_ngrams(vector_sequence: np.ndarray, n: int) -> Set[Tuple]:
    """
    Генерирует множество n-грамм из последовательности двумерных векторов.

    Каждая n-грамма представляется как кортеж кортежей,
    чтобы обеспечить хэшируемость для использования в множестве.

    Args:
        vector_sequence: Двумерный массив (последовательность векторов).
        n: Размер n-граммы.

    Returns:
        Множество уникальных n-грамм.
    """
    if len(vector_sequence) < n:
        return set()

    ngrams = set()

    # Итерируемся по всем возможным начальным индексам
    for i in range(len(vector_sequence) - n + 1):
        # Извлекаем n-грамму
        ngram_slice = vector_sequence[i:i + n - 1]

        # Преобразуем slice (np.ndarray) в кортеж кортежей для хэшируемости
        # Это позволяет нам поместить n-грамму во множество
        hashable_ngram = tuple(tuple(row) for row in ngram_slice)
        ngrams.add(hashable_ngram)

    return ngrams


def calculate_ngram_jaccard_similarity(
        vector_a: np.ndarray,
        vector_b: np.ndarray,
        n: int
) -> float:
    """
    Вычисляет схожесть Жаккара между двумя последовательностями векторов
    на основе их n-грамм.

    Args:
        vector_a: Первая последовательность векторов.
        vector_b: Вторая последовательность векторов.
        n: Размер n-граммы.

    Returns:
        Коэффициент схожести Жаккара (от 0.0 до 1.0).
    """
    # 1. Генерация множеств n-грамм
    ngrams_a = generate_ngrams(vector_a, n)
    ngrams_b = generate_ngrams(vector_b, n)

    # Если оба множества пусты (например, векторы слишком короткие), схожесть 1.0
    if not ngrams_a and not ngrams_b:
        return 1.0

    # 2. Вычисление Жаккара: |A ∩ B| / |A ∪ B|
    intersection = len(ngrams_a.intersection(ngrams_b))
    union = len(ngrams_a.union(ngrams_b))

    if union == 0:
        return 0.0

    jaccard_similarity = intersection / union
    return jaccard_similarity


def vector_to_unique_string(row: np.ndarray) -> str:
    """
    Преобразует двумерный вектор (одну строку np.array) в уникальную строку.
    Используем строковое представление элементов, разделенное уникальным
    разделителем (например, '|'), чтобы избежать путаницы в случае,
    если элементы имеют разную длину (например, 1 и 10).
    """
    # Преобразуем каждый элемент в строковый тип и объединяем
    return "|".join(map(str, row))


def levenshtein_distance_between_vector_sequences(vec1: np.ndarray, vec2: np.ndarray) -> int:
    """
    Вычисляет расстояние Левенштейна между двумя последовательностями
    двумерных векторов, где каждая строка рассматривается как один 'символ'.

    :param vec1: Первый двумерный вектор (массив N x M)
    :param vec2: Второй двумерный вектор (массив K x M)
    :return: Расстояние Левенштейна между последовательностями
    """

    # 1. Преобразование каждой строки в уникальный строковый "символ"
    seq1 = [vector_to_unique_string(row) for row in vec1]
    seq2 = [vector_to_unique_string(row) for row in vec2]

    # 2. Вычисление расстояния Левенштейна между двумя списками строк
    # Библиотека Levenshtein.distance работает только со строками.
    # Мы должны объединить наши "символы" обратно в одну большую строку,
    # используя разделитель, который гарантированно не встретится внутри
    # самих "символов".

    # Если мы используем уникальные строки из vector_to_unique_string,
    # то можем просто объединить их с помощью совершенно нового разделителя,
    # например, символа Unicode (хотя это излишне, если мы используем
    # стандартное сравнение списков, но для Levenshtein.distance нам нужна одна строка).

    # Простой и надежный способ для этого случая:
    # Преобразовать список "символов" в строку, используя уникальный разделитель.

    DELIMITER = "~~~"  # Разделитель, который не должен встречаться внутри элементов

    str1 = DELIMITER.join(seq1)
    str2 = DELIMITER.join(seq2)

    # 3. Вычисление расстояния
    distance = Levenshtein.distance(str1, str2)

    return distance


def calculate_similarity_percentage(trace1, trace2, common_vars=None):
    """
    Вычисляет процент схожести (0-100%) между двумя трассами
    """
    # Находим оптимальное сопоставление
    best_mapping, distance_dict = find_optimal_variable_mapping(trace1, trace2, common_vars)

    dict_mapping = {"best_mapping": best_mapping, "distance_dict": distance_dict}
    return dict_mapping


def calculate_max_possible_distance(trace1, trace2):
    """
    Оценивает максимально возможное расстояние между трассами такой же длины
    """
    # Простая эвристика: максимальное расстояние при полной несхожести
    # Можно настроить под вашу специфику данных
    max_len = max(len(trace1), len(trace2))
    return max_len * 10  # эмпирическая константа


from typing import List, Tuple


from typing import List, Tuple