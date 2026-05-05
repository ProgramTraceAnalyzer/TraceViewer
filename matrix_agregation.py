from similarity_matrix import *
import statistics
from typing import Dict, Any, List, Tuple

def agregate_matrixes(matrix_list):
    agregated_matrix = {}
    pair_values = {}
    for matrix in matrix_list: # For each matrix
        for var1, neighbor_dict in matrix.items(): # For each row
            if var1 not in pair_values.keys():
                pair_values[var1] = {}
            for var2, value in neighbor_dict.items(): # For each column
                if var2 not in pair_values[var1].keys():
                    pair_values[var1][var2] = []
                pair_values[var1][var2].append(value["similarity"])

    for var1, matrix_row in pair_values.items():
        if var1 not in agregated_matrix.keys():
            agregated_matrix[var1]={}
        for var2, similarity_list in matrix_row.items():
            if var2 not in agregated_matrix[var1].keys() and len(similarity_list)>0:
                agregated_matrix[var1][var2] = statistics.mean(similarity_list)
            else:
                agregated_matrix[var1][var2] = 0
    return agregated_matrix


def get_only_similarity_matrix(matrix) -> Dict[str, Dict[str, float]]:
    pair_values = {}
    for var1, neighbor_dict in matrix.items():  # For each row
        if var1 not in pair_values.keys():
            pair_values[var1] = {}
        for var2, value in neighbor_dict.items():  # For each column
            if var2 not in pair_values[var1].keys():
                pair_values[var1][var2] = value["similarity"]
    return pair_values


from typing import List, Dict
from collections import defaultdict

def get_pair_statistics(dict_list: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    if not dict_list:
        return {}

    total = len(dict_list)
    stats = defaultdict(lambda: defaultdict(int))

    # Считаем, сколько раз у каждого ключа встречается каждое значение
    for d in dict_list:
        for k, v in d.items():
            stats[k][v] += 1

    # Переводим счётчики в проценты
    result: Dict[str, Dict[str, float]] = {}
    for k, values in stats.items():
        result[k] = {v: count * 100 / total for v, count in values.items()}

    return result