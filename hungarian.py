from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_mapping(
    similarity_matrix: Dict[str, Dict[str, float]],
    threshold: float,
    unmatched_value: float = -1e9,
) -> Dict[str, Any]:
    """
    Строит mapping: строка -> столбец
    Условия:
    - суммарная схожесть максимальна
    - пары с similarity < threshold запрещены

    Параметры:
    - similarity_matrix: dict[row][col] = similarity
    - threshold: минимально допустимая схожесть
    - unmatched_value: очень маленькое значение для запрещённых пар

    Возвращает:
    {
        "mapping": {row_name: col_name, ...},
        "pairs": [{"row": ..., "col": ..., "score": ...}, ...],
        "total_score": float,
        "unmatched_rows": [...],
        "unused_cols": [...]
    }
    """

    row_names = list(similarity_matrix.keys())

    col_names_set = set()
    for row in similarity_matrix.values():
        col_names_set.update(row.keys())
    col_names = list(col_names_set)

    n_rows = len(row_names)
    n_cols = len(col_names)
    size = max(n_rows, n_cols)

    score_matrix = np.full((size, size), unmatched_value, dtype=float)

    for i, row_name in enumerate(row_names):
        row_data = similarity_matrix.get(row_name, {})
        for j, col_name in enumerate(col_names):
            score = row_data.get(col_name, unmatched_value)
            if score >= threshold:
                score_matrix[i, j] = score

    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    mapping = {}
    pairs = []
    matched_rows = set()
    matched_cols = set()
    total_score = 0.0

    for i, j in zip(row_ind, col_ind):
        if i >= n_rows or j >= n_cols:
            continue

        score = score_matrix[i, j]
        if score < threshold:
            continue

        row_name = row_names[i]
        col_name = col_names[j]

        mapping[row_name] = col_name
        pairs.append({
            "row": row_name,
            "col": col_name,
            "score": float(score),
        })

        matched_rows.add(row_name)
        matched_cols.add(col_name)
        total_score += float(score)

    unmatched_rows = [r for r in row_names if r not in matched_rows]
    unused_cols = [c for c in col_names if c not in matched_cols]

    return {
        "mapping": mapping,
        "pairs": sorted(pairs, key=lambda x: x["score"], reverse=True),
        "total_score": total_score,
        "unmatched_rows": unmatched_rows,
        "unused_cols": unused_cols,
    }