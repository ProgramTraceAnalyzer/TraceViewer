import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

state_sequence1 = []
state_sequence2 = []
var_mapping = {}

seq1_file = "file_examples/state_sequence1.json"
seq2_file = "file_examples/state_sequence2.json"


def remove_stutter_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет смежные дубликаты строк в DataFrame, корректно обрабатывая NaN.

    Смежные дубликаты - это строки, которые полностью идентичны непосредственно
    предшествующей строке. NaN значения считаются равными друг другу.
    Первая строка DataFrame всегда сохраняется.

    Args:
        df: Входной DataFrame, содержащий последовательность состояний.

    Returns:
        DataFrame с удаленными смежными дубликатами.
        Возвращает пустой DataFrame, если входной DataFrame пуст.
    """
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    # 1. Создаем булеву маску, которая будет True для строк, которые мы ХОТИМ СОХРАНИТЬ.
    #    Изначально предполагаем, что первую строку мы всегда сохраняем.
    rows_to_keep_mask = pd.Series(False, index=df.index)  # Изначально все False
    rows_to_keep_mask.iloc[0] = True  # Первая строка всегда True (сохраняем)

    # 2. Сдвигаем DataFrame на одну строку вниз для сравнения с предыдущей строкой.
    #    Исключаем первую строку из сдвинутого DataFrame, так как мы ее уже учли.
    df_shifted_for_comparison = df.shift(1).iloc[1:]

    # 3. Сравниваем строки, начиная со второй (индекс 1).
    #    Для этого сравниваем df.iloc[1:] с df_shifted_for_comparison.
    df_current_from_second = df.iloc[1:]

    # Создаем DataFrame для отслеживания идентичности каждого столбца
    rows_are_identical_per_column = pd.DataFrame(np.nan, index=df_current_from_second.index, columns=df.columns)

    for col in df.columns:
        # Получаем значения столбцов для текущей строки (начиная со второй) и предыдущей.
        current_col_values = df_current_from_second[col]
        shifted_col_values = df_shifted_for_comparison[col]

        # Идентичность в столбце достигается, если:
        # a) Оба значения NaN
        both_nan_mask = pd.isna(current_col_values) & pd.isna(shifted_col_values)
        # b) Оба значения НЕ NaN и равны
        not_nan_equal_mask = (~pd.isna(current_col_values)) & (~pd.isna(shifted_col_values)) & (
                    current_col_values == shifted_col_values)

        rows_are_identical_per_column[col] = both_nan_mask | not_nan_equal_mask

    # Строка (начиная со второй) является смежным дубликатом, если ВСЕ столбцы идентичны.
    is_consecutive_duplicate_from_second = rows_are_identical_per_column.all(axis=1)

    # Обновляем нашу маску: строки, которые НЕ являются смежными дубликатами (начиная со второй),
    # также должны быть сохранены.
    rows_to_keep_mask.loc[is_consecutive_duplicate_from_second.index] = ~is_consecutive_duplicate_from_second

    # Применяем маску для фильтрации DataFrame
    df_deduplicated = df.loc[rows_to_keep_mask]

    return df_deduplicated

def generate_dataframes(seq_file, variables):
    seq_df = pd.read_json(open(seq_file).read())
    scalar_memory_series = seq_df['memory'].apply(lambda x: x.get('scalar_memory'))
    scalar_memory_df = pd.DataFrame(scalar_memory_series.tolist())
    columns_to_keep = variables
    existing_columns = [col for col in columns_to_keep if col in scalar_memory_df.columns]
    df_only_variables = scalar_memory_df[existing_columns]
    df_deduplicated = remove_stutter_steps(df_only_variables)
    df_by_variables = {}
    df_by_variables_not_stutter = {}
    for v in variables:
        df_by_variables[v]=scalar_memory_df[v]
        df_by_variables_not_stutter[v] = remove_stutter_steps(pd.DataFrame(df_by_variables[v].tolist()))
    return seq_df, scalar_memory_df, df_only_variables, df_deduplicated, df_by_variables, df_by_variables_not_stutter

seq1_df, scalar_memory1_df, df1_only_variables, df_deduplicated, df_by_variables, df_by_variables_not_stutter = generate_dataframes(seq1_file,["current_side_A","current_side_B"])

plt.figure(figsize=(12, 6))

scalar_memory1_df.plot( y='current_side_A', label='Value A (Step X)', ax=plt.gca())
plt.show()
