import numpy as np
from typing import Union, List
from dot_processer import dot_file_to_adjacency_dict

def preprocess_sequence(list_values, nonstutter, reverse, variable_name, action_seq: List,
                               pidg_adj_matrix: dict, remove_unused = False):
    if len(list_values) == 0:
        return list_values
    if remove_unused:
        list_values = remove_not_used_var_values(list_values, variable_name, action_seq, pidg_adj_matrix)
    if nonstutter:
        list_values = remove_stutter_steps(list_values)
    if reverse:
        list_values = reverse_sequence(list_values)

    return list_values


def remove_not_used_var_values(value_seq: Union[np.ndarray, List[int]], variable_name, action_seq: List,
                               pidg_adj_matrix: dict):
    print("VALUE SEQ: ",value_seq, "type: ", type(value_seq))
    print("PIDG MATRIX: ",pidg_adj_matrix)
    print("ACTION SEQ: ", action_seq)
    print("VARIABLE NANE: ",variable_name)
    not_read_actions = []
    last_assign = -1
    first_assign = None
    for action_index in range(0, len(action_seq)):
        current_action = action_seq[action_index]
        print("CURRENT ACTION: ",current_action)
        if current_action["type"] == "assign" and current_action["assigned_variable"] == variable_name:
            read_pidg_nodes = None
            if action_index in pidg_adj_matrix.keys():
                read_pidg_nodes = pidg_adj_matrix[action_index]
            print("READ PIDG NODES: ",read_pidg_nodes)
            last_assign = action_index
            if first_assign is None:
                first_assign = action_index
            if read_pidg_nodes is None or len(read_pidg_nodes) == 0:
                not_read_actions.append(action_index)

    print("NOT READ ACTIONS: ", not_read_actions)
    print("LAST ASSIGN: ",last_assign)
    print("FIRST ASSIGN: ", first_assign)
    new_value_seq = []
    if first_assign is not None:
        for value_index in range(first_assign+1, len(value_seq)):
            print("VALUE INDEX ",value_index)
            if value_index-1 not in not_read_actions or last_assign == value_index-1:
                new_value_seq.append(int(value_seq[value_index]))
    print("NOT READ ACTIONS: ", not_read_actions)
    print("NEW VALUE SEQ: ", new_value_seq)
    return new_value_seq


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
