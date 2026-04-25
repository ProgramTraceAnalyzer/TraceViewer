from similarity_matrix import *
import statistics

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

    for var1, matrix_row in agregated_matrix.items():
        if var1 not in agregated_matrix.keys():
            agregated_matrix[var1]={}
        for var2, similarity_list in matrix_row.items():
            if var2 not in agregated_matrix[var1].keys() and len(similarity_list)>0:
                agregated_matrix[var1][var2] = statistics.mean(similarity_list)
            else:
                agregated_matrix[var1][var2] = 0
    return agregated_matrix

