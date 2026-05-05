from trace_builder import *
from similarity_matrix import *
import os
import json
import sys
from metrics import lcs, dtw
from matrix_agregation import *
from hungarian import *



def calculate_mapping_for_trace_pair(prog1_path, prog2_path, treshhold, metrics, remove_stutter, remove_not_used):
    print(prog1_path, " ", prog2_path)
    traces1_folder = prog1_path#os.path.join(prog1_path,"traces")
    traces2_folder = prog2_path#os.path.join(prog2_path,"traces")
    matrix_list = []
    for test_num in os.listdir(traces1_folder):
        print("CURRENT TEST NUM: ",str(test_num))
        prog1_test_path = os.path.join(traces1_folder, str(test_num))
        prog2_test_path = os.path.join(traces2_folder, str(test_num))
        prog1_state_seq = os.path.join(prog1_test_path, "state_sequence.json")
        prog2_state_seq = os.path.join(prog2_test_path, "state_sequence.json")
        prog1_act_seq = os.path.join(prog1_test_path, "action_sequence.json")
        prog2_act_seq = os.path.join(prog2_test_path, "action_sequence.json")
        prog1_pidg = os.path.join(prog1_test_path, "PIDG.dot")
        prog2_pidg = os.path.join(prog2_test_path, "PIDG.dot")
        matrix = calculate_var_distances(prog1_state_seq, prog2_state_seq, metrics, -1, remove_stutter, False,
                                         prog1_act_seq,
                                         prog2_act_seq, prog1_pidg, prog2_pidg, remove_not_used)
        matrix_list.append(matrix)
    agg = agregate_matrixes(matrix_list)
    mapping = hungarian_mapping(agg, treshhold)
    return mapping

def calculate_mapping_statistics_for_traces(prog1_path, prog2_path, treshhold, metrics, remove_stutter, remove_not_used):
    traces1_folder = prog1_path  # os.path.join(prog1_path,"traces")
    traces2_folder = prog2_path  # os.path.join(prog2_path,"traces")
    #matrix_list = []
    mapping_list = []
    for test_num in os.listdir(traces1_folder):
        print("CURRENT TEST NUM: ", str(test_num))
        prog1_test_path = os.path.join(traces1_folder, str(test_num))
        prog2_test_path = os.path.join(traces2_folder, str(test_num))
        prog1_state_seq = os.path.join(prog1_test_path, "state_sequence.json")
        prog2_state_seq = os.path.join(prog2_test_path, "state_sequence.json")
        prog1_act_seq = os.path.join(prog1_test_path, "action_sequence.json")
        prog2_act_seq = os.path.join(prog2_test_path, "action_sequence.json")
        prog1_pidg = os.path.join(prog1_test_path, "PIDG.dot")
        prog2_pidg = os.path.join(prog2_test_path, "PIDG.dot")
        matrix = calculate_var_distances(prog1_state_seq, prog2_state_seq, metrics, -1, remove_stutter, False,
                                         prog1_act_seq,
                                         prog2_act_seq, prog1_pidg, prog2_pidg, remove_not_used)
        similarity_matrix = get_only_similarity_matrix(matrix)
        print("SIMILARITY MATRIX:",similarity_matrix)
        mapping = hungarian_mapping(similarity_matrix,treshhold)
        mapping_list.append(mapping["mapping"])

    statistics = get_pair_statistics(mapping_list)
    return statistics
        #matrix_list.append(matrix)
    #agg = agregate_matrixes(matrix_list)
    #mapping = hungarian_mapping(agg, treshhold)
    #return mapping