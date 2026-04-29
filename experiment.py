from trace_builder import *
from similarity_matrix import *
import os
import json
import sys
from metrics import lcs, dtw
from matrix_agregation import *
from hungarian import *

experiment_folder = sys.argv[1]
task_config_filepath = sys.argv[2]
task_config = {}
function_name = ""
input_variables = {}
test_cases = []

with open(task_config_filepath, 'r', encoding='utf-8') as file:
    task_config = json.load(file)
    function_name = task_config["function_name"]
    input_variables = task_config["input_variables"]
    test_cases = task_config["test_cases"]

traces_paths = []

for root, dirs, files in os.walk(experiment_folder):
    for file in files:
        if file.endswith('.cpp'):
            cpp_path = os.path.join(experiment_folder, file)
            pg_path = os.path.join(experiment_folder, file + ".dot")
            build_PG(cpp_path, function_name, experiment_folder, pg_path)
            traces_path = os.path.join(experiment_folder, file + "_traces")
            traces_paths.append(traces_path)
            os.makedirs(traces_path, exist_ok=True)
            build_traces(pg_path, input_variables, test_cases, traces_path)

print(traces_paths)


# exit(0)

def calculate_mapping_for_trace_pair(prog1_path, prog2_path, treshhold, metrics, remove_stutter, remove_not_used):
    print(prog1_path, " ", prog2_path)
    test_count = len(test_cases)
    matrix_list = []
    for test_num in range(1, test_count + 1):
        prog1_test_path = os.path.join(prog1_path, str(test_num))
        prog2_test_path = os.path.join(prog2_path, str(test_num))
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


def process_traces(traces_paths, metrics, remove_stutter, remove_not_used):
    cross_prog_mapping = {}
    prog_count = len(traces_paths)
    print("PROG_COUNT: ", prog_count)
    for i in range(0, prog_count):
        prog1_path = traces_paths[i]
        if prog1_path not in cross_prog_mapping.keys():
            cross_prog_mapping[prog1_path] = {}
        for j in range(i + 1, prog_count):
            prog2_path = traces_paths[j]
            print("i=", i, " j=", j)
            mapping = calculate_mapping_for_trace_pair(prog1_path, prog2_path, 60, metrics, remove_stutter, remove_not_used)
            print("MAPPING")
            print(mapping["mapping"])
            if prog2_path not in cross_prog_mapping[prog1_path].keys():
                cross_prog_mapping[prog1_path][prog2_path] = mapping["mapping"]
    return cross_prog_mapping


cross_prog_mapping = process_traces(traces_paths, dtw, False, False)

with open("mapping.json", 'w', encoding='utf-8') as file:
    json.dump(cross_prog_mapping, file)
