import ast
import subprocess
from collections import Mapping
from typing import List, Tuple, Set, Optional
import pandas as pd
import numpy as np
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

script_path = Path(__file__).resolve()
script_dir = script_path.parent

pg_builder_path = os.path.join(script_dir, "PG_builder.py")
trace_builder = os.path.join(script_dir, "ProgramGraphAnalysis.exe")
tmp_files = "tmp_files"
traces = "traces"


def build_PG(cpp_file, function_name, pg_dest_path, pg_filename):
    os.makedirs(pg_dest_path, exist_ok=True)
    pg_dot_path = os.path.join(pg_dest_path, pg_filename)
    subprocess.run(['python', pg_builder_path, cpp_file, function_name, pg_dot_path], cwd=script_dir)
    return True


def build_traces(pg_file, input_variables, test_cases, traces_path):
    print("build tests...")
    os.makedirs(traces_path, exist_ok=True)
    test_num = 0
    for test in test_cases:
        test_num += 1
        test_dir = os.path.join(traces_path, str(test_num))
        print(test_dir)
        os.makedirs(test_dir, exist_ok=True)
        arg_list = [trace_builder, pg_file]
        input_data = test["data"]
        for input_var_name in input_data:
            if input_variables[input_var_name]["type"] == "scalar":
                arg_list += ("--scalar_var", input_var_name, str(input_data[input_var_name]))
            if input_variables[input_var_name]["type"] == "array":
                arg_list += ("--array_var", input_var_name)
                arg_list.append(str(input_variables[input_var_name]["size"]))
                arg_list += [str(el) for el in input_data[input_var_name]]
        # os.makedirs(attempt_dir, exist_ok=True)
        print(test_dir)
        print(arg_list)
        subprocess.run(arg_list, cwd=test_dir)
    return True
