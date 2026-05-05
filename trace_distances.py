import ast
import subprocess
from collections import Mapping
from typing import List, Tuple, Set, Optional
import pandas as pd
import numpy as np
from itertools import permutations, combinations

from PyQt5.QtGui import QColor
from fastdtw import fastdtw
from matplotlib.patches import Rectangle
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

from similarity_matrix import *

from trace_builder import *


from variable_mapping import calculate_mapping_for_trace_pair, calculate_mapping_statistics_for_traces

import json




import sys

root_folder = r"D:\Гугл-Диск\КулюкинКС_кандидатская\Сравнение трасс программ\Наши разработки\Эксперименты\Эксперимент_май2025\3\CppSolutions"
import os

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PairStatCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 4))
        super().__init__(self.figure)
        self.setParent(parent)

def generate_dtw_variables_report(root_folder, report_file):
    trace_files = {}

    for student in os.listdir(root_folder):
        student_folder = os.path.join(root_folder, student)
        if os.path.isdir(student_folder):
            for attempt_num in os.listdir(student_folder):
                attempt_folder = os.path.join(student_folder, attempt_num)
                if os.path.isdir(attempt_folder):
                    for test_num in os.listdir(attempt_folder):
                        test_folder = os.path.join(attempt_folder, test_num)
                        if os.path.isdir(test_folder):
                            trace_file = os.path.join(test_folder, 'state_sequence.json')
                            if test_num not in trace_files.keys():
                                trace_files[test_num] = []
                            trace_files[test_num].append(trace_file)

    report = {}
    common_vars = ['side_A', 'side_B']
    for test_num in trace_files.keys():
        report[test_num] = {}
        traces_for_test = trace_files[test_num]
        length = len(traces_for_test)
        for i in range(0, length):
            trace_file1 = traces_for_test[i]
            report[test_num][trace_file1] = {}
            for j in range(0, length):
                trace_file2 = traces_for_test[j]
                try:
                    dtws = calculate_var_distances(trace_file1, trace_file2, -1)
                    report[test_num][trace_file1][trace_file2] = dtws
                    # sim = calculate_similary_two_json_traces(trace_file1,trace_file2,common_vars)
                    # report[test_num][trace_file1][trace_file2]=sim
                except:
                    print("error: ", trace_file1, " ", trace_file2)

    with open(report_file, 'w', encoding='utf-8') as file:
        json.dump(report, file)


# calculate_similary_two_json_traces(sys.argv[1],sys.argv[2],['side_A', 'side_B'])


def visual_pair_statistics(pair_statistics, plot_widget):
    fig = Figure(figsize=(10, max(3, len(pair_statistics) * 1.6)), facecolor="white")
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    ax.set_facecolor("white")
    ax.axis("off")

    label_x = 8
    start_x = 20
    max_width = 80
    gap = 3
    bar_height = 0.5
    row_gap = 1.8

    keys = list(pair_statistics.keys())
    total_rows_height = (len(keys) - 1) * row_gap

    for i, key in enumerate(keys):
        y = total_rows_height - i * row_gap
        ax.text(label_x, y, str(key), fontsize=26, ha="center", va="center")

        x = start_x
        items = sorted(pair_statistics[key].items(), key=lambda item: item[1])

        for value, percent in items:
            width = max_width * (percent / 100.0)

            ax.add_patch(Rectangle(
                (x, y - bar_height / 2),
                width,
                bar_height,
                facecolor="#a8d8e8",
                edgecolor="black",
                linewidth=2
            ))

            ax.text(
                x + width / 2,
                y - bar_height / 2 - 0.12,
                f"{value} ({percent:.0f}%)",
                fontsize=18,
                ha="center",
                va="top"
            )

            x += width + gap

    ax.set_xlim(0, start_x + max_width + 20)
    ax.set_ylim(-0.8, total_rows_height + 0.8)

    fig.tight_layout()

    layout = plot_widget.layout()
    if layout is None:
        layout = QVBoxLayout()
        plot_widget.setLayout(layout)

    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()

    layout.addWidget(canvas)
    canvas.draw()


from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem,
    QLabel, QMessageBox, QTextEdit, QComboBox, QTabWidget, QSplitter, QCheckBox
)
from PyQt5.QtCore import pyqtSlot, Qt

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sequence Matrix Builder")

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main vertical splitter
        main_splitter = QSplitter(Qt.Vertical)

        # ---- Upper section: File selectors (seq1, seq2, code1, code2) ----
        upper_splitter = QSplitter(Qt.Horizontal)

        # seq1 and seq2 group
        seq_group = QWidget()
        seq_layout = QHBoxLayout(seq_group)
        seq_layout.addWidget(QLabel("seq1_file:"))
        self.seq1_file_edit = QLineEdit()
        self.seq1_file_edit.setReadOnly(True)
        seq_layout.addWidget(self.seq1_file_edit)
        self.seq1_button = QPushButton("Browse seq1.json")
        self.seq1_button.clicked.connect(self.select_seq1_file)
        seq_layout.addWidget(self.seq1_button)

        seq_layout.addWidget(QLabel("seq2_file:"))
        self.seq2_file_edit = QLineEdit()
        self.seq2_file_edit.setReadOnly(True)
        seq_layout.addWidget(self.seq2_file_edit)
        self.seq2_button = QPushButton("Browse seq2.json")
        self.seq2_button.clicked.connect(self.select_seq2_file)
        seq_layout.addWidget(self.seq2_button)

        # code1 and code2 group
        code_group = QWidget()
        code_layout = QHBoxLayout(code_group)
        code_layout.addWidget(QLabel("code1_file:"))
        self.code1_file_edit = QLineEdit()
        self.code1_file_edit.setReadOnly(True)
        code_layout.addWidget(self.code1_file_edit)
        self.code1_button = QPushButton("Browse code1.cpp")
        self.code1_button.clicked.connect(self.select_code1_file)
        code_layout.addWidget(self.code1_button)

        code_layout.addWidget(QLabel("code2_file:"))
        self.code2_file_edit = QLineEdit()
        self.code2_file_edit.setReadOnly(True)
        code_layout.addWidget(self.code2_file_edit)
        self.code2_button = QPushButton("Browse code2.cpp")
        self.code2_button.clicked.connect(self.select_code2_file)
        code_layout.addWidget(self.code2_button)

        #upper_splitter.addWidget(seq_group)
        upper_splitter.addWidget(code_group)
        upper_splitter.setSizes([1, 1])  # Equal initial sizes

        main_splitter.addWidget(upper_splitter)

        # ---- Middle section: Code views ----
        code_views_widget = QWidget()
        code_views_layout = QHBoxLayout(code_views_widget)

        code1_layout = QVBoxLayout()
        self.code1_view_text = QTextEdit()
        button_recompile_code1 = QPushButton("Обновить")
        button_recompile_code1.clicked.connect(self.update_code1)
        code1_layout.addWidget(self.code1_view_text)
        code1_layout.addWidget(button_recompile_code1)

        code2_layout = QVBoxLayout()
        self.code2_view_text = QTextEdit()
        button_recompile_code2 = QPushButton("Обновить")
        button_recompile_code2.clicked.connect(self.update_code2)
        code2_layout.addWidget(self.code2_view_text)
        code2_layout.addWidget(button_recompile_code2)

        code_views_layout.addLayout(code1_layout)
        code_views_layout.addLayout(code2_layout)

        #self.code1_view_text.textChanged.connect(self.update_code1)
        #self.code2_view_text.textChanged.connect(self.update_code2)
        #code_views_layout.addWidget(self.code1_view_text)
        #code_views_layout.addWidget(self.code2_view_text)
        main_splitter.addWidget(code_views_widget)



        # ---- Lower section: Task config, Test select, Data type, Table ----
        lower_widget = QWidget()
        lower_layout = QVBoxLayout(lower_widget)

        matrix_layout = QHBoxLayout()

        # Table widget
        self.var_matrix = QTableWidget()
        self.var_matrix.setRowCount(0)
        self.var_matrix.setColumnCount(0)
        self.var_matrix.setHorizontalHeaderLabels(["Column 1", "Column 2"])
        self.var_matrix.cellClicked.connect(self.on_cell_clicked)
        matrix_layout.addWidget(self.var_matrix)

        task_layout = QVBoxLayout()

        # Task config
        self.task_config_file_edit = QLineEdit()
        self.task_config_push_button = QPushButton("Выбрать файл...")
        self.task_config_push_button.clicked.connect(self.select_task_config)
        task_config_layout = QHBoxLayout()
        task_config_layout.addWidget(QLabel("Файл конфигурации задания"))
        task_config_layout.addWidget(self.task_config_file_edit)
        task_config_layout.addWidget(self.task_config_push_button)
        task_layout.addLayout(task_config_layout)

        # Test select
        self.test_select = QComboBox()
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Выберите тест:"))
        test_layout.addWidget(self.test_select)
        self.test_select.currentIndexChanged.connect(self._handle_test_selection)
        task_layout.addLayout(test_layout)

        # Data type select
        self.data_type_select = QComboBox()
        data_type_layout = QHBoxLayout()
        data_type_layout.addWidget(QLabel("Выберите данные для построения матрицы: "))
        data_type_layout.addWidget(self.data_type_select)
        self.data_type_select.addItem("Значения переменных")
        self.data_type_select.addItem("Чтение переменных")
        self.VALUE_SEQUENCE = 0
        self.READ_SEQUENCE = 1
        self.data_type_select.currentIndexChanged.connect(self._data_type_changed)
        task_layout.addLayout(data_type_layout)

        # ---- Metric select widget ----
        self.metric_select = QComboBox()
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Метрика:"))
        metric_layout.addWidget(self.metric_select)
        self.metrics_dict = {"LCS": lcs, "Smith-Waterman": smith_waterman, "DTW": dtw}
        self.fill_metrcis_select()
        self.metric_select.currentIndexChanged.connect(self._metric_changed)
        task_layout.addLayout(metric_layout)

        # ---- Stutter step widget ----
        self.stutter_step_checkbox = QCheckBox()
        stutter_step_layout = QHBoxLayout()
        stutter_step_layout.addWidget(QLabel("Убрать заикания: "))
        stutter_step_layout.addWidget(self.stutter_step_checkbox)
        self.stutter_step_checkbox.stateChanged.connect(self.on_checkbox_changed)
        task_layout.addLayout(stutter_step_layout)

        # ----- reverse sequence widget ----
        self.reverse_checkbox = QCheckBox()
        reverse_layout = QHBoxLayout()
        reverse_layout.addWidget(QLabel("Перевернуть последовательность:"))
        reverse_layout.addWidget(self.reverse_checkbox)
        self.reverse_checkbox.stateChanged.connect(self.on_reverse_checkbox_changed)
        task_layout.addLayout(reverse_layout)

        # ----- remove unused widget ----
        self.not_readable_checkbox = QCheckBox()
        not_readable_layout = QHBoxLayout()
        not_readable_layout.addWidget(QLabel("Убрать нечитаемые значения:"))
        not_readable_layout.addWidget(self.not_readable_checkbox)
        self.not_readable_checkbox.stateChanged.connect(self.on_not_readable_checkbox_changed)
        task_layout.addLayout(not_readable_layout)

        matrix_layout.addLayout(task_layout)
        lower_layout.addLayout(matrix_layout)


        main_splitter.addWidget(lower_widget)

        # Set initial sizes (adjust as needed)
        main_splitter.setSizes([100, 200, 300])  # Upper, Middle, Lower

        # Set main layout
        layout = QVBoxLayout(central)
        layout.addWidget(main_splitter)
        central.setLayout(layout)


        # ---- Sequences Layout ----
        seqX = [1, 3, 4, 9, 12, 15]
        seqY = [2, 5, 8, 10, 14]
        path = [(0, 0), (1, 2), (2, 2), (3, 3), (4, 4)]
        matrix = np.random.rand(len(seqX), len(seqY)) * 10

        self.tab_widget = QTabWidget()
        self.seq_view_layout = SequenceViewSplitter(seqX, seqY, path, matrix)
        self.tab_widget.addTab(self.seq_view_layout,"Sequence View")
        self.var_action_history_code1 = VariableActionHistorySplitter()
        #self.var_action_history_code1.update_by_files(self.seq1_path, self.act_seq1_path)
        self.var_action_history_code2 = VariableActionHistorySplitter()
        #self.var_action_history_code2.update_by_files(self.seq2_path, self.act_seq2_path)
        splitter_action_variables = QSplitter()
        splitter_action_variables.setOrientation(Qt.Horizontal)
        splitter_action_variables.addWidget(self.var_action_history_code1)
        splitter_action_variables.addWidget(self.var_action_history_code2)
        self.tab_widget.addTab(splitter_action_variables, "Action Sequence")

        splitter_mapping = QSplitter()
        splitter_mapping.setOrientation(Qt.Vertical)
        self.mapping_table_widget = QTableWidget()
        self.mapping_table_widget.setColumnCount(2)
        self.mapping_table_widget.setHorizontalHeaderLabels(["Прог 1", "Прог 2"])

        mapping_btn = QPushButton("Обновить mapping")
        mapping_btn.clicked.connect(self.update_mapping)
        splitter_mapping.addWidget(self.mapping_table_widget)

        # Контейнер под график
        self.pair_stat_plot = QWidget()
        self.pair_stat_plot_layout = QVBoxLayout(self.pair_stat_plot)

        # Сам canvas matplotlib
        self.canvas_pair_stat = PairStatCanvas(self)

        # Добавляем canvas в контейнер
        self.pair_stat_plot_layout.addWidget(self.canvas_pair_stat)

        splitter_mapping.addWidget(self.pair_stat_plot)

        splitter_mapping.addWidget(mapping_btn)
        self.tab_widget.addTab(splitter_mapping, "Mapping")


        main_splitter.addWidget(self.tab_widget)



        # ---- State ----
        self.seq1_path = None
        self.seq2_path = None

        self.code_files = ["",""]
        self.traces_path = ["",""]
        self.traces_path[0] = os.path.join(script_dir, os.path.join(tmp_files, os.path.join("0","traces")))
        self.traces_path[1] = os.path.join(script_dir, os.path.join(tmp_files, os.path.join("1", "traces")))
        self.code_files[0] = os.path.join(script_dir, os.path.join(tmp_files, os.path.join("0","code.cpp")))
        self.code_files[1] = os.path.join(script_dir, os.path.join(tmp_files, os.path.join("1", "code.cpp")))
        self.task_config_path = ""
        self.task_config = {}
        self.currentMatrixRow = None
        self.currentMatrixCol = None

    def fill_metrcis_select(self):
        self.metrics_functions = []
        for metric_name, metric_function in self.metrics_dict.items():
            self.metric_select.addItem(metric_name)
            self.metrics_functions.append(metric_function)

    # ----------------------------------------------------------------------
    # File selection slots
    # ----------------------------------------------------------------------
    @pyqtSlot()
    def select_seq1_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select seq1 JSON file",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self.seq1_path = path
            self.seq1_file_edit.setText(path)
            self.check_files_and_build()

    @pyqtSlot()
    def select_seq2_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select seq2 JSON file",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self.seq2_path = path
            self.seq2_file_edit.setText(path)
            self.check_files_and_build()

    def select_code1_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select seq1 cpp - file",
            str(Path.home()),
            "JSON Files (*.cpp);;All Files (*)"
        )
        if path:
            self.code1_path = path
            self.code1_file_edit.setText(path)
            #self.code_files[0] = path
            with open(path, 'r') as f:
                self.code1_view_text.setText(f.read())
            self.update_code1()
            #self.build_all_pg_and_traces()


    def select_code2_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select seq1 cpp - file",
            str(Path.home()),
            "JSON Files (*.cpp);;All Files (*)"
        )
        if path:
            self.code2_path = path
            # self.code_files[1] = path
            self.code2_file_edit.setText(path)
            with open(path, 'r') as f:
                self.code2_view_text.setText(f.read())
            self.update_code2()
            #self.build_all_pg_and_traces()

    def update_views(self):
        self.build_matrix()
        self.reshow_current_cell_plots()
        print("UPDATE VIEWS")
        self.var_action_history_code1.update_by_files(self.seq1_path,self.act_seq1_path, self.read_seq1_path)
        self.var_action_history_code2.update_by_files(self.seq2_path, self.act_seq2_path, self.read_seq2_path)

    def update_code1(self):
        code = self.code1_view_text.toPlainText()
        with open(self.code_files[0], 'w') as f:
            f.write(code)
        self.build_all_pg_and_traces_and_update_views()

    def update_code2(self):
        code = self.code2_view_text.toPlainText()
        with open(self.code_files[1], 'w') as f:
            f.write(code)
        self.build_all_pg_and_traces_and_update_views()

    def set_code_file_text(self):
        pass

    def select_task_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select task config json file",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self.task_config_path = path
            self.task_config_file_edit.setText(path)
            self.read_task_config(path)
            self.fill_test_num((self.task_config["test_cases"]))
            self.build_all_pg_and_traces()

    def _data_type_changed(self):
        self.update_views()

    def _metric_changed(self):
        self.update_views()


    def reshow_current_cell_plots(self):
        if self.currentMatrixRow is not None and self.currentMatrixCol is not None:
            self.var_matrix.setCurrentCell(self.currentMatrixRow, self.currentMatrixCol)
            self.show_for_cell(self.currentMatrixRow, self.currentMatrixCol)

    def on_checkbox_changed(self):
        self.update_views()

    def on_reverse_checkbox_changed(self):
        self.update_views()

    def on_not_readable_checkbox_changed(self):
        self.update_views()


    def _handle_test_selection(self, index: int):
        """
        Обработчик сигнала currentIndexChanged.
        Если выбран не нулевой индекс (т.е., выбран реальный тест),
        удаляем элемент с индексом 0 ("Выберите тест").
        """



        print("Выбран тест ", self.test_select.currentIndex())
        self.seq_view_layout.set_title(self.generate_test_case_str(self.task_config["test_cases"][self.test_select.currentIndex()]))
        tmp_files_dir = os.path.join(script_dir, tmp_files)
        self.seq1_path = os.path.join(os.path.join(os.path.join(os.path.join(tmp_files_dir, "0"), traces),
                                                   str(self.test_select.currentIndex() + 1)), "state_sequence.json")
        self.seq2_path = os.path.join(os.path.join(os.path.join(os.path.join(tmp_files_dir, "1"), traces),
                                                   str(self.test_select.currentIndex() + 1)), "state_sequence.json")

        self.act_seq1_path = os.path.join(os.path.join(os.path.join(os.path.join(tmp_files_dir, "0"), traces),
                                                   str(self.test_select.currentIndex() + 1)), "action_sequence.json")
        self.act_seq2_path = os.path.join(os.path.join(os.path.join(os.path.join(tmp_files_dir, "1"), traces),
                                                   str(self.test_select.currentIndex() + 1)), "action_sequence.json")

        self.pidg1_path = os.path.join(os.path.join(os.path.join(os.path.join(tmp_files_dir, "0"), traces),
                                                   str(self.test_select.currentIndex() + 1)), "PIDG.dot")
        self.pidg2_path = os.path.join(os.path.join(os.path.join(os.path.join(tmp_files_dir, "1"), traces),
                                                   str(self.test_select.currentIndex() + 1)), "PIDG.dot")

        self.read_seq1_path = os.path.join(os.path.join(os.path.join(os.path.join(tmp_files_dir, "0"), traces),
                                                   str(self.test_select.currentIndex() + 1)), "read_var_sequence.json")
        self.read_seq2_path = os.path.join(os.path.join(os.path.join(os.path.join(tmp_files_dir, "1"), traces),
                                                        str(self.test_select.currentIndex() + 1)),
                                           "read_var_sequence.json")
        print("SEQ1: ", self.seq1_path)
        print("SEQ2: ", self.seq2_path)
        self.update_views()

    def generate_test_case_str(self, test_case) -> str:
        res = test_case["name"] + "\n"
        for var_name, var_value in test_case["data"].items():
            res += str(var_name) + "=" + str(var_value) + "\n"
        return res

    def fill_test_num(self, test_cases):
        self.test_select.clear()
        #self.test_select.addItem("Выберите тест")
        count = len(test_cases)
        for i in range(0, count):
            print("11111 FILL TEST NUM")
            test_case_str = self.generate_test_case_str(test_cases[i])
            print("FILL TEST NUM")
            self.test_select.addItem(str(i+1)+" "+test_case_str)

        self.test_select.setCurrentIndex(0)

    # ----------------------------------------------------------------------
    # Helper to determine if both files are selected
    # ----------------------------------------------------------------------
    def check_files_and_build(self):
        if self.seq1_path and self.seq2_path:
            self.build_matrix()

    # ----------------------------------------------------------------------
    # Slot that is called when both files are selected
    # ----------------------------------------------------------------------
    @pyqtSlot()
    def build_matrix(self):
        metrics_index_selected = self.metric_select.currentIndex()
        allignment_function = self.metrics_functions[metrics_index_selected]
        matrix = None
        non_stutter = self.stutter_step_checkbox.isChecked()
        reverse = self.reverse_checkbox.isChecked()
        remove_unused = self.not_readable_checkbox.isChecked()
        if self.data_type_select.currentIndex() == self.VALUE_SEQUENCE:
            print("self.VALUE_SEQUENCE")
            matrix = calculate_var_distances(self.seq1_path, self.seq2_path, allignment_function, -1, non_stutter, reverse, self.act_seq1_path, self.act_seq2_path, self.pidg1_path, self.pidg2_path, remove_unused)
        elif self.data_type_select.currentIndex() == self.READ_SEQUENCE:
            print("self.READ_SEQUENCE")
            matrix = calculate_read_var_distances(self.read_seq1_path, self.read_seq2_path, allignment_function, reverse)
        else:
            return
        print("BUILD MATRIX...")
        self.matrix = matrix
        print(matrix)
        # 1. Составляем списки строк и столбцов
        rows = sorted(matrix.keys())
        cols = sorted({k for inner in matrix.values() for k in inner.keys()})

        # 2. Настраиваем таблицу
        self.var_matrix.clearContents()
        self.var_matrix.setRowCount(len(rows))
        self.var_matrix.setColumnCount(len(cols))
        self.var_matrix.setVerticalHeaderLabels(rows)
        self.var_matrix.setHorizontalHeaderLabels(cols)

        # 3. Заполняем ячейки
        for r, row_key in enumerate(rows):
            for c, col_key in enumerate(cols):
                # Если в матрице нет значения, оставляем ячейку пустой
                value = matrix[row_key].get(col_key, "")
                similarity = value["similarity"]
                item = QTableWidgetItem("{similarity:.2f}%".format(similarity=similarity))
                item.setBackground(self.get_color_for_similarity(similarity))
                item.setData(Qt.UserRole, value)
                self.var_matrix.setItem(r, c, item)

    # Функция для вычисления цвета на основе similarity (0.0 - 1.0)
    def get_color_for_similarity(self, similarity):
        # Нормализуем в диапазон [0.0, 1.0]
        normalized = similarity / 100.0
        if normalized <= 0.0:
            return QColor(255, 200, 200)  # бледно-красный
        elif normalized >= 1.0:
            return QColor(200, 255, 200)  # бледно-зелёный
        elif normalized <= 0.5:
            # Градиент от красного (255,200,200) к желтому (255,255,180)
            t = normalized * 2  # [0.0, 1.0] в сегменте 0–50%
            r = 255
            g = int(200 + (255 - 200) * t)  # 200 → 255
            b = int(200 + (180 - 200) * t)  # 200 → 180
            return QColor(r, g, b)
        else:
            # Градиент от желтого (255,255,180) к зелёному (200,255,200)
            t = (normalized - 0.5) * 2  # [0.0, 1.0] в сегменте 50–100%
            r = int(255 - (255 - 200) * t)  # 255 → 200
            g = 255
            b = int(180 + (200 - 180) * t)  # 180 → 200
            return QColor(r, g, b)

    @pyqtSlot(int, int)
    def on_cell_clicked(self, row: int, column: int):
        """
        Слот, вызываемый при клике на ячейку.
        """
        self.currentMatrixRow = row
        self.currentMatrixCol = column
        self.show_for_cell(row,column)


    def show_for_cell(self, row: int, column: int):
        item = self.var_matrix.item(row, column)
        row_header = self.var_matrix.verticalHeaderItem(row).text()
        column_header = self.var_matrix.horizontalHeaderItem(column).text()



        # Если ячейка пустая – ничего не делаем
        if item is None or item.text() == "":
            QMessageBox.information(self, "Пустая ячейка",
                                    "Вы нажали на пустую ячейку.")
            return

        # Получаем сохранённый словарь
        value = item.data(Qt.UserRole)
        if not isinstance(value, dict):
            # На случай, если что‑то пошло не так
            QMessageBox.warning(self, "Ошибка",
                                "В ячейке нет корректных данных.")
            return

        # Вызываем ваш метод с нужными параметрами
        list1 = value.get("row_list_values")
        list2 = value.get("col_list_values")
        matrix = value.get("matrix")
        path = value.get("path")
        #similarity = value.get("similarity")

        # metrics_index_selected = self.metric_select.currentIndex()
        # matrix, path, similarity = self.metrics_functions[metrics_index_selected](list1, list2)
        self.seq_view_layout.update(list1,list2,path,matrix, row_header, column_header)
        self.var_action_history_code1.set_current_variable(row_header)
        self.var_action_history_code2.set_current_variable(column_header)

    def show_plot(self, a, b, a_header="Ряд 1", b_header="Ряд 2"):
        # Гарантируем 1D float-массивы
        a_np = np.array(a, dtype=float).flatten()
        b_np = np.array(b, dtype=float).flatten()

        # Проверка
        if a_np.ndim != 1:
            raise ValueError(f"a must be 1D, got shape {a_np.shape}")
        if b_np.ndim != 1:
            raise ValueError(f"b must be 1D, got shape {b_np.shape}")

        # Используем safe_euclidean вместо scipy.spatial.distance.euclidean
        distance, path = fastdtw(a_np, b_np, dist=safe_euclidean)

        print("PATH: ",path)

        # Визуализация
        # 3. Создаём новый Figure (или очищаем старый)
        self.plot_canvas.figure.clear()  # удаляем старый рисунок
        ax = self.plot_canvas.figure.add_subplot(111)

        ax.plot(a_np, label=a_header, color='blue', linewidth=4, marker='o', markersize=5)
        ax.plot(b_np, label=b_header, color='red', linewidth=4, marker='s', markersize=5)

        for i, j in path:
            ax.plot([i, j], [a_np[i], b_np[j]], color='green', linestyle='--', alpha=0.5, linewidth=1.8)

        ax.set_title(f'FastDTW: расстояние = {distance:.6f}', fontsize=16)
        ax.set_xlabel('Индекс')
        ax.set_ylabel('Значение')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()

    def draw_matrix(self, matrix=None, path=None, x_labels=None, y_labels=None):
        """
        Рисует матрицу в self.plot_canvas_matrix.
        :param path: список (row, col) – клетки, которые нужно закрасить зеленым.
        :param x_labels: список подписей к столбцам (ось X).
        :param y_labels: список подписей к строкам (ось Y).
        """
        self.plot_canvas_matrix.figure.clear()  # удаляем старый рисунок
        ax = self.plot_canvas_matrix.figure.add_subplot(111)

        data = np.array(matrix)
        n_rows, n_cols = data.shape

        # Цвета: белый по умолчанию, зеленый для path
        colors = np.full((n_rows, n_cols, 3), 1.0)  # белый
        for r, c in path:
            if 0 <= r < n_rows and 0 <= c < n_cols:
                colors[r, c] = [0.0, 1.0, 0.0]  # зеленый

        # Рисуем клетки
        x = np.arange(n_cols + 1)
        y = np.arange(n_rows + 1)
        ax.pcolor(x, y, colors, edgecolors='k', linewidths=1)

        # Выводим числа из матрицы
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(j + 0.5, i + 0.5,
                        str(data[i, j]),
                        ha='center', va='center',
                        fontsize=10, color='black')

        # Подписи к осям
        if x_labels is not None and len(x_labels) == n_cols:
            ax.set_xticks(np.arange(0.5, n_cols, 1))
            ax.set_xticklabels(x_labels)
        else:
            ax.set_xticks([])  # скрываем, если нет корректных подписей

        if y_labels is not None and len(y_labels) == n_rows:
            ax.set_yticks(np.arange(0.5, n_rows, 1))
            ax.set_yticklabels(y_labels)
        else:
            ax.set_yticks([])  # скрываем, если нет корректных подписей

        ax.set_aspect('equal')
        ax.invert_yaxis()  # (0,0) в левом верхнем углу

        self.plot_canvas_matrix.draw()

    def read_task_config(self, task_config_path):
        with open(self.task_config_path, 'r', encoding='utf-8') as f:
            self.task_config = json.load(f)

    def build_pg_and_traces(self, prog_num):
        tmp_path = os.path.join(script_dir, tmp_files)
        pg_dest_path = os.path.join(tmp_path, str(prog_num))
        traces_path = os.path.join(pg_dest_path, traces)
        cpp_path = self.code_files[prog_num]
        pg_filename = "pg.dot"
        function_name = self.task_config["function_name"]
        if (build_PG(cpp_path, function_name, pg_dest_path, pg_filename)):
            print("PG build success")
            pg_file_path = os.path.join(pg_dest_path, pg_filename)
            if build_traces(pg_file_path, self.task_config["input_variables"], self.task_config["test_cases"],
                            traces_path):
                print("Traces built")

    def build_all_pg_and_traces_and_update_views(self):
        if self.task_config_path != "" and self.code_files[0] != "" and self.code_files[1] != "":
            self.build_pg_and_traces(0)
            self.build_pg_and_traces(1)
            self.update_views()

    def build_all_pg_and_traces(self):
        if self.task_config_path != "" and self.code_files[0] != "" and self.code_files[1] != "":
            self.build_pg_and_traces(0)
            self.build_pg_and_traces(1)

    def visualise_alignment(self, x: list, y: list, path: list[tuple], widget: FigureCanvas) -> None:
        """
        Визуализирует выравнивание двух последовательностей в виде таблицы с соединениями между ячейками.

        Args:
            x: Первая последовательность (верхний ряд)
            y: Вторая последовательность (нижний ряд)
            path: Список пар индексов (i, j) указывающих соответствие элементов

        Returns:
            None (отображает график)
        """

        # 1. Создать фигуру и оси с помощью matplotlib
        fig, ax = plt.subplots(figsize=(max(len(x), len(y)) * 1.2, 3))

        # Определяем максимальное количество колонок для корректного отображения
        max_cols = max(len(x), len(y))

        # 2. Нарисовать таблицу с двумя строками ячеек:
        #    - Верхняя строка: элементы из x
        #    - Нижняя строка: элементы из y

        # Координаты для центров ячеек:
        # Верхний ряд (x): y_coord = 1
        # Нижний ряд (y): y_coord = 0

        # 4. Добавить цифры в соответствующие ячейки
        # Верхний ряд (последовательность x)
        for i, val in enumerate(x):
            ax.text(i, 1, str(val), ha='center', va='center', fontsize=24,
                    bbox=dict(boxstyle="square,pad=0.3", fc="lightblue", ec="black", lw=0.5))

        # Нижний ряд (последовательность y)
        for j, val in enumerate(y):
            ax.text(j, 0, str(val), ha='center', va='center', fontsize=24,
                    bbox=dict(boxstyle="square,pad=0.3", fc="lightgreen", ec="black", lw=0.5))

        # 3. Для каждой пары (i, j) в path:
        #    - Найти координаты центра i-й ячейки в верхней строке
        #    - Найти координаты центра j-й ячейки в нижней строке
        #    - Нарисовать красную линию между этими центрами
        for i, j in path:
            x_coord_top = i
            y_coord_top = 1

            x_coord_bottom = j
            y_coord_bottom = 0

            ax.plot([x_coord_top, x_coord_bottom], [y_coord_top, y_coord_bottom],
                    color='red', linewidth=5, linestyle='-', marker='o', markersize=10)

        # 5. Настроить внешний вид (убрать оси, сделать равные ячейки и т.д.)
        ax.set_xticks([])  # Убрать метки по оси X
        ax.set_yticks([])  # Убрать метки по оси Y
        ax.set_xlim(-0.5, max_cols - 0.5)  # Ограничить оси для видимости всех ячеек
        ax.set_ylim(-0.7, 1.7)  # Ограничить оси для видимости двух рядов и линий

        ax.set_aspect('equal', adjustable='box')  # Сделать ячейки квадратными (или хотя бы пропорциональными)
        ax.autoscale_view()  # Автоматически настроить масштабирование

        plt.title("Визуализация выравнивания последовательностей")
        plt.grid(False)  # Убрать сетку
        self.tab_alignment_Smith.draw()

    def update_mapping(self):
        self.mapping_table_widget.setRowCount(0)
        mapping = calculate_mapping_for_trace_pair(self.traces_path[0],self.traces_path[1],40,lcs,True,True)
        mapping = mapping["mapping"]
        self.mapping_table_widget.setRowCount(len(mapping))
        for row, (key, value) in enumerate(mapping.items()):
            self.mapping_table_widget.setItem(row, 0, QTableWidgetItem(key))  # Ключ
            self.mapping_table_widget.setItem(row, 1, QTableWidgetItem(value))  # Значение
        mapping_statistics = calculate_mapping_statistics_for_traces(self.traces_path[0],self.traces_path[1],40,lcs,True,True)
        print("MAPPING STATISTICS: ",mapping_statistics)
        visual_pair_statistics(mapping_statistics, self.canvas_pair_stat)


# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":

    #print(get_read_variables_dfs(r"D:\Универ\Кандидатская Диссертация\Вспомогательные программы\TraceViewer\tmp_files\0\traces\1\read_var_sequence.json"))
    """
    x = [1,0,0,0,0,1,0,0,0,1,0,0,0,0]
    y = [1,0,0,0,0,0,1,0,0,1,0,0,0,0]
    path = smith_waterman(x,y,2,-1,-1)
    print(path)
    exit(0)
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

    # cpp_ath = r"D:\Гугл-Диск\КулюкинКС_кандидатская\Сравнение трасс программ\Наши разработки\Эксперименты\Эксперимент_май2025\3\CppSolutions\aaa168_2024.txt\1.cpp"
    # function_name = "cut_rectangle_into_squares"
    # pg_dest_path = os.path.join(script_dir, "tmp_files")
    # build_PG(cpp_ath,function_name,pg_dest_path,"pg.dot")
