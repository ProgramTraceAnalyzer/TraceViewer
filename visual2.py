import sys
import os
import json
from itertools import chain
from typing import List, Dict, Tuple, Any, Optional

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout, QWidget,
    QLabel, QPushButton, QListWidget, QHBoxLayout, QCheckBox, QComboBox,
    QInputDialog, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt

import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from pandas.core.dtypes.common import is_numeric_dtype, is_integer_dtype, is_float_dtype

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

import pandas as pd
import numpy as np

# Placeholder for generate_dataframes (replace with your actual implementation or import)
def generate_read_var_dataframe(read_seq_file):
    """
    read_seq_file: путь к JSON-файлу вида [["a","b"],["b","c"],["a"],["a","d"]]
    Возвращает pandas.DataFrame, где строки — шаги, столбцы — переменные,
    значения True/False — читается ли переменная на данном шаге.
    """
    with open(read_seq_file, 'r', encoding='utf-8') as f:
        seq = json.load(f)

    # Все уникальные переменные
    variables = sorted(set(chain.from_iterable(seq)))

    # Множества для быстрого поиска
    step_sets = [set(step) for step in seq]

    # Матрица булевых значений
    data = [[var in step for var in variables] for step in step_sets]

    df = pd.DataFrame(data, columns=variables, dtype=bool)
    return df

def remove_stutter_steps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    rows_to_keep_mask = pd.Series(False, index=df.index)
    rows_to_keep_mask.iloc[0] = True
    df_shifted_for_comparison = df.shift(1).iloc[1:]
    df_current_from_second = df.iloc[1:]
    rows_are_identical_per_column = pd.DataFrame(np.nan, index=df_current_from_second.index, columns=df.columns)
    for col in df.columns:
        current_col_values = df_current_from_second[col]
        shifted_col_values = df_shifted_for_comparison[col]
        both_nan_mask = pd.isna(current_col_values) & pd.isna(shifted_col_values)
        not_nan_equal_mask = (~pd.isna(current_col_values)) & (~pd.isna(shifted_col_values)) & (
                    current_col_values == shifted_col_values)
        rows_are_identical_per_column[col] = both_nan_mask | not_nan_equal_mask
    is_consecutive_duplicate_from_second = rows_are_identical_per_column.all(axis=1)
    rows_to_keep_mask.loc[is_consecutive_duplicate_from_second.index] = ~is_consecutive_duplicate_from_second
    ret_df = df.loc[rows_to_keep_mask]
    ret_df.index = pd.RangeIndex(start=0, stop=len(ret_df), step=1)
    return ret_df


def generate_dataframes(seq_file, variables):
    seq_df = pd.read_json(open(seq_file).read())
    scalar_memory_series = seq_df['memory'].apply(lambda x: x.get('scalar_memory') if isinstance(x, dict) else {})
    scalar_memory_df = pd.DataFrame(scalar_memory_series.tolist())
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
            df_by_variables_not_stutter[v].index = pd.RangeIndex(start=0, stop=len(df_by_variables_not_stutter[v]), step=1)
            print("NON Stutter")
            print(df_by_variables_not_stutter[v])
        else:
            df_by_variables[v] = pd.Series([np.nan] * len(seq_df))
            df_by_variables_not_stutter[v] = pd.DataFrame(columns=[v])

    return seq_df, scalar_memory_df, df_only_variables, df_deduplicated, df_by_variables, df_by_variables_not_stutter

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        fig.patch.set_facecolor('white')


class MainWindow(QMainWindow):
    def __init__(self, fileA=None, fileB=None, fileA_read_vars = None, fileB_read_vars = None):
        super().__init__()

        self.fileA: Optional[str] = fileA
        self.fileB: Optional[str] = fileB

        self.fileA_read_vars = fileA_read_vars
        self.fileB_read_vars = fileB_read_vars

        self.seqA = None
        self.scalarA = None
        self.onlyA = None
        self.dedupA = None

        self.seqB = None
        self.scalarB = None
        self.onlyB = None
        self.dedupB = None
        self.significant_vars: List[str] = []
        self.plot_canvas = MplCanvas(self, width=8, height=6, dpi=100)

        self.setWindowTitle("Data Visualizer")
        self.resize(800, 600)

        self._init_ui()

        if self.fileA and self.fileB:
             try:
                 self.load_pair(self.fileA, self.fileB, auto=True)
                 self.statusBar().showMessage(f"Auto-loaded {self.fileA} and {self.fileB}")
             except Exception as e:
                 self.statusBar().showMessage(f"Auto-load failed: {e}")


    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Select Plot Type
        layout.addWidget(QLabel("Select Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItem("Line Plot")
        self.plot_type_combo.addItem("Scatter Plot")
        self.plot_type_combo.addItem("Bar Chart")
        self.plot_type_combo.addItem("Matrix")
        self.plot_type_combo.addItem("Read Matrix")
        self.plot_type_combo.addItem("Clones Graph")
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot)
        layout.addWidget(self.plot_type_combo)

        # Select Variable A
        layout.addWidget(QLabel("Variable A:"))
        self.var_combo_A = QComboBox()
        self.var_combo_A.currentIndexChanged.connect(self.update_plot)
        layout.addWidget(self.var_combo_A)

        # Select Variable B
        layout.addWidget(QLabel("Variable B:"))
        self.var_combo_B = QComboBox()
        self.var_combo_B.currentIndexChanged.connect(self.update_plot)
        layout.addWidget(self.var_combo_B)

        # Select Stutter Type
        layout.addWidget(QLabel("Тип временного ряда"))
        self.time_series_type_combo = QComboBox()
        self.time_series_type_combo.addItem("Оригинал")
        self.time_series_type_combo.addItem("Без заиканий")
        self.time_series_type_combo.addItem("Без заиканий. Только по одной переменной")
        self.time_series_type_combo.currentIndexChanged.connect(self.update_plot)
        layout.addWidget(self.time_series_type_combo)

        self.var_mapping_layout = QVBoxLayout()
        layout.addLayout(self.var_mapping_layout)

        layout.addWidget(self.plot_canvas)  # Add the plot canvas



        # Refresh Button
        refresh_button = QPushButton("Refresh Plot")
        refresh_button.clicked.connect(self.update_plot)
        layout.addWidget(refresh_button)



    def add_mapping_widgets(self):
        print("Add mapping widgets")
        self.var_mapping_prog1_combos = []
        self.var_mapping_prog2_combos = []
        add_mapping_pair_btn = QPushButton("+")
        add_mapping_pair_btn.clicked.connect(self.add_mapping_combo_pair)
        self.var_mapping_layout.addWidget(add_mapping_pair_btn)


    def add_mapping_combo_pair(self):
        varsA, varsB = self.extract_scalar_keys_from_both_programs()
        var_prog1_combo = QComboBox()
        var_prog2_combo = QComboBox()
        self.add_vars_to_combo_box(var_prog1_combo, varsA)
        self.add_vars_to_combo_box(var_prog2_combo, varsB)
        self.var_mapping_prog1_combos.append(var_prog1_combo)
        self.var_mapping_prog2_combos.append(var_prog2_combo)
        mapping_pair_layout = QHBoxLayout()
        mapping_pair_layout.addWidget(var_prog1_combo)
        mapping_pair_layout.addWidget(var_prog2_combo)
        self.var_mapping_layout.addLayout(mapping_pair_layout)

    def extract_mapping(self):
        mapping = {}
        for i in range(0, len(self.var_mapping_prog1_combos)):
            mapping[self.var_mapping_prog1_combos[i].currentText()] = self.var_mapping_prog2_combos[i].currentText()
        return mapping


    def extract_scalar_keys_from_both_programs(self):
        varsA = self._extract_scalar_keys(self.fileA)
        varsB = self._extract_scalar_keys(self.fileB)
        return varsA, varsB

    def load_pair(self, fileA: Optional[str], fileB: Optional[str], auto: bool = False):
        """
        Load both files (if not None). If one is None, only load the provided one.
        After loading both, compute significant variables (intersection).
        """
        if fileA:
            if not os.path.exists(fileA):
                raise FileNotFoundError(fileA)
            self.fileA = fileA

        if fileB:
            if not os.path.exists(fileB):
                raise FileNotFoundError(fileB)
            self.fileB = fileB

        if self.fileA and self.fileB:
            try:
                varsA, varsB = self.extract_scalar_keys_from_both_programs()

                """
                inter = sorted(list(set(varsA) & set(varsB)))

                if not inter:
                    inter = sorted(list(set(varsA) | set(varsB)))
                self.significant_vars = inter
               

                self._call_generate_for_both()

                self.var_combo_A.clear()
                self.var_combo_B.clear()
                for v in self.significant_vars:
                    self.var_combo_A.addItem(v)
                    self.var_combo_B.addItem(v)
                 """

                """
                if self.significant_vars:
                    self.var_combo_A.setCurrentIndex(0)
                    self.var_combo_B.setCurrentIndex(0)
                """
                self.add_vars_to_combo_box(self.var_combo_A, varsA)
                self.add_vars_to_combo_box(self.var_combo_B, varsB)

                self.update_plot()
                if not auto:
                    self.statusBar().showMessage(f"Loaded {os.path.basename(self.fileA)} and {os.path.basename(self.fileB)}")
            except Exception as e:
                QMessageBox.warning(self, "Load failed", f"Failed to load pair: {e}")
                raise

    def add_vars_to_combo_box(self, combo_box, vars):
        for v in vars:
            combo_box.addItem(v)


    def _extract_scalar_keys(self, path: str) -> List[str]:
        """
        Быстрое извлечение всех ключей scalar_memory в JSON sequence.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        keys = set()
        if not isinstance(data, list):
            return []
        for item in data:
            if not isinstance(item, dict):
                continue
            memory = item.get("memory", {})
            if isinstance(memory, dict):
                scalar = memory.get("scalar_memory", {})
                if isinstance(scalar, dict):
                    keys.update(scalar.keys())
        return sorted(keys)


    def _call_generate_for_both(self):
        print("start generate")
        """
        Вызов generate_dataframes для обоих файлов по списку self.significant_vars.
        """
        if not (self.fileA and self.fileB):
            return
        try:
            self.seqA, self.scalarA, self.onlyA, self.dedupA, self.by_varsA, self.by_notstutterA = generate_dataframes(self.fileA, self.significant_vars)
            print("seqA generated")
        except Exception as e:
            print("Error calling generate dataframes",e)
        try:
            self.seqB, self.scalarB, self.onlyB, self.dedupB, self.by_varsB, self.by_notstutterB = generate_dataframes(self.fileB, self.significant_vars)
        except Exception as e:
           print("Error calling generate dataframes",e)




    def update_plot(self):
        self._call_generate_for_both()
        if not hasattr(self, 'plot_canvas') or not hasattr(self, 'var_combo_A') or not hasattr(self, 'var_combo_B'):
            return

        self.plot_canvas.axes.clear()
        plot_type = self.plot_type_combo.currentText()
        var_A = self.var_combo_A.currentText()
        var_B = self.var_combo_B.currentText()

        if not var_A or not var_B:
            self.plot_canvas.axes.text(0.5, 0.5, "Select variables A and B", ha="center", va="center")
            self.plot_canvas.draw()
            return

        if plot_type == "Line Plot":
            self._plot_line(var_A, var_B)
        elif plot_type == "Scatter Plot":
            self._plot_scatter(var_A, var_B)
        elif plot_type == "Bar Chart":
            self._plot_bar(var_A, var_B)  # Пример
        elif plot_type == "Matrix":
            self.add_mapping_widgets()
            show_matrix_btn = QPushButton("Показать матрицу")
            show_matrix_btn.clicked.connect(self._matrix)
            self.var_mapping_layout.addWidget(show_matrix_btn)
            #self._plot_bar(var_A, var_B)  # Пример
        elif plot_type == "Clones Graph":
            self.add_mapping_widgets()
            show_indicators_btn = QPushButton("Показать графики")
            show_indicators_btn.clicked.connect(self._clone_indicators)
            self.var_mapping_layout.addWidget(show_indicators_btn)
            #self._plot_bar(var_A, var_B)  # Пример
        elif plot_type == "Read Matrix":
            read_dfA = generate_read_var_dataframe(self.fileA_read_vars)
            read_dfB = generate_read_var_dataframe(self.fileB_read_vars)
            self.draw_two_read_matrixes(read_dfA, read_dfB)
        else:
            self.plot_canvas.axes.text(0.5, 0.5, "Unknown plot type", ha="center", va="center")

        self.plot_canvas.draw()

    def current_dfs(self, var_A, var_B):
        if self.time_series_type_combo.currentIndex() == 0:
            return self.scalarA, self.scalarB
        if self.time_series_type_combo.currentIndex() == 1:
            return self.dedupA, self.dedupB
        if self.time_series_type_combo.currentIndex() == 2:
            return self.by_notstutterA[var_A], self.by_notstutterB[var_B]

    def _plot_line(self, var_A, var_B):
        print(self.by_notstutterA.keys())
        print("varA=",var_A,"  varB=",var_B)
        dfA, dfB = self.current_dfs(var_A, var_B)

        if dfA is None or dfB is None:
            self.plot_canvas.axes.text(0.5, 0.5, "DataFrames missing", ha="center", va="center")
            return

        if var_A not in dfA.columns or var_B not in dfB.columns:
            self.plot_canvas.axes.text(0.5, 0.5, "Selected variables not found", ha="center", va="center")
            return

        ax = self.plot_canvas.axes
        ax.clear()

        ax.plot(dfA[var_A], label=f"A: {var_A}")
        ax.plot(dfB[var_B], label=f"B: {var_B}")

        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title(f"Line Plot: {var_A} vs {var_B}")
        ax.legend()

    def _matrix(self):
        #dfA, dfB = self.current_dfs(var_A, var_B)
        print("start matrix")
        mapping = self.extract_mapping()
        copy_df1, copy_df2 = self.generate_nonstutter_mapped_dataframes(self.dedupA, self.dedupB, mapping)
        eq_matrix = self.df_equality_matrix(copy_df1, copy_df2, mapping)
        ng_clones = self.fing_ng_clones(eq_matrix)
        gaps = self.find_gaps(ng_clones)
        clones = self.build_clones(ng_clones, gaps)
        print("NG CLONES: ", ng_clones)
        print("GAPS: ", gaps)
        print("CLONES: ", clones)
        #print(eq_matrix)
        self.draw_clones(eq_matrix, copy_df1) #draw_similarity_matrix(eq_matrix)

    def _clone_indicators(self):
        mapping = self.extract_mapping()
        copy_df1, copy_df2 = self.generate_nonstutter_mapped_dataframes(self.dedupA, self.dedupB, mapping)
        eq_matrix = self.df_equality_matrix(copy_df1, copy_df2, mapping)
        ng_clones = self.fing_ng_clones(eq_matrix)
        gaps = self.find_gaps(ng_clones)
        clones = self.build_clones(ng_clones, gaps)
        print("IS POIN IN NG CLONE: ",self.is_point_in_ng_clone("x", 5, clones[0]))
        self.plot_two_indicators(clones[0], copy_df1, copy_df2)

    def draw_two_read_matrixes(self, matrix1, matrix2):
        """
        Рисует рядом две булевы матрицы чтения на self.plot_canvas:
        - ячейки: False -> белый, True -> зелёный
        - столбцы — переменные, строки — шаги
        Поддерживает вход как pandas.DataFrame (с подписями) и как массивы/списки.
        """
        import numpy as np
        import pandas as pd
        from matplotlib.colors import ListedColormap, BoundaryNorm

        def to_data_and_labels(matrix):
            row_labels = None
            col_labels = None
            if isinstance(matrix, pd.DataFrame):
                data = matrix.astype(bool).to_numpy(dtype=int)
                row_labels = matrix.index
                col_labels = matrix.columns
            else:
                arr = np.asarray(matrix)
                if arr.size == 0:
                    data = arr
                else:
                    data = np.asarray(arr, dtype=bool).astype(int)
            # Гарантируем 2D для imshow
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data, row_labels, col_labels

        data1, rows1, cols1 = to_data_and_labels(matrix1)
        data2, rows2, cols2 = to_data_and_labels(matrix2)

        # Подготовка фигуры и осей
        fig = self.plot_canvas.figure
        fig.clf()
        axes = fig.subplots(nrows=1, ncols=2)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        ax1, ax2 = axes
        self.plot_canvas.axes = axes

        # Общая цветовая схема
        cmap = ListedColormap(['white', 'green'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        def draw_one(ax, data, rows, cols, title):
            ax.clear()
            if data.size == 0:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                return

            # Транспонируем, чтобы по X были шаги, по Y — переменные
            data_T = data.T

            ax.imshow(
                data_T,
                cmap=cmap,
                norm=norm,
                interpolation='nearest',
                aspect='auto',
                origin='upper'
            )

            # Сетка по границам ячеек c учетом новой формы
            ax.set_xticks(np.arange(-0.5, data_T.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, data_T.shape[0], 1), minor=True)
            ax.grid(which='minor', color='lightgray', linewidth=0.5)
            ax.tick_params(which='minor', bottom=False, left=False)

            # Подписи осей: X — шаги (rows), Y — переменные (cols)
            if rows is not None and len(rows) <= 50:
                ax.set_xticks(np.arange(len(rows)))
                ax.set_xticklabels([str(x) for x in rows], rotation=90)
            else:
                ax.set_xticks([])

            if cols is not None and len(cols) <= 50:
                ax.set_yticks(np.arange(len(cols)))
                ax.set_yticklabels([str(x) for x in cols])
            else:
                ax.set_yticks([])

            ax.set_xlabel('Шаг')
            ax.set_ylabel('Переменная')
            ax.set_title(title)

        draw_one(ax1, data1, rows1, cols1, 'Матрица чтения 1')
        draw_one(ax2, data2, rows2, cols2, 'Матрица чтения 2')

        fig.tight_layout()
        self.plot_canvas.draw_idle()

    def draw_read_matrix(self, matrix):
        """
        Визуализирует матрицу чтения (булеву):
        - столбцы — переменные
        - строки — шаги
        - True — зелёный, False — белый
        Рисует на self.plot_canvas (MplCanvas), аналогично draw_similarity_matrix.
        """
        import numpy as np
        import pandas as pd
        from matplotlib.colors import ListedColormap, BoundaryNorm

        # Получаем/создаём ось
        ax = getattr(self.plot_canvas, 'axes', None)
        if ax is None:
            ax = self.plot_canvas.figure.subplots()
            self.plot_canvas.axes = ax
        ax.clear()

        # Подготовка данных и подписей
        row_labels = None
        col_labels = None

        if isinstance(matrix, pd.DataFrame):
            # Приводим к bool, затем к int (False->0, True->1)
            data = matrix.astype(bool).to_numpy(dtype=int)
            row_labels = matrix.index
            col_labels = matrix.columns
        else:
            arr = np.asarray(matrix)
            if arr.size == 0:
                data = arr
            else:
                data = np.asarray(arr, dtype=bool).astype(int)

        # Нет данных
        if data.size == 0:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            self.plot_canvas.draw_idle()
            return

        # Двухцветная карта: 0 -> white, 1 -> green
        cmap = ListedColormap(['white', 'green'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        ax.imshow(
            data,
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            aspect='auto',
            origin='upper'
        )

        # Сетка по границам ячеек
        ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
        ax.grid(which='minor', color='lightgray', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Подписи осей и тики (ограничим количество подписей)
        if row_labels is not None and len(row_labels) <= 50:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels([str(x) for x in row_labels])
        else:
            ax.set_yticks([])

        if col_labels is not None and len(col_labels) <= 50:
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels([str(x) for x in col_labels], rotation=90)
        else:
            ax.set_xticks([])

        ax.set_xlabel('Переменная')
        ax.set_ylabel('Шаг')
        ax.set_title('Матрица чтения')

        self.plot_canvas.draw_idle()

    def plot_read_matrix(self, df):
        import numpy as np
        from matplotlib.colors import ListedColormap

        # Получаем объект Figure из MplCanvas (поддержка figure и fig)
        fig = getattr(self.plot_canvas, 'figure', None)
        if fig is None and hasattr(self.plot_canvas, 'fig'):
            fig = self.plot_canvas.fig
        if fig is None:
            raise AttributeError("plot_canvas не содержит 'figure' или 'fig'.")

        # Полная очистка фигуры и создание новой оси
        fig.clf()
        ax = fig.add_subplot(111)

        if df is None or df.empty:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            fig.tight_layout()
            self.plot_canvas.draw()
            return

        data = df.astype(bool).to_numpy(dtype=int)
        cmap = ListedColormap(['white', 'green'])

        ax.imshow(
            data,
            cmap=cmap,
            vmin=0, vmax=1,
            interpolation='none',
            aspect='auto'
        )

        ax.set_xlabel('Переменная')
        ax.set_ylabel('Шаг')

        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels([str(c) for c in df.columns], rotation=90, ha='right')
        ax.set_yticks(range(df.shape[0]))
        ax.set_yticklabels([str(i) for i in df.index])

        ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)

        fig.tight_layout()
        self.plot_canvas.draw()

    def draw_clones(self, matrix, df : pd.DataFrame):
        ng_clones = self.fing_ng_clones(matrix)
        gaps = self.find_gaps(ng_clones)

        # Извлекаем или создаем оси Matplotlib
        ax = getattr(self.plot_canvas, 'axes', None)
        if ax is None:
            ax = self.plot_canvas.figure.subplots()
            self.plot_canvas.axes = ax

        ax.clear()

        # Обработка данных матрицы (DataFrame или ndarray)
        if isinstance(matrix, pd.DataFrame):
            data = matrix.to_numpy().astype(int)
            row_labels = list(matrix.index)  # Преобразуем в список, чтобы избежать проблем
            col_labels = list(matrix.columns)  # Преобразуем в список
        else:
            data = np.asarray(matrix, dtype=bool).astype(int)
            row_labels = [str(i) for i in range(data.shape[0])] if data.shape[
                                                                       0] <= 50 else []  # Нумерация по умолчанию, если нет лейблов
            col_labels = [str(i) for i in range(data.shape[1])] if data.shape[1] <= 50 else []

        N_X = data.shape[0]  # Количество строк
        N_Y = data.shape[1]  # Количество столбцов

        # Установка пределов осей
        ax.set_xlim(0.5, N_Y + 0.5)
        ax.set_ylim(N_X + 0.5, 0.5)  # Инвертированная ось Y

        # Настройка меток осей
        ax.set_xticks(np.arange(1, N_Y + 1))
        ax.set_yticks(np.arange(1, N_X + 1))

        # Установка меток строк/столбцов (с проверкой на переполнение)
        if len(col_labels) <= 50:
            ax.set_xticklabels(col_labels, rotation=90, fontsize=10)
        else:
            ax.set_xticklabels([])  # Не отображать, если слишком много
        ax.set_xlabel('Трасса программы №2')

        if len(row_labels) <= 50:
            ax.set_yticklabels(row_labels, fontsize=10)
        else:
            ax.set_yticklabels([])  # Не отображать, если слишком много
        ax.set_ylabel('Трасса программы №2')

        ax.set_title('Матрица рассеяния')

        # Настройка сетки
        ax.grid(which='both', color='k', linestyle='-', linewidth=0.5)
        ax.tick_params(length=0, which='both')

        # Функция преобразования координат
        def get_center(coord):
            return (coord['x'] + 1, coord['y'] + 1)  # Преобразуем 0-индексацию в 1-индексацию и меняем местами x и y

        # Рисование ng_clones
        for clone in ng_clones:
            start_coord = clone['start']
            finish_coord = clone['finish']

            x_start, y_start = get_center(start_coord)  # Преобразуем координаты начала
            x_end, y_end = get_center(finish_coord)  # и конца

            # Рисуем линию
            ax.plot([x_start, x_end], [y_start, y_end], color='black', linewidth=2)

            # Перебираем все целочисленные точки вдоль линии
            num_points = max(abs(x_end - x_start), abs(y_end - y_start)) + 1  # +1 чтобы захватить конечную точку
            x_coords = np.linspace(x_start, x_end, num_points)
            y_coords = np.linspace(y_start, y_end, num_points)

            for x, y in zip(x_coords, y_coords):
                x_int = int(round(x))
                y_int = int(round(y))

                # Проверяем, что точка (x_int, y_int) находится в пределах матрицы.  Преобразуем в 0-индексацию для проверки
                if 0 <= y_int - 1 < data.shape[0] and 0 <= x_int - 1 < data.shape[1]:
                    ax.plot(x_int, y_int, marker='D', color='black', markersize=8)  # 'D' для ромбика
                    row = df.iloc[y_int-1]
                    label = '\n'.join(f'{col}={row[col]}' for col in df.columns)
                    ax.annotate(label, (x_int, y_int), xytext=(20, 10), textcoords='offset points', ha='left', va='center')



        # Рисование G-связей (gaps)
        for gap in gaps:
            start_coord = gap['start']
            finish_coord = gap['finish']

            x_start, y_start = get_center(start_coord)
            x_end, y_end = get_center(finish_coord)

            ax.plot([x_start, x_end], [y_start, y_end], color='gray', linewidth=2.5, alpha=0.7)
            ax.plot(x_start, y_start, 'o', color='gray', markersize=6)
            ax.plot(x_end, y_end, 'o', color='gray', markersize=6)

        self.plot_canvas.draw_idle()

    def plot_indicator_for_df(self, df, coord_name, clones, ax=None, title=None,
                              marker='o', marker_size=24,
                              color_y0='tab:red', color_y1='tab:green',
                              integer_atol=1e-9,
                              text_offset=0.06,
                              text_fontsize=9):
        if not is_numeric_dtype(df.index):
            raise ValueError("Индекс DataFrame должен быть числовым (int/float).")

        x_vals = df.index.to_numpy(dtype=float)
        y_vals = np.array(
            [1 if self.is_point_in_ng_clone(coord_name, float(x), clones) else 0 for x in x_vals],
            dtype=int
        )

        fig = self.plot_canvas.figure
        if ax is None:
            fig.clf()
            ax = fig.add_subplot(111)

        ax.step(x_vals, y_vals, where='post', color='tab:blue', linewidth=1.5)

        if is_integer_dtype(df.index):
            integer_mask = np.ones_like(x_vals, dtype=bool)
        else:
            integer_mask = np.isfinite(x_vals) & np.isclose(x_vals, np.round(x_vals), atol=integer_atol)

        xi = x_vals[integer_mask]
        yi = y_vals[integer_mask]
        if xi.size:
            pos_idx = np.nonzero(integer_mask)[0]
            rows_int = df.iloc[pos_idx]

            mask0 = (yi == 0)
            mask1 = (yi == 1)
            if mask0.any():
                ax.scatter(xi[mask0], yi[mask0], s=marker_size, color=color_y0, marker=marker, zorder=3)
            if mask1.any():
                ax.scatter(xi[mask1], yi[mask1], s=marker_size, color=color_y1, marker=marker, zorder=3)

            # Форматирование значений: числа -> целые без .0
            def format_value(val, col):
                if is_integer_dtype(df[col]):
                    return str(int(val)) if pd.notna(val) else "NaN"
                if is_float_dtype(df[col]):
                    return str(int(round(float(val)))) if pd.notna(val) else "NaN"
                return str(val)

            for k, (x, y) in enumerate(zip(xi, yi)):
                row = rows_int.iloc[k]
                label = "\n".join(f"{col}={format_value(row[col], col)}" for col in rows_int.columns)

                if y == 1:
                    y_text = y + text_offset
                    va = 'bottom'
                    txt_color = color_y1
                else:
                    y_text = y - text_offset
                    va = 'top'
                    txt_color = color_y0

                ax.text(x, y_text, label,
                        fontsize=text_fontsize,
                        color=txt_color,
                        ha='center', va=va,
                        zorder=4)

        ax.set_ylim(-2, 3)
        ax.set_yticks([0, 1])
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel("Index")
        ax.set_ylabel("In ng_clone")
        if title:
            ax.set_title(title)

        return ax

    def plot_two_indicators(self, clones, df1, df2):
        fig = self.plot_canvas.figure
        fig.clf()

        # Вторая ось делим X с первой
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        self.plot_indicator_for_df(df=df1, coord_name='y', clones=clones, ax=ax1, title="Трасса программы №1")
        self.plot_indicator_for_df(df=df2, coord_name='x', clones=clones, ax=ax2, title="Трасса программы №2")

        # У верхнего графика убираем подписи и тики по X
        ax1.set_xlabel("")
        ax1.tick_params(labelbottom=False)

        # Уменьшаем отступы у заголовков и подписей осей
        ax1.set_title("Трасса программы №1", pad=2)
        ax2.set_title("Трасса программы №2", pad=2)
        ax1.yaxis.set_label_coords(-0.06, 0.5)  # или ax1.set_ylabel(..., labelpad=2)
        ax2.yaxis.set_label_coords(-0.06, 0.5)

        # Контроль вертикального зазора
        fig.subplots_adjust(hspace=0.05)  # уменьшите значение по вкусу

        # Не используйте tight_layout вместе с subplots_adjust — он может пересчитать зазоры обратно.
        # Если всё же нужен tight_layout, то так:
        # fig.tight_layout(pad=0.4, h_pad=0.2)

        if hasattr(self.plot_canvas, "draw_idle"):
            self.plot_canvas.draw_idle()
        else:
            self.plot_canvas.draw()

    def draw_similarity_matrix(self, matrix):
        # matrix: np.ndarray[bool] или pd.DataFrame c dtype=bool
        ax = getattr(self.plot_canvas, 'axes', None)
        if ax is None:  # на случай, если в MplCanvas нет axes
            ax = self.plot_canvas.figure.subplots()
            self.plot_canvas.axes = ax

        ax.clear()

        # Подготовка данных и подписей
        if isinstance(matrix, pd.DataFrame):
            data = matrix.to_numpy().astype(int)  # False->0, True->1
            row_labels = matrix.index
            col_labels = matrix.columns
        else:
            data = np.asarray(matrix, dtype=bool).astype(int)
            row_labels = None
            col_labels = None

        # Двухцветная карта: 0 -> white, 1 -> green
        cmap = ListedColormap(['white', 'green'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        im = ax.imshow(
            data,
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            aspect='auto',
            origin='upper'
        )

        # Тонкая сетка по ячейкам (опционально)
        ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
        ax.grid(which='minor', color='lightgray', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Подписи осей (ограничим, чтобы не захламлять при больших размерах)
        if row_labels is not None and len(row_labels) <= 50:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels)
        else:
            ax.set_yticks([])

        if col_labels is not None and len(col_labels) <= 50:
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=90)
        else:
            ax.set_xticks([])

        ax.set_xlabel('df2 rows')
        ax.set_ylabel('df1 rows')
        ax.set_title('Similarity matrix')

        self.plot_canvas.draw_idle()

    def align_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Выравнивает два DataFrame путем вставки копий смежных строк так, чтобы:
          - длины стали равными,
          - количество совпавших пар строк (сравнение по всем столбцам, NaN==NaN) было максимальным,
          - при прочих равных количество вставок было минимальным.

        Ограничения/предположения:
          - Столбцы должны быть одинаковыми и в одинаковом порядке.
          - Допускаются только вставки (дублирование смежных строк). Удаления не выполняются.
          - Если один из датафреймов пуст, а другой нет — задача невыполнима (нечего дублировать).

        Возвращает:
          aligned_df1, aligned_df2 — новые DataFrame одинаковой длины.
        """
        # Проверки столбцов
        if list(df1.columns) != list(df2.columns):
            raise ValueError("Оба DataFrame должны иметь одинаковые столбцы в одинаковом порядке.")

        n, m = len(df1), len(df2)

        # Граничные случаи
        if n == 0 and m == 0:
            return df1.copy(), df2.copy()
        if (n == 0) != (m == 0):
            raise ValueError("Невозможно выровнять: один DataFrame пуст, другой — нет (копировать нечего).")

        # Преобразуем к numpy для быстрых операций и сравнения с учетом NaN==NaN
        arr1 = df1.to_numpy()
        arr2 = df2.to_numpy()

        def rows_equal(i: int, j: int) -> bool:
            a = arr1[i]
            b = arr2[j]
            # Поэлементное сравнение с учетом NaN==NaN
            eq = (a == b)
            nan_eq = pd.isna(a) & pd.isna(b)
            return bool(np.all(eq | nan_eq))

        # DP по двум критериям: (mismatch_count, gap_count), минимизация лексикографически
        dp_mis = np.zeros((n + 1, m + 1), dtype=int)
        dp_gap = np.zeros((n + 1, m + 1), dtype=int)
        # backpointer: 0=diag, 1=up (gap в df2), 2=left (gap в df1)
        bp = np.zeros((n + 1, m + 1), dtype=np.uint8)

        # Инициализация границ: только вставки
        for i in range(1, n + 1):
            dp_mis[i, 0] = 0
            dp_gap[i, 0] = i
            bp[i, 0] = 1  # up
        for j in range(1, m + 1):
            dp_mis[0, j] = 0
            dp_gap[0, j] = j
            bp[0, j] = 2  # left

        # Заполнение таблицы
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # diag
                mis_d = dp_mis[i - 1, j - 1] + (0 if rows_equal(i - 1, j - 1) else 1)
                gap_d = dp_gap[i - 1, j - 1]

                # up (gap в df2)
                mis_u = dp_mis[i - 1, j]
                gap_u = dp_gap[i - 1, j] + 1

                # left (gap в df1)
                mis_l = dp_mis[i, j - 1]
                gap_l = dp_gap[i, j - 1] + 1

                # Выбираем минимум лексикографически: сначала по mismatches, затем по gaps
                best_mis, best_gap, best_dir = mis_d, gap_d, 0  # diag
                if (mis_u < best_mis) or (mis_u == best_mis and gap_u < best_gap):
                    best_mis, best_gap, best_dir = mis_u, gap_u, 1  # up
                if (mis_l < best_mis) or (mis_l == best_mis and gap_l < best_gap):
                    best_mis, best_gap, best_dir = mis_l, gap_l, 2  # left

                dp_mis[i, j] = best_mis
                dp_gap[i, j] = best_gap
                bp[i, j] = best_dir

        # Восстановление пути
        ops = []
        i, j = n, m
        while i > 0 or j > 0:
            d = bp[i, j]
            ops.append(d)
            if d == 0:  # diag
                i -= 1
                j -= 1
            elif d == 1:  # up (gap в df2)
                i -= 1
            else:  # left (gap в df1)
                j -= 1
            ops.reverse()

            # Построение выровненных списков строк
            aligned1 = []
            aligned2 = []
            i = j = 0
            for d in ops:
                if d == 0:
                    # diag: сопоставляем реальные строки
                    aligned1.append(arr1[i].tolist())
                    aligned2.append(arr2[j].tolist())
                    i += 1
                    j += 1
                elif d == 2:
                    # left: вставка в df1 (дублируем смежную строку df1)
                    if aligned1:
                        row1 = aligned1[-1]  # копия предыдущей выровненной строки
                    else:
                        row1 = arr1[i].tolist()  # в начале — копия ближайшей следующей
                    aligned1.append(row1)
                    aligned2.append(arr2[j].tolist())
                    j += 1
                else:  # d == 1
                    # up: вставка в df2 (дублируем смежную строку df2)
                    aligned1.append(arr1[i].tolist())
                    if aligned2:
                        row2 = aligned2[-1]
                    else:
                        row2 = arr2[j].tolist()
                    aligned2.append(row2)
                    i += 1

            aligned_df1 = pd.DataFrame(aligned1, columns=df1.columns)
            aligned_df2 = pd.DataFrame(aligned2, columns=df2.columns)
            return aligned_df1, aligned_df2

    def generate_nonstutter_mapped_dataframes(self, df1 : pd.DataFrame, df2 : pd.DataFrame, mapping):
        copy_df1 = df1.copy()
        copy_df2 = df2.copy()

        # 1) Переименовать столбцы в df2 по mapping (нужна инверсия словаря)
        inv_mapping = {v: k for k, v in mapping.items()}
        copy_df2 = copy_df2.rename(columns=inv_mapping)

        # 2) Оставить только участвующие в mapping столбцы
        needed = list(mapping.keys())  # целевые имена колонок (как в df1)

        copy_df1 = copy_df1[[c for c in needed if c in copy_df1.columns]]
        copy_df2 = copy_df2[[c for c in needed if c in copy_df2.columns]]


        copy_df1 = remove_stutter_steps(copy_df1)
        print("COPY DF1\n", copy_df1)
        copy_df2 = remove_stutter_steps(copy_df2)
        print("COPY DF2\n", copy_df2)
        return copy_df1, copy_df2


    def fing_ng_clones(self, matrix_df : pd.DataFrame, min_ng_len = 1):
        matrix_df = matrix_df.copy()
        ng_clones = []
        print("MATRIX COLUMNS: ", matrix_df.at[0,2])
        #return ng_clones
        len_ref = len(matrix_df)
        len_stud = len(matrix_df.keys().tolist())
        for i in range(0, len_ref):
            for j in range(0, len_stud):
                if matrix_df.at[i,j] == True:
                    ng_len = 0
                    k = 0
                    while (i + k) < len_ref and (j + k) < len_stud and matrix_df.at[i + k, j + k]:
                        matrix_df.at[i + k, j + k] = False
                        ng_len += 1
                        k += 1
                    if (ng_len >= min_ng_len):
                        ng_clones.append({"start": {"y": i, "x": j}, "finish": {"y": i + k - 1, "x": j + k - 1}})
        return ng_clones

    def find_gaps(self, ng_clones):
        gaps = []
        for i in range(0, len(ng_clones)):
            for j in range(0, len(ng_clones)):
                if i != j:
                    clone_i = ng_clones[i]
                    clone_j = ng_clones[j]
                    if clone_j["start"]["y"] >= clone_i["finish"]["y"] and clone_j["start"]["x"] >= clone_i["finish"]["x"]:
                        gaps.append({"start": {"y": clone_i["finish"]["y"], "x": clone_i["finish"]["x"]}, "finish": {"y": clone_j["start"]["y"], "x": clone_j["start"]["x"]}})
        return gaps

    def build_clones(self, ng_clones, gaps):
        # Ключ для точек (x, y), чтобы удобно сравнивать и индексировать
        def pkey(pt):
            return (pt['x'], pt['y'])

        # Индексация по стартовой точке для быстрого поиска следующих элементов
        from collections import defaultdict
        ng_by_start = defaultdict(list)
        gaps_by_start = defaultdict(list)

        for i, ng in enumerate(ng_clones):
            ng_by_start[pkey(ng['start'])].append(i)
        for j, gp in enumerate(gaps):
            gaps_by_start[pkey(gp['start'])].append(j)

        clones = []

        # Множества для отслеживания использованных элементов в текущем пути
        used_ng = set()
        used_gap = set()

        # Преобразуем путь в формат удобный для возврата
        def path_to_output(path):
            out = []
            for kind, idx in path:
                obj = ng_clones[idx] if kind == 'ng' else gaps[idx]
                out.append({
                    'type': 'ng_clone' if kind == 'ng' else 'gap',
                    'start': obj['start'],
                    'finish': obj['finish'],
                })
            return out

        # Рекурсивно расширяем путь, когда последний элемент — ng_clone
        def dfs_from_ng(ng_idx, path):
            # Добавляем текущий ng в путь
            path.append(('ng', ng_idx))
            used_ng.add(ng_idx)

            cur_finish = pkey(ng_clones[ng_idx]['finish'])

            # Пытаемся сделать шаг: gap -> ng
            extended = False  # Был ли выполнен хотя бы один валидный шаг (gap+ng)
            for gap_idx in gaps_by_start.get(cur_finish, []):
                if gap_idx in used_gap:
                    continue

                # Пробуем поставить gap
                used_gap.add(gap_idx)
                path.append(('gap', gap_idx))

                next_start = pkey(gaps[gap_idx]['finish'])
                any_ng = False  # Удалось ли после этого gap поставить хотя бы один ng

                for next_ng_idx in ng_by_start.get(next_start, []):
                    if next_ng_idx in used_ng:
                        continue
                    any_ng = True
                    dfs_from_ng(next_ng_idx, path)

                # Откат gap
                path.pop()
                used_gap.remove(gap_idx)

                if any_ng:
                    extended = True

            # Если не удалось сделать ни одного шага gap+ng,
            # то текущий путь максимален и заканчивается на ng_clone — фиксируем его.
            if not extended:
                clones.append(path_to_output(path))

            # Откат текущего ng
            used_ng.remove(ng_idx)
            path.pop()

        # Запускаем построение из каждого ng_clone как из старта
        for i in range(len(ng_clones)):
            dfs_from_ng(i, [])

        return clones

    def is_point_in_ng_clone(self,coord_name, coord_num_value, clone):
        if coord_name not in ("x", "y"):
            raise ValueError('coord_name должен быть "x" или "y"')
        if not isinstance(coord_num_value, (int, float)):
            raise TypeError("coord_num_value должен быть числом")
        if not clone:
            return False

        v = coord_num_value

        for segment in clone:
            seg_type = segment.get("type")
            start = segment.get("start", {})
            finish = segment.get("finish", {})

            if seg_type is None or coord_name not in start or coord_name not in finish:
                # Пропускаем некорректные сегменты
                continue

            a = start[coord_name]
            b = finish[coord_name]

            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue

            lo, hi = (a, b) if a <= b else (b, a)

            # Проверяем попадание в отрезок и тип сегмента
            if lo <= v <= hi and seg_type == "ng_clone":
                return True

        return False

    def df_equality_matrix(self, df1 : pd.DataFrame, df2 : pd.DataFrame, mapping):
        mapping = self.extract_mapping()
        copy_df1, copy_df2 = self.generate_nonstutter_mapped_dataframes(df1,df2,mapping)

        aligned_df1, aligned_df2 = self.align_dataframes(copy_df1, copy_df2)

        print("ALIGNED DF1:\n",aligned_df1)
        print("ALIGNED DF2:\n", aligned_df2)

        A = copy_df1.to_numpy()
        B = copy_df2.to_numpy()

        # маски пропусков
        A_na = pd.isna(A)
        B_na = pd.isna(B)

        # поэлементное сравнение (с NaN считаем равными, если NaN в обеих строках одного столбца)
        eq = (A[:, None, :] == B[None, :, :]) | (A_na[:, None, :] & B_na[None, :, :])

        # итоговая матрица: строки равны, если все столбцы равны
        M = eq.all(axis=2)  # shape: (len(df1), len(df2)), dtype: bool

        # при желании — оформить как DataFrame с индексами исходных строк
        matrix_df = pd.DataFrame(M, index=copy_df1.index, columns=copy_df2.index)
        print("MATRIX DF: ",matrix_df)
        return matrix_df

    def _plot_scatter(self, var_A, var_B):
        dfA = self.by_notstutterA[var_A] #self.scalarA
        dfB = self.by_notstutterB[var_B] #self.scalarB

        if dfA is None or dfB is None:
            self.plot_canvas.axes.text(0.5, 0.5, "DataFrames missing", ha="center", va="center")
            return

        if var_A not in dfA.columns or var_B not in dfB.columns:
            self.plot_canvas.axes.text(0.5, 0.5, "Selected variables not found", ha="center", va="center")
            return

        ax = self.plot_canvas.axes
        ax.clear()

        ax.scatter(dfA[var_A], dfB[var_B], label=f"{var_A} vs {var_B}")

        ax.set_xlabel(var_A)
        ax.set_ylabel(var_B)
        ax.set_title(f"Scatter Plot: {var_A} vs {var_B}")
        ax.legend()

    def _plot_bar(self, var_A, var_B):
        #dfA = self.scalarA
        #dfB = self.scalarB
        dfA, dfB = self.current_dfs(var_A, var_B)
        if dfA is None or dfB is None:
            self.plot_canvas.axes.text(0.5, 0.5, "DataFrames missing", ha="center", va="center")
            return

        if var_A not in dfA.columns or var_B not in dfB.columns:
            self.plot_canvas.axes.text(0.5, 0.5, "Selected variables not found", ha="center", va="center")
            return

        ax = self.plot_canvas.axes
        ax.clear()

        # Assuming you want to plot the mean values for each variable
        mean_A = dfA[var_A].mean()
        mean_B = dfB[var_B].mean()

        # Bar chart data
        labels = [f"A: {var_A}", f"B: {var_B}"]
        values = [mean_A, mean_B]

        ax.bar(labels, values, color=['blue', 'green'])  # Customize colors if desired

        ax.set_ylabel("Mean Value")
        ax.set_title(f"Bar Chart: Mean Values of {var_A} and {var_B}")

def main():
    app = QApplication(sys.argv)

    # Get file names from command line arguments
    fileA = None
    fileB = None
    fileA_read_seq = None
    fileB_read_seq = None
    if len(sys.argv) >= 3:  # Expect at least two file arguments
        fileA = sys.argv[1]
        fileB = sys.argv[2]

    if len(sys.argv) >= 5:
        fileA_read_seq = sys.argv[3]
        fileB_read_seq = sys.argv[4]

    # Create and show the main window
    w = MainWindow(fileA, fileB, fileA_read_seq, fileB_read_seq)
    w.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()