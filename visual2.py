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
    return df.loc[rows_to_keep_mask]


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
        show_matrix_btn = QPushButton("Показать матрицу")
        show_matrix_btn.clicked.connect(self._matrix)
        self.var_mapping_layout.addWidget(show_matrix_btn)

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
        eq_matrix = self.df_equality_matrix(self.dedupA, self.dedupB, mapping)
        #print(eq_matrix)
        self.draw_similarity_matrix(eq_matrix)

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

            # Подписи осей (ограничим количество, чтобы не захламлять)
            if rows is not None and len(rows) <= 50:
                ax.set_yticks(np.arange(len(rows)))
                ax.set_yticklabels([str(x) for x in rows])
            else:
                ax.set_yticks([])

            if cols is not None and len(cols) <= 50:
                ax.set_xticks(np.arange(len(cols)))
                ax.set_xticklabels([str(x) for x in cols], rotation=90)
            else:
                ax.set_xticks([])

            ax.set_xlabel('Переменная')
            ax.set_ylabel('Шаг')
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

    def df_equality_matrix(self, df1 : pd.DataFrame, df2 : pd.DataFrame, mapping):
        copy_df1 = df1.copy()
        copy_df2 = df2.copy()
        # mapping задан как соответствие: {имя в df1: имя в df2}
        mapping = self.extract_mapping()

        # 1) Переименовать столбцы в df2 по mapping (нужна инверсия словаря)
        inv_mapping = {v: k for k, v in mapping.items()}
        copy_df2 = copy_df2.rename(columns=inv_mapping)

        # 2) Оставить только участвующие в mapping столбцы
        needed = list(mapping.keys())  # целевые имена колонок (как в df1)

        copy_df1 = copy_df1[[c for c in needed if c in copy_df1.columns]]
        copy_df2 = copy_df2[[c for c in needed if c in copy_df2.columns]]

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
        matrix_df = pd.DataFrame(M, index=df1.index, columns=df2.index)
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