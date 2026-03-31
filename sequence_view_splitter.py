from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QSplitter, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class SequenceViewSplitter( QSplitter):


    def __init__(self, seqX=None, seqY=None, path=None, matrix=None, seqX_name="Sequence X", seqY_name="Sequence Y"):
        super().__init__()
        self.setOrientation(Qt.Vertical)
        self.seqX = seqX
        self.seqY = seqY
        self.path = path
        self.matrix = matrix
        self.seqX_name = seqX_name
        self.seqY_name = seqY_name

        # Создаем виджеты для графиков
        self.create_widgets()
        self.addWidgets()

    def set_title(self, title):
        self.label.setText(title)

    def create_widgets(self):
        """Создает виджеты matplotlib"""
        # Виджет для графика последовательностей
        self.fig_sequences = Figure(figsize=(8, 4))
        self.canvas_sequences = FigureCanvas(self.fig_sequences)

        # Виджет для матрицы скоринга
        self.fig_matrix = Figure(figsize=(6, 6))
        self.canvas_matrix = FigureCanvas(self.fig_matrix)

        # Виджет для визуализации выравнивания
        self.fig_alignment = Figure(figsize=(8, 4))
        self.canvas_alignment = FigureCanvas(self.fig_alignment)

    def addWidgets(self):
        """Добавляет виджеты в layout"""

        self.label = QLabel()
        self.addWidget(self.label)

        # Создаем горизонтальный layout для первых двух графиков
        top_splitter = QSplitter(Qt.Horizontal)

        # Виджет-контейнер для верхней части


        # Добавляем графики последовательностей и матрицы в верхний layout
        top_splitter.addWidget(self.canvas_sequences)
        top_splitter.addWidget(self.canvas_matrix)

        # Добавляем все в основной layout
        self.addWidget(top_splitter)
        self.addWidget(self.canvas_alignment)

        # Обновляем отображение если есть данные
        if self.seqX is not None and self.seqY is not None:
            self.update(self.seqX, self.seqY, self.path, self.matrix, self.seqX_name, self.seqY_name)

    def plot_sequences(self):
        """Рисует график двух последовательностей с соединениями точек из path"""


        self.fig_sequences.clear()
        ax = self.fig_sequences.add_subplot(111)
        self.canvas_sequences.draw()

        if self.seqX is None or self.seqY is None or self.path is None:
            return

        # Первая последовательность
        x1 = range(len(self.seqX))
        line1 = ax.plot(x1, self.seqX, 'b-', linewidth=2, marker='^',
                        markersize=8, label=self.seqX_name, alpha=0.5)

        # Вторая последовательность
        x2 = range(len(self.seqY))
        line2 = ax.plot(x2, self.seqY, 'r-', linewidth=2, marker='v',
                        markersize=8, label=self.seqY_name, alpha=0.5)

        # Рисуем пунктирные линии соответствий между точками
        for i, j in self.path:
            if i < len(self.seqX) and j < len(self.seqY):
                # Координаты точек
                x_start = i
                y_start = self.seqX[i]
                x_end = j
                y_end = self.seqY[j]

                # Рисуем пунктирную линию между соответствующими точками
                ax.plot([x_start, x_end], [y_start, y_end],
                        'g--', linewidth=1, alpha=0.7)

        ax.set_xlabel('Step [i]')
        ax.set_ylabel('Value')
        ax.set_title('Sequences Visualization with Alignment Path')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.canvas_sequences.draw()

    def plot_matrix(self):
        """Рисует матрицу скоринга с выделенным path - версия с imshow"""


        self.fig_matrix.clear()
        ax = self.fig_matrix.add_subplot(111)
        self.canvas_matrix.draw()

        if self.matrix is None or self.path is None:
            return

        # Создаем маску для цветов
        color_matrix = np.zeros(self.matrix.shape + (3,))  # RGB матрица
        # Все клетки серые (0.8, 0.8, 0.8)
        color_matrix[:, :, :] = 0.8

        # Клетки из path - зеленые (0, 1, 0)
        for i, j in self.path:
            if i < self.matrix.shape[0] and j < self.matrix.shape[1]:
                color_matrix[i, j, :] = [0, 0.8, 0]  # Зеленый

        # Отображаем цветную матрицу
        ax.imshow(color_matrix)

        # Добавляем числа поверх цветных клеток
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                ax.text(j, i, f'{self.matrix[i, j]:.0f}',
                        ha='center', va='center', fontsize=8,
                        color='black' if color_matrix[i, j, 1] > 0.5 else 'black')

        ax.set_xlabel(self.seqY_name+' Index')
        ax.set_ylabel(self.seqX_name+' Index')
        ax.set_title('Scoring Matrix')

        self.canvas_matrix.draw()

    def visualise_alignment(self, x: list, y: list, path: list):



        """Визуализирует выравнивание двух последовательностей"""
        self.fig_alignment.clear()
        ax = self.fig_alignment.add_subplot(111)
        self.canvas_alignment.draw()
        if self.path == None:
            return

        # Определяем максимальное количество колонок для корректного отображения
        max_cols = max(len(x), len(y))

        # Верхний ряд (последовательность x)
        for i, val in enumerate(x):
            ax.text(i, 1, str(int(val)), ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle="square,pad=0.3", fc="lightblue", ec="black", lw=0.5))

        # Нижний ряд (последовательность y)
        for j, val in enumerate(y):
            ax.text(j, 0, str(int(val)), ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle="square,pad=0.3", fc="lightgreen", ec="black", lw=0.5))

        # Рисуем соединения между ячейками
        for i, j in path:
            x_coord_top = i
            y_coord_top = 1

            x_coord_bottom = j
            y_coord_bottom = 0

            ax.plot([x_coord_top, x_coord_bottom], [y_coord_top, y_coord_bottom],
                    color='red', linewidth=2, linestyle='-', marker='o', markersize=8)

        # Настраиваем внешний вид
        ax.set_xticks([])  # Убрать метки по оси X
        ax.set_yticks([])  # Убрать метки по оси Y
        ax.set_xlim(-0.5, max_cols - 0.5)
        ax.set_ylim(-0.7, 1.7)

        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Sequence Alignment Visualization")
        ax.grid(False)

        self.canvas_alignment.draw()

    def update(self, seqX, seqY, path, matrix, seqX_name="Sequence X", seqY_name="Sequence Y"):
        """Обновляет все графики новыми данными"""
        self.seqX = seqX
        self.seqY = seqY
        self.path = path
        self.matrix = matrix
        self.seqX_name = seqX_name
        self.seqY_name = seqY_name

        print("WATERMAN !!!!!!!")

        # Обновляем все графики
        self.plot_sequences()
        self.plot_matrix()
        self.visualise_alignment(seqX, seqY, path)


# Пример использования
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow


    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Sequence Alignment Viewer")
            self.setGeometry(100, 100, 1200, 800)

            # Пример данных
            seqX = [1, 3, 4, 9, 12, 15]
            seqY = [2, 5, 8, 10, 14]
            path = [(0, 0), (1, 2), (2, 2), (3, 3), (4, 4)]
            matrix = np.random.rand(len(seqX), len(seqY)) * 10

            # Создаем layout
            central_widget = QWidget()
            layout = SequenceViewSplitter(seqX, seqY, path, matrix, "var_a", "remind")


            self.setCentralWidget(layout)


    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
