from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QSplitter, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class TraceViewSplitter( QSplitter):

    def __init__(self):
        super().__init__()

        trace_df1 = None
        trace_df2 = None
        variable_mapping = {}

    def add_mapping_widgets(self,vars_prog1, vars_prog2, similarity_matrix):
        pass

    def generate_state_dict(self, vector) -> dict:
        return {}