from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QSplitter, QLabel, QComboBox, QTableWidget, \
    QTableWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import json

class VariableActionHistorySplitter(QSplitter):

    def __init__(self):
        super().__init__()
        self.setOrientation(Qt.Vertical)
        variable_select_layout = QSplitter()
        variable_select_layout.setOrientation(Qt.Horizontal)
        variable_select_layout.addWidget(QLabel("Переменная:"))
        self.variable_select_combo = QComboBox()
        variable_select_layout.addWidget(self.variable_select_combo)

        # self.addWidget(variable_select_layout)
        self.history_table_widget = QTableWidget()
        self.history_table_widget.setColumnCount(2)
        self.history_table_widget.setHorizontalHeaderLabels(["Действие", "Значение после действия"])
        self.addWidget(self.history_table_widget)
        self.variable_select_combo.currentIndexChanged.connect(self.fill_actions)

        self.state_seq = []
        self.action_seq = []
        self.variables = set()
        self.current_variable = None

    def extract_variables(self):
        self.variables = set()
        for item in self.state_seq:
            scalar_memory = item["memory"]["scalar_memory"]
            for variable in scalar_memory.keys():
                if variable not in self.variables:
                    self.variables.add(variable)
            array_memory = item["memory"]["array_memory"]
            for variable in array_memory.keys():
                if variable not in self.variables:
                    self.variables.add(variable)

    def fill_variable_combo(self):
        self.variable_select_combo.clear()
        self.extract_variables()
        for v in self.variables:
            self.variable_select_combo.addItem(v)

    def fill_actions(self):
        current_variable = self.current_variable#variable_select_combo.currentText()
        print("BEFORE SET 0")
        self.history_table_widget.setRowCount(0)
        print("AFTER SET 0")
        #self.table_widget.clear()  # Очищает ячейки и заголовки
        #self.table_widget.setHorizontalHeaderLabels(["Действие", "Значение после действия"])  # Восстанавливаем заголовки
        #self.table_widget.setRowCount(0)

        for i in range(0,len(self.action_seq)):
            act = self.action_seq[i]
            state = self.state_seq[i+1]

            current_value = None
            if current_variable in state["memory"]["scalar_memory"].keys():
                current_value = state["memory"]["scalar_memory"][current_variable]
            if current_variable in state["memory"]["array_memory"].keys():
                current_value = state["memory"]["array_memory"][current_variable]

            if act["type"] == "assign" and act["assigned_variable"] == current_variable:
                print("ACT ",act)
                row_count = self.history_table_widget.rowCount()
                self.history_table_widget.setRowCount(row_count+1)
                print("SET ROW COUNT: ",row_count)
                if current_value is not None:
                    self.history_table_widget.setItem(row_count,1,QTableWidgetItem(str(current_value)))
                self.history_table_widget.setItem(row_count,0,QTableWidgetItem(act["assigned_variable"]+"="+act["expression"]))

    def update_state(self,state_seq,action_seq):
        self.state_seq = state_seq
        self.action_seq = action_seq
        print("ACTION SEQ: ",action_seq)
        print("STATE SEQ: ",state_seq)
        #self.fill_variable_combo()
        #self.fill_actions()

    def set_current_variable(self,var):
        self.current_variable = var
        self.fill_actions()


    def update_by_files(self, state_seq_json, action_seq_json):
        with open(state_seq_json, 'r') as f:
            state_seq = json.load(f)

        with open(action_seq_json, 'r') as f:
            action_seq = json.load(f)
        self.update_state(state_seq,action_seq)
