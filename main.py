import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QFrame, QLineEdit, QComboBox, QTextEdit)
from PyQt5.QtCore import Qt
import subprocess
import json
import threading

def run_subprocess(args):
    """
  Функция, которая будет выполняться в отдельном потоке
  для запуска подпроцесса.
  """
    try:
        # subprocess.run() блокирует до завершения подпроцесса,
        # поэтому это будет выполняться в отдельном потоке.
        result = subprocess.run(args, capture_output=True, text=True)
        print(f"Подпроцесс завершен. Код возврата: {result.returncode}")
        print(f"Стандартный вывод:\n{result.stdout}")
        print(f"Стандартная ошибка:\n{result.stderr}")
    except FileNotFoundError:
        print(f"Ошибка: Команда '{args[0]}' не найдена.")
    except Exception as e:
        print(f"Произошла ошибка при выполнении подпроцесса: {e}")


class FileSelectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        self.trace_builder = os.path.join(script_dir, "ProgramGraphAnalysis.exe")

    def initUI(self):
        self.setWindowTitle('Выбор файлов и отображение размеров')
        self.setGeometry(300, 300, 600, 300)

        script_path = os.path.abspath(__file__)
        self.script_dir = os.path.dirname(script_path)

        # Основной вертикальный макет
        main_layout = QVBoxLayout()
        self.main_layout = main_layout

        # Макет для выбора первого файла
        file1_layout = QHBoxLayout()
        self.file1_label = QLabel("Файл первой программы:")
        self.file1_path_label = QLabel("Путь к файлу не выбран")
        self.file1_path_label.setStyleSheet("color: gray;")
        self.select_file1_button = QPushButton("Обзор...")
        self.select_file1_button.clicked.connect(self.select_file1)
        file1_layout.addWidget(self.file1_label)
        file1_layout.addWidget(self.file1_path_label, 1)  # Stretch factor
        file1_layout.addWidget(self.select_file1_button)
        main_layout.addLayout(file1_layout)

        # Разделитель
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator1)

        # Макет для выбора второго файла
        file2_layout = QHBoxLayout()
        self.file2_label = QLabel("Файл второй программы:")
        self.file2_path_label = QLabel("Путь к файлу не выбран")
        self.file2_path_label.setStyleSheet("color: gray;")
        self.select_file2_button = QPushButton("Обзор...")
        self.select_file2_button.clicked.connect(self.select_file2)
        file2_layout.addWidget(self.file2_label)
        file2_layout.addWidget(self.file2_path_label, 1)  # Stretch factor
        file2_layout.addWidget(self.select_file2_button)
        main_layout.addLayout(file2_layout)

        # Разделитель
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator2)

        # Макет для выбора второго файла
        file3_layout = QHBoxLayout()
        self.file3_label = QLabel("Файл конфигурации задания:")
        self.file3_path_label = QLabel("Путь к файлу не выбран")
        self.file3_path_label.setStyleSheet("color: gray;")
        self.select_file3_button = QPushButton("Обзор...")
        self.select_file3_button.clicked.connect(self.select_file3)
        file3_layout.addWidget(self.file3_label)
        file3_layout.addWidget(self.file3_path_label, 1)  # Stretch factor
        file3_layout.addWidget(self.select_file3_button)
        main_layout.addLayout(file3_layout)

        # Разделитель
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator3)

        # Добавляем LineEdit
        self.line_edit_label = QLabel("Введите текст:")
        self.func_name_line_edit = QLineEdit()
        # Добавим плейсхолдер:
        self.func_name_line_edit.setPlaceholderText("Введите что-нибудь")
        # Добавляем horizontal layout
        line_edit_layout = QHBoxLayout()
        line_edit_layout.addWidget(self.line_edit_label)
        line_edit_layout.addWidget(self.func_name_line_edit)
        main_layout.addLayout(line_edit_layout)

        # Разделитель
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.HLine)
        separator4.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator4)

        self.add_text_view_widgets()

        # Кнопка "Выполнить"
        self.execute_button = QPushButton("Выполнить")
        self.execute_button.clicked.connect(self.execute_operation)
        self.execute_button.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        main_layout.addWidget(self.execute_button)

        # Разделитель
        separator5 = QFrame()
        separator5.setFrameShape(QFrame.HLine)
        separator5.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator5)

        # Label для вывода результатов
        self.result_label = QLabel("Размеры выбранных файлов:")
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setWordWrap(True)
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

        # Переменные для хранения путей к файлам
        self.file1_path = None
        self.file2_path = None

    def select_file1(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл первой программы", "",
                                                   "Файлы С/С++ (*.cpp);;Все файлы (*)", options=options)
        if file_path:
            self.file1_path = file_path
            self.file1_path_label.setText(file_path)
            self.file1_path_label.setStyleSheet("color: black;")
            self.load_file_content(file_path, self.text_edit_1)

    def select_file2(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл второй программы", "",
                                                   "Файлы С/С++ (*.cpp);;Все файлы (*)", options=options)
        if file_path:
            self.file2_path = file_path
            self.file2_path_label.setText(file_path)
            self.file2_path_label.setStyleSheet("color: black;")
            self.load_file_content(file_path, self.text_edit_2)

    def select_file3(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл конфигурации задания", "",
                                                   "Файлы JSON (*.json);;Все файлы (*)", options=options)
        if file_path:
            self.file3_path = file_path
            self.file3_path_label.setText(file_path)
            self.file3_path_label.setStyleSheet("color: black;")

    def get_file_size_formatted(self, file_path):
        if not os.path.exists(file_path):
            return "Файл не найден"

        size_in_bytes = os.path.getsize(file_path)
        if size_in_bytes < 1024:
            return f"{size_in_bytes} Байт"
        elif size_in_bytes < 1024 * 1024:
            return f"{size_in_bytes / 1024:.2f} КБайт"
        elif size_in_bytes < 1024 * 1024 * 1024:
            return f"{size_in_bytes / (1024 * 1024):.2f} МБайт"
        else:
            return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} ГБайт"

    def execute_operation(self):
        pg_builder_path = os.path.join(self.script_dir, "PG_builder.py")
        tmp1_path = os.path.join(os.path.join(self.script_dir, "tmp_files"), "1")
        tmp2_path = os.path.join(os.path.join(self.script_dir, "tmp_files"), "2")
        traces1_path = os.path.join(tmp1_path, "traces")
        traces2_path = os.path.join(tmp2_path, "traces")
        pg1_dot_path = os.path.join(tmp1_path, "pg.dot")
        pg2_dot_path = os.path.join(tmp2_path, "pg.dot")
        os.makedirs(tmp1_path, exist_ok=True)
        os.makedirs(tmp2_path, exist_ok=True)
        os.makedirs(traces1_path, exist_ok=True)
        os.makedirs(traces2_path, exist_ok=True)
        func_name = ""
        results = []

        self.test1_path = traces1_path
        self.test2_path = traces2_path

        if self.file3_path:
            print("JSON OK")
            json_path = self.file3_path
            with open(json_path, 'r') as task_config_json_file:
                task_config = json.load(task_config_json_file)
                func_name = task_config["function_name"]
                input_variables = task_config["input_variables"]
                test_cases = task_config["test_cases"]
                if self.file1_path:
                    subprocess.run(['python', pg_builder_path, self.file1_path, func_name, pg1_dot_path], cwd=tmp1_path)
                    if self.build_traces_for_tests(pg1_dot_path, input_variables, test_cases, traces1_path):
                        results.append("Трассы для программы №1 построены\n")
                else:
                    results.append("Первый файл не выбран.")

                if self.file2_path:
                    subprocess.run(['python', pg_builder_path, self.file2_path, func_name, pg2_dot_path], cwd=tmp2_path)
                    if self.build_traces_for_tests(pg2_dot_path, input_variables, test_cases, traces2_path):
                        results.append("Трассы для программы №2 построены\n")
                        self.add_select_test_widget(len(test_cases))

                else:
                    results.append("Второй файл не выбран.")

        else:
            results.append("Файл конфигурации задания не выбран.")

        self.result_label.setText("\n".join(results))

    def load_file_content(self, file_path, text_edit):
        """
        Загружает содержимое файла в QTextEdit.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                text_edit.setText(content)
        except Exception as e:
            text_edit.setText(f"Ошибка при загрузке файла: {e}")

    def add_text_view_widgets(self):
        # Создаем Horizontal Layout для QTextEdit'ов
        horizontal_layout = QHBoxLayout()

        # Создаем виджеты QTextEdit
        self.text_edit_1 = QTextEdit()
        self.text_edit_2 = QTextEdit()

        # Добавляем QTextEdit'ы в Horizontal Layout
        horizontal_layout.addWidget(self.text_edit_1)
        horizontal_layout.addWidget(self.text_edit_2)

        # Добавляем Horizontal Layout в основной Vertical Layout
        self.main_layout.addLayout(horizontal_layout)

    def add_select_test_widget(self, tests_number):
        label = QLabel("Выбрать номер теста:".format(tests_number))
        self.main_layout.addWidget(label)

        # Создаем QComboBox (Select виджет)
        self.deal_combo = QComboBox()
        self.deal_combo.addItem("Выберите номер теста")

        # Заполняем QComboBox номерами сделок (от 1 до количества элементов в массиве)
        for i in range(1, tests_number + 1):
            self.deal_combo.addItem(str(i))  # Добавляем номер сделки в QComboBox

        # Подключаем сигнал currentIndexChanged к функции, которая будет вызываться при изменении выбранного элемента
        self.deal_combo.currentIndexChanged.connect(self.deal_selected)

        self.main_layout.addWidget(self.deal_combo)

        self.setLayout(self.main_layout)

    def deal_selected(self, index):
        selected_text = self.deal_combo.currentText()
        if selected_text != "Выберите номер теста":
            # Удаляем "Выберите значение"
            index_to_remove = self.deal_combo.findText("Выберите номер теста")
            if index_to_remove >= 0:  # Проверяем, существует ли он еще
                self.deal_combo.removeItem(index_to_remove)  # Удаляем элемент

        program1_test_path = os.path.join(self.test1_path, selected_text)
        program2_test_path = os.path.join(self.test2_path, selected_text)
        state_seq1_file_path = os.path.join(program1_test_path, "state_sequence.json")
        state_seq2_file_path = os.path.join(program2_test_path, "state_sequence.json")
        read_seq1_file_path = os.path.join(program1_test_path, "read_var_sequence.json")
        read_seq2_file_path = os.path.join(program2_test_path, "read_var_sequence.json")
        visualizer_file_path = os.path.join(self.script_dir, "visual2.py")

        # --- Определение пути к исполняемому файлу Python в venv ---
        venv_path = os.path.join(os.path.dirname(__file__), 'venv')
        if sys.platform == "win32":
            python_executable = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:  # Linux/macOS
            python_executable = os.path.join(venv_path, 'bin', 'python')
        args = [python_executable, visualizer_file_path, state_seq1_file_path, state_seq2_file_path,
                read_seq1_file_path, read_seq2_file_path]
        print(args)
        thread = threading.Thread(target=run_subprocess, args=(args,))
        # Запускаем поток
        thread.start()
        # subprocess.run(args)

    def build_traces_for_tests(self, pg_dot_file_path, input_variables, test_cases, program_test_root_path):
        print("build tests...")
        test_num = 0
        for test in test_cases:
            test_num += 1
            test_dir = os.path.join(program_test_root_path, str(test_num))
            print(test_dir)
            os.makedirs(test_dir, exist_ok=True)
            arg_list = [self.trace_builder, pg_dot_file_path]
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileSelectorApp()
    ex.show()
    sys.exit(app.exec_())
