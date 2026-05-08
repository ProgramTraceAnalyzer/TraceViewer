from trace_builder import *
from similarity_matrix import *
import os
import json
import sys
from metrics import lcs, dtw
from matrix_agregation import *
from hungarian import *

def matrix_and_mapping_for_traces(prog1_path, prog2_path, treshhold, metrics, remove_stutter, remove_not_used):
    traces1_folder = prog1_path  # os.path.join(prog1_path,"traces")
    traces2_folder = prog2_path  # os.path.join(prog2_path,"traces")
    matrix_list = []
    mapping_list = []
    matrix_list_raw = []
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
        matrix_list.append(similarity_matrix)
        matrix_list_raw.append(matrix)
        mapping = hungarian_mapping(similarity_matrix,treshhold)
        mapping_list.append(mapping["mapping"])
    print("MAPPING LIST: ",mapping_list)
    statistics = get_pair_statistics(mapping_list)
    agg = agregate_matrixes(matrix_list_raw)
    total_mapping = hungarian_mapping(agg, treshhold)
    return statistics, mapping_list, matrix_list, agg, total_mapping["mapping"]

def generate_html_report(test_cases, statistics, mapping_list, matrix_list, agg, total_mapping):
    html = ""
    test_count = len(matrix_list)
    for i in range(0,test_count):
        html += f"<h1>Тест №{i+1}</h1><br>"
        html+="вход: <br>"
        for variable, value in test_cases[i]["data"].items():
            html += variable + "=" + str(value) + "<br>"

        html+="<br>"
        html+=similarity_matrix_to_html(matrix_list[i],mapping_list[i])
        #html+=dict_to_html_table(mapping_list[i])
        html+="<hr/>"
    html += f"<h1>Усредненная матрица</h1>"
    html+= "total_mapping = "+str(total_mapping)
    html += similarity_matrix_to_html(agg, total_mapping)
    return html

from html import escape

def dict_to_html_table(d):
    rows = []
    for key, value in d.items():
        rows.append(
            f"<tr><td>{escape(str(key))}</td><td>{escape(str(value))}</td></tr>"
        )
    return "<table border=\"1\" cellspacing=\"0\" cellpadding=\"6\" style=\"border-collapse: collapse; text-align: center;\">\n" + "\n".join(rows) + "\n</table>"

def similarity_matrix_to_html(similarity_matrix, mapping):
    """
    similarity_matrix: dict вида
    {
        "a": {"w": 45, "h": 80},
        "b": {"w": 83, "h": 21}
    }

    Возвращает HTML-строку с таблицей.
    """

    def lerp(a, b, t):
        return int(round(a + (b - a) * t))

    def color_for_percent(value):
        value = max(0, min(100, value))

        # 0%   -> бледно-розовый
        # 50%  -> бледно-желтый
        # 100% -> бледно-зеленый
        c0 = (248, 215, 218)  # light pink
        c50 = (255, 243, 205) # light yellow
        c100 = (212, 237, 218) # light green

        if value <= 50:
            t = value / 50
            r = lerp(c0[0], c50[0], t)
            g = lerp(c0[1], c50[1], t)
            b = lerp(c0[2], c50[2], t)
        else:
            t = (value - 50) / 50
            r = lerp(c50[0], c100[0], t)
            g = lerp(c50[1], c100[1], t)
            b = lerp(c50[2], c100[2], t)

        return f"rgb({r}, {g}, {b})"

    if not similarity_matrix:
        return "<table></table>"

    col_names = []
    seen = set()
    for row in similarity_matrix.values():
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                col_names.append(key)

    parts = []
    parts.append("""
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse; text-align: center;">
    <tr>
        <th></th>
""".strip())

    for col in col_names:
        parts.append(f'        <th>{escape(str(col))}</th>')

    parts.append("    </tr>")

    for row_name, row_data in similarity_matrix.items():
        parts.append("    <tr>")
        parts.append(f'        <th>{escape(str(row_name))}</th>')

        for col in col_names:
            value = row_data.get(col, 0)
            border_px = 1
            print("ROW NAME: ",row_name,"  COL NAME: ",str(col))
            if row_name in mapping.keys():
                if mapping[row_name]==str(col):
                    border_px = 5


            bg = color_for_percent(value)
            parts.append(
                f'        <td style="border: {border_px}px solid #009542; background-color: {bg};">{value:.2f}%</td>'
            )

        parts.append("    </tr>")

    parts.append("</table>")

    return "\n".join(parts)

import json

def main():
    prog1_path = r"D:\Универ\Кандидатская Диссертация\Вспомогательные программы\TraceViewer\tmp_files\0\traces"
    prog2_path = r"D:\Универ\Кандидатская Диссертация\Вспомогательные программы\TraceViewer\tmp_files\1\traces"
    config_path = r"D:\Гугл-Диск\КулюкинКС_кандидатская\Сравнение трасс программ\Наши разработки\Эксперименты\Эксперимент_май2025\3\task_config.json"

    config = {}
    with open(config_path,"r",encoding="utf-8") as f:
        config = json.load(f)

    test_cases = config["test_cases"]

    treshhold = 40
    metrics = lcs
    remove_stutter = True
    remove_not_used = True

    statistics, mapping_list, matrix_list, agg, total_mapping = matrix_and_mapping_for_traces(prog1_path, prog2_path, treshhold, metrics,
                                                                          remove_stutter, remove_not_used)
    html = generate_html_report(test_cases, statistics, mapping_list, matrix_list, agg, total_mapping)
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)

main()
