import re
import re
from collections import defaultdict

def dot_text_to_adjacency_dict(dot_text):
    # Извлекаем все узлы и рёбра
    nodes = set()
    edges = []

    # Находим все узлы (числа, не участвующие в стрелочных связях)
    node_matches = re.findall(r'^\s*(\d+)\s*$', dot_text, re.MULTILINE)
    nodes.update(node_matches)

    # Находим все рёбра (стрелочные связи)
    edge_matches = re.findall(r'(\d+)\s*->\s*(\d+)', dot_text)
    for src, dst in edge_matches:
        nodes.add(src)
        nodes.add(dst)
        edges.append((int(src), int(dst)))

    # Создаем ассоциативный контейнер (словарь словарей)
    adj_dict = defaultdict(dict)

    for src, dst in edges:
        adj_dict[src][dst] = 1

    # Добавляем все узлы, даже если у них нет исходящих рёбер
    for node in nodes:
        if node not in adj_dict:
            adj_dict[node] = {}

    return dict(adj_dict)

def dot_file_to_adjacency_dict(dot_file_path):
    with open(dot_file_path, 'r') as file:
        dot_text = file.read()
    return dot_text_to_adjacency_dict(dot_text)



#matr = dot_file_to_adjacency_dict(r"D:\Универ\Кандидатская Диссертация\Вспомогательные программы\TraceViewer\tmp_files\0\traces\1\PIDG.dot")
#print(matr)