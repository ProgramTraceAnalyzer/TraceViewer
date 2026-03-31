import json

report = {}
with open("report.json","r",encoding="utf-8") as file:
    report = json.load(file)


def dict_to_html_table(data):
    # Собираем все возможные заголовки столбцов из всех строк
    all_headers = set()
    for row_data in data.values():
        all_headers.update(row_data.keys())
    headers = sorted(list(all_headers))  # Сортируем для единообразия

    # Начинаем создание HTML таблицы
    html = '<table border="1" style="border-collapse: collapse;">\n'

    # Добавляем заголовок таблицы
    html += '  <tr>\n'
    html += '    <th></th>\n'  # Пустая ячейка для заголовка строк
    for header in headers:
        html += f'    <th>{header}</th>\n'
    html += '  </tr>\n'

    # Добавляем строки с данными
    for row_key, row_data in data.items():
        html += '  <tr>\n'
        html += f'    <th>{row_key}</th>\n'  # Заголовок строки

        for header in headers:
            # Если данные есть - показываем, если нет - пустая ячейка
            value = (row_data.get(header, ''))
            td_text = ""
            print(type(value))
            if type(value) == dict:
                val_obj = (value)
                distance_dict = val_obj["distance_dict"]
                for metric in distance_dict.keys():
                    td_text += "<b>" + metric + "</b>:" + str(distance_dict[metric]["similarity"]) + "%<br>"

            html += f'    <td>{td_text}</td>\n'

        html += '  </tr>\n'

    html += '</table>'
    return html

file_html = ""
for test_num in report.keys():
    file_html += f"<h1>Тест №{test_num}</h1>"
    file_html += dict_to_html_table(report[test_num])

with open("report.html","w",encoding="utf-8") as file:
    file.write(file_html)