# yaml_converter.py

import os
try:
    import yaml
except ImportError:
    # Оставляем переменную yaml как None, если библиотека не найдена
    yaml = None

def create_yolo_yaml(output_dir: str, class_names: dict):
    """
    Создает файл dataset.yaml, необходимый для обучения моделей YOLO.

    :param output_dir: Корневая директория, где сохранен датасет.
    :param class_names: Словарь, сопоставляющий индекс класса (str) с именем класса (str).
    :returns: Кортеж (bool, str). (True, сообщение об успехе) или (False, сообщение об ошибке).
    """
    if yaml is None:
        error_msg = ("Библиотека PyYAML не найдена. Невозможно создать dataset.yaml.\n"
                     "Пожалуйста, установите ее, например, через OSGeo4W Shell: python -m pip install pyyaml")
        print(f"ОШИБКА: {error_msg}")
        return False, error_msg

    try:
        # Для YAML принято использовать целочисленные ключи
        names_map = {int(k): v for k, v in class_names.items()}

        # Структура данных для YAML файла
        yaml_data = {
            'path': os.path.abspath(output_dir).replace('\\', '/'),  # Абсолютный путь с forward slashes
            'train': os.path.join('images', 'train').replace('\\', '/'), # Относительные пути
            'val': os.path.join('images', 'val').replace('\\', '/'),
            'test': os.path.join('images', 'test').replace('\\', '/'),
            'names': names_map
        }

        yaml_file_path = os.path.join(output_dir, 'dataset.yaml')

        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            # sort_keys=False сохраняет порядок полей, как в yaml_data
            yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

        success_msg = f"Файл {yaml_file_path} успешно создан."
        print(success_msg)
        return True, success_msg

    except Exception as e:
        error_msg = f"Произошла ошибка при создании YAML файла: {e}"
        print(error_msg)
        return False, error_msg