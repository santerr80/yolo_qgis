# intersection.py

import processing
from qgis.core import QgsVectorLayer

def perform_intersection(input_layer: QgsVectorLayer, overlay_layer: QgsVectorLayer, prefix: str = 'ovrly_'):
    """
    Выполняет операцию пересечения (Intersection) между двумя векторными слоями.

    :param input_layer: Основной входной слой.
    :param overlay_layer: Слой, с которым выполняется пересечение.
    :param prefix: Префикс для полей из слоя 'overlay_layer' во избежание дублирования.
    :returns: Кортеж (QgsVectorLayer, str). В случае успеха возвращает (результирующий_слой, None).
              В случае ошибки возвращает (None, сообщение_об_ошибке).
    """
    if not input_layer or not input_layer.isValid():
        return None, "Основной входной слой (Input) недействителен."
    
    if not overlay_layer or not overlay_layer.isValid():
        return None, "Слой для пересечения (Overlay) недействителен."

    try:
        parameters = {
            'INPUT': input_layer,
            'OVERLAY': overlay_layer,
            'INPUT_FIELDS': [],  # Берем все поля из первого слоя
            'OVERLAY_FIELDS': [], # Берем все поля из второго слоя
            'OVERLAY_FIELDS_PREFIX': prefix,
            'OUTPUT': 'memory:'  # Результат будет создан как временный слой
        }

        # Запускаем алгоритм "Intersection"
        result = processing.run("native:intersection", parameters)
        output_layer = result.get('OUTPUT')

        if not output_layer or not isinstance(output_layer, QgsVectorLayer):
            return None, "Алгоритм 'intersection' не вернул корректный векторный слой."

        return output_layer, None # Успешное выполнение

    except Exception as e:
        error_message = f"Произошла ошибка при выполнении алгоритма 'intersection': {e}"
        print(error_message)
        return None, error_message