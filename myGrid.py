# grid_creator.py

import processing
from qgis.core import QgsVectorLayer, QgsWkbTypes

def create_grid_layer(source_layer: QgsVectorLayer, h_spacing: float, v_spacing: float, h_overlay: float, v_overlay: float):
    """
    Создает и возвращает временный полигональный слой сетки на основе экстента исходного слоя.

    :param source_layer: Векторный слой, экстент которого используется для создания сетки.
    :param h_spacing: Горизонтальный шаг (ширина ячейки без перекрытия).
    :param v_spacing: Вертикальный шаг (высота ячейки без перекрытия).
    :param h_overlay: Горизонтальное перекрытие.
    :param v_overlay: Вертикальное перекрытие.
    :returns: Кортеж (QgsVectorLayer, str). В случае успеха возвращает (слой_сетки, None).
              В случае ошибки возвращает (None, сообщение_об_ошибке).
    """
    if not source_layer or not source_layer.isValid():
        return None, "Исходный слой для определения экстента недействителен."

    try:
        extent = source_layer.extent()
        crs = source_layer.crs()
        
        # Проверка, что экстент корректен
        if extent.isEmpty():
            return None, f"Экстент слоя '{source_layer.name()}' пуст. Невозможно создать сетку."

        extent_str = f'{extent.xMinimum()},{extent.xMaximum()},{extent.yMinimum()},{extent.yMaximum()}'

        parameters = {
            'TYPE': 2,  # Тип сетки: Полигоны (Rectangles)
            'EXTENT': extent_str,
            'HSPACING': h_spacing,
            'VSPACING': v_spacing,
            'HOVERLAY': h_overlay,
            'VOVERLAY': v_overlay,
            'CRS': crs,
            'OUTPUT': 'memory:'  # Создаем временный слой в памяти
        }

        # Запуск алгоритма QGIS
        result = processing.run("qgis:creategrid", parameters)
        grid_layer = result.get('OUTPUT')

        if not grid_layer or not isinstance(grid_layer, QgsVectorLayer):
            return None, "Алгоритм 'creategrid' не вернул корректный векторный слой."
        
        # Проверка, что слой не пустой
        if grid_layer.featureCount() == 0:
             return None, "Созданная сетка не содержит ни одного объекта. Проверьте параметры (экстент, шаг)."

        return grid_layer, None  # Успешное выполнение

    except Exception as e:
        # Перехватываем любые ошибки во время выполнения алгоритма
        error_message = f"Произошла ошибка при выполнении алгоритма 'creategrid': {e}"
        print(error_message)
        return None, error_message