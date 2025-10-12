# map_exporter.py

import os
from qgis.core import (
    QgsProject,
    QgsMapSettings,
    QgsMapRendererParallelJob,
    QgsVectorLayer,
    QgsRectangle
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor

def create_world_file(map_settings, image_file_path):
    """
    Создает файл привязки (world file) для экспортированного изображения.
    """
    file_name, _ = os.path.splitext(image_file_path)
    # Определяем расширение в зависимости от формата изображения
    image_ext = os.path.splitext(image_file_path)[1].lower()
    if image_ext == '.png':
        world_ext = '.pgw'
    elif image_ext in ['.jpg', '.jpeg']:
        world_ext = '.jgw'
    elif image_ext == '.tif':
        world_ext = '.tfw'
    else: # По умолчанию
        world_ext = '.wld'

    world_file_path = file_name + world_ext
    extent = map_settings.extent()
    img_width = map_settings.outputSize().width()
    img_height = map_settings.outputSize().height()

    if img_width == 0 or img_height == 0:
        return False, f"Неверный размер изображения (ширина или высота равна 0) для файла {image_file_path}"

    x_res = extent.width() / img_width
    y_res = -extent.height() / img_height
    x_coord_up_left = extent.xMinimum() + (x_res / 2)
    y_coord_up_left = extent.yMaximum() + (y_res / 2)

    try:
        with open(world_file_path, 'w') as f:
            f.write(f"{x_res}\n{0.0}\n{0.0}\n{y_res}\n{x_coord_up_left}\n{y_coord_up_left}\n")
        print(f"Файл привязки сохранен: {world_file_path}")
        return True, None
    except Exception as e:
        return False, f"Ошибка при создании файла привязки {world_file_path}: {e}"


def export_views(grid_layer: QgsVectorLayer,
                 output_dir: str,
                 image_format: str,
                 width_px: int,
                 height_px: int,
                 dpi: int,
                 progress_reporter=None):
    """
    Экспортирует вид карты для каждого полигона в слое сетки.

    :param grid_layer: Слой сетки, по которому будет производиться экспорт.
    :param output_dir: Папка для сохранения изображений.
    :param image_format: Формат изображения (например, 'png').
    :param width_px: Ширина изображения в пикселях.
    :param height_px: Высота изображения в пикселях.
    :param dpi: Разрешение DPI.
    :param progress_reporter: Экземпляр ProgressReporter для обновления ProgressBar.
    :returns: Кортеж (bool, str). (True, None) при успехе, (False, сообщение_об_ошибке) при неудаче.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана выходная директория: {output_dir}")
        
    project = QgsProject.instance()
    root = project.layerTreeRoot()
    
    visible_layers = [layer for layer in project.mapLayers().values() if root.findLayer(layer.id()).isVisible()]
    if not visible_layers:
        return False, "В проекте нет видимых слоев для экспорта."

    total_features = grid_layer.featureCount()
    if total_features == 0:
        return False, "Слой сетки не содержит объектов для экспорта."

    for i, feature in enumerate(grid_layer.getFeatures()):
        if progress_reporter and progress_reporter.is_canceled():
            return False, "Операция отменена пользователем."
        
        if progress_reporter:
            progress_reporter.set_progress(i + 1, total_features)

        extent = feature.geometry().boundingBox()
        image_name = f"tile_{feature.id()}.{image_format.lower()}"
        file_path = os.path.join(output_dir, image_name)

        settings = QgsMapSettings()
        settings.setLayers(visible_layers)
        settings.setBackgroundColor(QColor(255, 255, 255))
        settings.setOutputSize(QSize(width_px, height_px))
        settings.setOutputDpi(dpi)
        settings.setDestinationCrs(grid_layer.crs())
        settings.setExtent(extent)

        job = QgsMapRendererParallelJob(settings)
        job.start()
        job.waitForFinished()

        # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
        # Было: if job.hasError():
        # Стало: Проверяем, пуст ли список ошибок
        errors = job.errors()
        if errors:
            # Объединяем все сообщения об ошибках в одну строку
            error_string = ", ".join(errors)
            return False, f"Ошибка рендеринга для объекта {feature.id()}: {error_string}"
        
        try:
            img = job.renderedImage()
            img.save(file_path, image_format.lower())
            print(f"Экспортирован файл: {file_path}")

            success, msg = create_world_file(settings, file_path)
            if not success:
                print(f"ПРЕДУПРЕЖДЕНИЕ: {msg}")

        except Exception as e:
            return False, f"Ошибка сохранения файла {file_path}: {e}"

    return True, None