# -*- coding: utf-8 -*-

"""
================================================================================
 ЕДИНЫЙ СКРИПТ ДЛЯ ЭКСПОРТА ВИДА КАРТЫ В ИЗОБРАЖЕНИЕ В QGIS
 (Версия с исправлением ошибки TypeError и добавлением файла привязки)
================================================================================
"""

import os
from qgis.core import (
    QgsProject,
    QgsMapSettings,
    QgsMapRendererParallelJob,
    QgsPointXY,
    QgsCoordinateReferenceSystem,
    QgsRectangle
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor

# --- 1. НАСТРОЙКИ ЭКСПОРТА (ИЗМЕНИТЕ ЭТИ ЗНАЧЕНИЯ) ---

output_folder = 'C:/temp/'
image_name = 'final_map_export2.png'
img_width_pixels = 1300
img_height_pixels = 1300
dpi = 512
center_x = 9259919.29782563
center_y = 7315233.58836072
crs_string = 'EPSG:3395'

# --- 2. ОСНОВНОЙ КОД (НИЧЕГО МЕНЯТЬ НЕ НУЖНО) ---

def create_world_file(settings, image_file_path):
    """
    Создает файл привязки (world file) для экспортированного изображения.
    """
    print("--- Создание файла привязки ---")
    
    # Определение имени файла привязки (.pgw для .png)
    file_name, _ = os.path.splitext(image_file_path)
    world_file_path = file_name + '.pgw'

    # Получаем экстент и размеры из настроек рендеринга
    extent = settings.extent()
    img_width = settings.outputSize().width()
    img_height = settings.outputSize().height()

    # Рассчитываем параметры файла привязки
    # A: Размер пикселя по оси X
    x_res = extent.width() / img_width
    # D: Поворот по оси Y (обычно 0)
    d_rot = 0
    # B: Поворот по оси X (обычно 0)
    b_rot = 0
    # E: Размер пикселя по оси Y (отрицательный, т.к. начало координат вверху)
    y_res = -extent.height() / img_height
    # C: X-координата центра верхнего левого пикселя
    x_coord_up_left = extent.xMinimum() + (x_res / 2)
    # F: Y-координата центра верхнего левого пикселя
    y_coord_up_left = extent.yMaximum() + (y_res / 2)

    try:
        with open(world_file_path, 'w') as f:
            f.write(f"{x_res}\n")
            f.write(f"{d_rot}\n")
            f.write(f"{b_rot}\n")
            f.write(f"{y_res}\n")
            f.write(f"{x_coord_up_left}\n")
            f.write(f"{y_coord_up_left}\n")
        print(f"УСПЕХ! Файл привязки сохранен в: {world_file_path}")
    except Exception as e:
        print(f"ОШИБКА при создании файла привязки: {e}")


def export_map():
    """
    Основная функция для рендеринга и сохранения карты.
    """
    file_path = os.path.join(output_folder, image_name)
    
    if not os.path.exists(output_folder):
        print(f"ОШИБКА: Папка '{output_folder}' не существует.")
        return

    print("--- Начинается процесс экспорта карты ---")
    
    settings = QgsMapSettings()

    root = QgsProject.instance().layerTreeRoot()
    visible_layers = []
    for layer in QgsProject.instance().mapLayers().values():
        node = root.findLayer(layer.id())
        if node and node.isVisible():
            visible_layers.append(layer)

    if not visible_layers:
        print("ОШИБКА: В проекте нет видимых слоев для экспорта.")
        return
        
    settings.setLayers(visible_layers)
    settings.setBackgroundColor(QColor(255, 255, 255))
    settings.setOutputSize(QSize(img_width_pixels, img_height_pixels))
    settings.setOutputDpi(dpi)
    
    crs = QgsCoordinateReferenceSystem(crs_string)
    settings.setDestinationCrs(crs)
    
    # ***************************************************************
    # РАСЧЕТ И УСТАНОВКА ЭКСТЕНТА ДЛЯ ПЛОСКИХ КООРДИНАТ
    # ***************************************************************
    center_point = QgsPointXY(center_x, center_y)
    
    # Рассчитываем ширину и высоту экстента в метрах
    # Формула: (размер_в_пикселях / dpi) * дюймов_в_метре * масштаб
    inch_per_meter = 39.3701
    map_width = (img_width_pixels / dpi) * inch_per_meter 
    map_height = (img_height_pixels / dpi) * inch_per_meter

    # Создаем прямоугольник экстента
    x_min = center_point.x() - (map_width / 2.0)
    x_max = center_point.x() + (map_width / 2.0)
    y_min = center_point.y() - (map_height / 2.0)
    y_max = center_point.y() + (map_height / 2.0)
    
    calculated_extent = QgsRectangle(x_min, y_min, x_max, y_max)
    
    # Устанавливаем рассчитанный экстент
    settings.setExtent(calculated_extent)
    
    print(f"Параметры рендеринга:")
    print(f"  - Размер: {settings.outputSize().width()}x{settings.outputSize().height()} px")
    print(f"  - Разрешение: {settings.outputDpi()} DPI")
    print(f"  - Центр (в {crs.authid()}): {center_x:.2f}, {center_y:.2f}")
    print(f"  - Рассчитанный экстент: {calculated_extent.toString()}")

    render = QgsMapRendererParallelJob(settings)

    def on_finished():
        try:
            img = render.renderedImage()
            img.save(file_path, "png")
            print("--------------------------------------------------")
            print(f"УСПЕХ! Карта сохранена в: {file_path}")
            
            # Создаем файл привязки после сохранения изображения
            create_world_file(settings, file_path)

            print("--------------------------------------------------")
        except Exception as e:
            print(f"ОШИБКА при сохранении файла: {e}")

    render.finished.connect(on_finished)
    render.start()
    
    global qgis_render_job
    qgis_render_job = render

    print("\nРендеринг запущен в фоновом режиме. Ожидайте сообщения о завершении...")

# --- 3. ЗАПУСК СКРИПТА ---
export_map()