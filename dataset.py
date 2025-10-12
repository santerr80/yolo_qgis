# -*- coding: utf-8 -*-

from qgis.core import (
    QgsProject,
    QgsFeatureRequest,
    QgsRectangle,
    QgsApplication
)
from qgis.utils import iface
import os
import json
import random
from datetime import datetime, timezone

# --- НАЧАЛО: НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ---

# Имя векторного слоя (как оно отображается в QGIS)
vector_layer_name = 'Objects' 
# Имя поля, содержащего названия классов
class_field_name = 'class_name'
# Путь к папке, куда будет сохранен набор данных
output_dir = r'C:\Temp'
# Размер области захвата в единицах карты (например, в метрах)
view_size = 150
# Процент изображений, которые пойдут в обучающую выборку (остальные - в валидационную)
train_ratio_percent = 80
# Формат изображений: 'png' или 'jpg'
image_format = 'png'

# --- Метаданные для файла data.ndjson ---
dataset_name = 'My QGIS Dataset'
dataset_desc = 'Dataset generated via QGIS Python console script.'
dataset_url = 'local'

# --- КОНЕЦ: НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ---


def run_yolo_dataset_creation():
    """
    Основная функция для создания набора данных YOLO.
    """
    # --- 1. ПОЛУЧЕНИЕ И ПРОВЕРКА ПАРАМЕТРОВ ---
    
    # Поиск слоя в проекте
    layers = QgsProject.instance().mapLayersByName(vector_layer_name)
    if not layers:
        print(f"ОШИБКА: Слой с именем '{vector_layer_name}' не найден в проекте.")
        return
    vector_layer = layers[0]

    # Проверка наличия поля класса
    if vector_layer.fields().indexFromName(class_field_name) == -1:
        print(f"ОШИБКА: Поле с именем '{class_field_name}' не найдено в слое '{vector_layer_name}'.")
        return
        
    # Проверка системы координат
    if vector_layer.crs().isGeographic():
        print("ОШИБКА: Слой находится в географической системе координат (градусы). "
              "Пожалуйста, перепроецируйте его в проекционную СК (например, UTM) для корректного измерения в метрах.")
        return

    train_ratio = train_ratio_percent / 100.0

    # --- 2. ПОДГОТОВКА СТРУКТУРЫ ПАПОК И ФАЙЛОВ ---
    print("Подготовка структуры папок...")
    images_dir = os.path.join(output_dir, 'images')
    train_dir = os.path.join(images_dir, 'train')
    val_dir = os.path.join(images_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    ndjson_path = os.path.join(output_dir, 'data.ndjson')

    # --- 3. СБОР КЛАССОВ И СОЗДАНИЕ ФАЙЛА NDJSON ---
    print("Сбор уникальных классов...")
    class_names_list = sorted(list(vector_layer.uniqueValues(vector_layer.fields().indexOf(class_field_name))))
    class_map = {name: i for i, name in enumerate(class_names_list)}
    
    # Создаем или перезаписываем заголовок NDJSON
    print("Создание файла data.ndjson с метаданными...")
    class_names_dict = {str(i): name for i, name in enumerate(class_names_list)}
    now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    with open(ndjson_path, 'w', encoding='utf-8') as f:
        dataset_meta = {
            "type": "dataset", "task": "detect", "name": dataset_name, "description": dataset_desc,
            "url": dataset_url, "class_names": class_names_dict, "bytes": 0, "version": 1,
            "created_at": now_iso, "updated_at": now_iso
        }
        f.write(json.dumps(dataset_meta) + '\n')

    # --- 4. ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ---
    canvas = iface.mapCanvas()
    processed_feature_ids = set()
    features = list(vector_layer.getFeatures())
    feature_count = len(features)
    
    print(f"Найдено {feature_count} объектов для обработки.")

    for i, feature in enumerate(features):
        
        # Пропускаем объект, если он уже был включен в какой-либо снимок
        # if feature.id() in processed_feature_ids:
        #     progress = int((i + 1) * 100 / feature_count)
        #     print(f"Прогресс: {progress}% (объект {feature.id()} уже обработан)")
        #     continue

        # Центрирование холста на объекте
        geom = feature.geometry()
        centroid = geom.centroid().asPoint()
        half_size = view_size / 2
        new_extent = QgsRectangle(
            centroid.x() - half_size, centroid.y() - half_size,
            centroid.x() + half_size, centroid.y() + half_size
        )
        canvas.setExtent(new_extent)
        canvas.refresh()
        QgsApplication.processEvents() # Даем QGIS время на перерисовку

        # Поиск всех объектов, попавших в новый кадр
        request = QgsFeatureRequest().setFilterRect(new_extent)
        features_in_view = list(vector_layer.getFeatures(request))

        if not features_in_view:
            continue

        # Сохранение изображения
        img_width = canvas.size().width()
        img_height = canvas.size().height()
        split = 'train' if random.random() < train_ratio else 'val'
        
        # Генерируем уникальное имя файла
        img_name = f"img_{len(os.listdir(train_dir)) + len(os.listdir(val_dir)):06d}.{image_format}"
        relative_img_path = os.path.join('images', split, img_name).replace('\\', '/')
        output_img_path = os.path.join(output_dir, relative_img_path)
        
        print(f"Сохранение изображения: {output_img_path}")
        canvas.saveAsImage(output_img_path)

        # Формирование аннотаций
        boxes = []
        for feat_in_view in features_in_view:
            processed_feature_ids.add(feat_in_view.id()) # Отмечаем как обработанный
            class_name = feat_in_view[class_field_name]
            if class_name not in class_map: 
                continue
            
            class_id = class_map[class_name]
            bbox = feat_in_view.geometry().boundingBox()

            # Конвертация в YOLO-формат (0-1)
            box_center_x = (bbox.center().x() - new_extent.xMinimum()) / new_extent.width()
            box_center_y = (new_extent.yMaximum() - bbox.center().y()) / new_extent.height()
            box_width = bbox.width() / new_extent.width()
            box_height = bbox.height() / new_extent.height()
            
            # Ограничиваем значения диапазоном [0.0, 1.0]
            boxes.append([
                class_id, 
                max(0.0, min(1.0, box_center_x)), 
                max(0.0, min(1.0, box_center_y)),
                max(0.0, min(1.0, box_width)), 
                max(0.0, min(1.0, box_height))
            ])
        
        # Запись в NDJSON
        image_record = {
            "type": "image", "file": relative_img_path, "width": img_width, "height": img_height,
            "split": split, "annotations": {"boxes": boxes}
        }
        with open(ndjson_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(image_record) + '\n')

        # Обновление прогресса в консоли
        progress = int((i + 1) * 100 / feature_count)
        print(f"Прогресс: {progress}%")

    print(f"Обработка завершена. Набор данных сохранен в: {output_dir}")

# --- ЗАПУСК СКРИПТА ---
run_yolo_dataset_creation()