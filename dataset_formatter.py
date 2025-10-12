# dataset_formatter.py

import os
import json
import random
from datetime import datetime, timezone

from qgis.core import QgsVectorLayer, QgsFeatureRequest

def format_yolo_dataset(
    intersected_layer: QgsVectorLayer,
    grid_layer: QgsVectorLayer,
    class_field: str,
    output_dir: str,
    image_format: str,
    image_width: int,
    image_height: int,
    splits: dict,
    metadata: dict,
    delete_void: bool,
    progress_reporter=None
):
    """
    Создает файл data.ndjson и структуру данных для YOLOv8.

    :param intersected_layer: Слой с объектами, пересеченными с сеткой. Должен содержать поле 'ovrly_id'.
    :param grid_layer: Слой сетки, используемый для итерации по тайлам.
    :param class_field: Имя поля с названиями классов в intersected_layer.
    :param output_dir: Корневая папка для датасета.
    :param image_format: Формат изображений (png, jpg).
    :param image_width: Ширина изображений в пикселях.
    :param image_height: Высота изображений в пикселях.
    :param splits: Словарь с разбивкой, например {'train': 80, 'val': 10, 'test': 10}.
    :param metadata: Словарь с метаданными для заголовка ndjson.
    :param delete_void: Если True, тайлы без объектов не будут включены в датасет.
    :param progress_reporter: Экземпляр ProgressReporter для обновления ProgressBar.
    :returns: Кортеж (bool, str). (True, None) при успехе, (False, сообщение_об_ошибке) при неудаче.
    """
    try:
        # --- 1. Подготовка структуры и путей ---
        images_root_dir = os.path.join(output_dir, 'images')
        ndjson_path = os.path.join(output_dir, 'data.ndjson')
        for split_name in splits.keys():
            os.makedirs(os.path.join(images_root_dir, split_name), exist_ok=True)

        # --- 2. Сбор классов и создание заголовка NDJSON ---
        print("Сбор уникальных классов и создание data.ndjson...")
        if intersected_layer.fields().indexFromName(class_field) == -1:
            return False, f"Поле класса '{class_field}' не найдено в слое пересечений."
        
        class_names_list = sorted(list(intersected_layer.uniqueValues(intersected_layer.fields().indexOf(class_field))))
        class_map = {name: i for i, name in enumerate(class_names_list)}
        
        class_names_dict = {str(i): name for i, name in enumerate(class_names_list)}
        now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        with open(ndjson_path, 'w', encoding='utf-8') as f:
            dataset_meta = {
                "type": "dataset", "task": "detect",
                "name": metadata.get('name', 'QGIS Dataset'),
                "description": metadata.get('desc', ''),
                "url": metadata.get('url', ''),
                "class_names": class_names_dict, "bytes": 0, "version": 1,
                "created_at": now_iso, "updated_at": now_iso
            }
            f.write(json.dumps(dataset_meta) + '\n')

        # --- 3. Подготовка к разбивке на train/val/test ---
        split_names = list(splits.keys())
        split_weights = list(splits.values())
        
        # --- 4. Основной цикл по тайлам сетки ---
        total_tiles = grid_layer.featureCount()
        overlay_id_field = 'ovrly_id' # Поле, добавляемое алгоритмом Intersection
        if grid_layer.fields().indexFromName('id') != -1:
            overlay_id_field = 'ovrly_id'
        elif grid_layer.fields().indexFromName('fid') != -1:
            overlay_id_field = 'ovrly_fid'
        else: # Пробуем найти поле по префиксу, если ID не стандартный
            prefix = 'ovrly_'
            original_id = [f.name() for f in grid_layer.fields() if f.name() in ['id', 'fid', 'ID']]
            if original_id:
                overlay_id_field = prefix + original_id[0]
            else:
                 return False, "Не удалось найти поле ID ('id' или 'fid') в слое сетки для связи со слоем пересечений."

        for i, tile_feature in enumerate(grid_layer.getFeatures()):
            if progress_reporter and progress_reporter.is_canceled():
                return False, "Операция отменена."
            if progress_reporter:
                progress_reporter.set_progress(i + 1, total_tiles)

            tile_id = tile_feature.id()
            tile_extent = tile_feature.geometry().boundingBox()
            
            # Ищем все объекты, принадлежащие этому тайлу
            request = QgsFeatureRequest().setFilterExpression(f'"{overlay_id_field}" = {tile_id}')
            features_in_tile = list(intersected_layer.getFeatures(request))
            
            # Пропускаем пустые тайлы, если включена опция
            if not features_in_tile and delete_void:
                continue

            # Определяем, в какую выборку попадет тайл
            split = random.choices(split_names, weights=split_weights, k=1)[0]
            
            # Формируем путь к изображению, которое УЖЕ было создано экспортером
            img_name = f"tile_{tile_id}.{image_format.lower()}"
            relative_img_path = os.path.join('images', split, img_name).replace('\\', '/')
            
            # Перемещаем файл из 'images' в нужную подпапку (train/val/test)
            source_path = os.path.join(images_root_dir, img_name)
            dest_path = os.path.join(output_dir, relative_img_path)
            if os.path.exists(source_path):
                os.rename(source_path, dest_path)
            else:
                print(f"Предупреждение: Исходный файл не найден: {source_path}")

            # Формирование аннотаций
            boxes = []
            for feat_in_tile in features_in_tile:
                class_name = feat_in_tile[class_field]
                if class_name not in class_map: continue
                
                class_id = class_map[class_name]
                bbox = feat_in_tile.geometry().boundingBox()

                # Конвертация в YOLO-формат (0-1)
                box_center_x = (bbox.center().x() - tile_extent.xMinimum()) / tile_extent.width()
                box_center_y = (tile_extent.yMaximum() - bbox.center().y()) / tile_extent.height()
                box_width = bbox.width() / tile_extent.width()
                box_height = bbox.height() / tile_extent.height()
                
                boxes.append([
                    class_id,
                    max(0.0, min(1.0, box_center_x)), max(0.0, min(1.0, box_center_y)),
                    max(0.0, min(1.0, box_width)), max(0.0, min(1.0, box_height))
                ])

            # Запись в NDJSON
            image_record = {
                "type": "image", "file": relative_img_path, "width": image_width, "height": image_height,
                "split": split, "annotations": {"boxes": boxes}
            }
            with open(ndjson_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(image_record) + '\n')
                
        return True, None
    except Exception as e:
        return False, f"Произошла критическая ошибка при форматировании датасета: {e}"