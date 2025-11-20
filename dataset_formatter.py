# dataset_formatter.py

import os
import json
import random
from datetime import datetime, timezone

from qgis.core import QgsVectorLayer, QgsFeatureRequest, QgsWkbTypes


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
    progress_reporter=None,
):
    """
    Создает файл data.ndjson и структуру данных для YOLOv8.
    Поддерживает задачи 'detect' (рамки) и 'segment' (полигоны).
    """
    try:
        task_type = metadata.get(
            "task", "detect"
        ).lower()  # Получаем тип задачи из метаданных

        # --- 1. Подготовка структуры и путей ---
        # В режиме обновления изображения уже экспортированы в temp_new_data/images
        # В обычном режиме они в output_dir/images
        if "temp_new_data" in output_dir:
            images_root_dir = os.path.join(output_dir, "images")
        else:
            images_root_dir = os.path.join(output_dir, "images")
        ndjson_path = os.path.join(output_dir, "data.ndjson")
        for split_name in splits.keys():
            os.makedirs(os.path.join(images_root_dir, split_name), exist_ok=True)

        # --- 2. Сбор классов и создание заголовка NDJSON ---
        print(f"Сбор уникальных классов для задачи '{task_type}'...")
        if intersected_layer.fields().indexFromName(class_field) == -1:
            return False, f"Поле класса '{class_field}' не найдено в слое пересечений."

        class_names_list = sorted(
            list(
                intersected_layer.uniqueValues(
                    intersected_layer.fields().indexOf(class_field)
                )
            )
        )
        class_map = {name: i for i, name in enumerate(class_names_list)}

        class_names_dict = {str(i): name for i, name in enumerate(class_names_list)}
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        with open(ndjson_path, "w", encoding="utf-8") as f:
            dataset_meta = {
                "type": "dataset",
                "task": task_type,  # <--- ИСПОЛЬЗУЕМ ТИП ЗАДАЧИ
                "name": metadata.get("name", "QGIS Dataset"),
                "description": metadata.get("desc", ""),
                "url": metadata.get("url", ""),
                "class_names": class_names_dict,
                "bytes": 0,
                "version": 1,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            f.write(json.dumps(dataset_meta) + "\n")

        # --- 3. Подготовка к разбивке на train/val/test ---
        split_names = list(splits.keys())
        split_weights = list(splits.values())

        # --- 4. Основной цикл по тайлам сетки ---
        total_tiles = grid_layer.featureCount()
        prefix = "ovrly_"
        id_field_names = [
            f.name() for f in grid_layer.fields() if f.name().lower() in ["id", "fid"]
        ]
        if not id_field_names:
            return False, "Не удалось найти поле ID ('id' или 'fid') в слое сетки."
        overlay_id_field = prefix + id_field_names[0]

        for i, tile_feature in enumerate(grid_layer.getFeatures()):
            if progress_reporter and progress_reporter.is_canceled():
                return False, "Операция отменена."
            if progress_reporter:
                progress_reporter.set_progress(i + 1, total_tiles)

            tile_id = tile_feature.id()
            tile_extent = tile_feature.geometry().boundingBox()
            img_name = f"tile_{tile_id}.{image_format.lower()}"

            request = QgsFeatureRequest().setFilterExpression(
                f'"{overlay_id_field}" = {tile_id}'
            )
            features_in_tile = list(intersected_layer.getFeatures(request))

            if not features_in_tile and delete_void:
                image_to_delete_path = os.path.join(images_root_dir, img_name)
                if os.path.exists(image_to_delete_path):
                    try:
                        os.remove(image_to_delete_path)
                        print(f"Удален пустой тайл: {image_to_delete_path}")
                    except OSError as e:
                        print(
                            f"Ошибка при удалении пустого тайла {image_to_delete_path}: {e}"
                        )
                continue

            split = random.choices(split_names, weights=split_weights, k=1)[0]
            relative_img_path = os.path.join("images", split, img_name).replace(
                "\\", "/"
            )
            source_path = os.path.join(images_root_dir, img_name)
            dest_path = os.path.join(output_dir, relative_img_path)
            if os.path.exists(source_path):
                os.rename(source_path, dest_path)

            annotations = {}
            # --- ИЗМЕНЕНИЕ: ЛОГИКА В ЗАВИСИМОСТИ ОТ ТИПА ЗАДАЧИ ---
            if task_type == "detect":
                boxes = []
                for feat in features_in_tile:
                    class_id = class_map.get(feat[class_field])
                    if class_id is None:
                        continue
                    bbox = feat.geometry().boundingBox()
                    box_center_x = (
                        bbox.center().x() - tile_extent.xMinimum()
                    ) / tile_extent.width()
                    box_center_y = (
                        tile_extent.yMaximum() - bbox.center().y()
                    ) / tile_extent.height()
                    box_width = bbox.width() / tile_extent.width()
                    box_height = bbox.height() / tile_extent.height()
                    boxes.append(
                        [
                            class_id,
                            max(0.0, min(1.0, box_center_x)),
                            max(0.0, min(1.0, box_center_y)),
                            max(0.0, min(1.0, box_width)),
                            max(0.0, min(1.0, box_height)),
                        ]
                    )
                if boxes:
                    annotations["boxes"] = boxes

            elif task_type == "segment":
                segments = []
                for feat in features_in_tile:
                    class_id = class_map.get(feat[class_field])
                    if class_id is None:
                        continue

                    geom = feat.geometry()
                    # Проверяем, что это полигональная геометрия
                    if geom.wkbType() not in [
                        QgsWkbTypes.Polygon,
                        QgsWkbTypes.MultiPolygon,
                    ]:
                        continue

                    # Обрабатываем мультиполигоны как набор отдельных полигонов
                    polygons = (
                        geom.asMultiPolygon()
                        if geom.isMultipart()
                        else [geom.asPolygon()]
                    )

                    for poly in polygons:
                        # Берем только внешнее кольцо, игнорируя дырки
                        exterior_ring = poly[0]
                        normalized_points = []
                        for point in exterior_ring:
                            norm_x = (
                                point.x() - tile_extent.xMinimum()
                            ) / tile_extent.width()
                            norm_y = (
                                tile_extent.yMaximum() - point.y()
                            ) / tile_extent.height()
                            normalized_points.extend(
                                [max(0.0, min(1.0, norm_x)), max(0.0, min(1.0, norm_y))]
                            )

                        if normalized_points:
                            segments.append([class_id, normalized_points])
                if segments:
                    annotations["segments"] = segments
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            # Записываем в NDJSON, только если есть аннотации (или если не удаляем пустые)
            if annotations or not delete_void:
                image_record = {
                    "type": "image",
                    "file": relative_img_path,
                    "width": image_width,
                    "height": image_height,
                    "split": split,
                    "annotations": annotations,
                }
                with open(ndjson_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(image_record) + "\n")

        return True, None
    except Exception as e:
        return False, f"Произошла критическая ошибка при форматировании датасета: {e}"
