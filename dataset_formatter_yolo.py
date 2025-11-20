# dataset_formatter_yolo.py

import os
import random
from qgis.core import QgsVectorLayer, QgsFeatureRequest, QgsWkbTypes


def save_yolo_native_dataset(
    intersected_layer: QgsVectorLayer,
    grid_layer: QgsVectorLayer,
    class_field: str,
    output_dir: str,
    image_format: str,
    splits: dict,
    metadata: dict,
    delete_void: bool,
    progress_reporter=None,
):
    """
    Создает набор данных в нативном формате YOLOv8 (папки, .txt файлы, dataset.yaml).
    Поддерживает задачи 'detect' (рамки) и 'segment' (полигоны).
    """
    try:
        task_type = metadata.get("task", "detect").lower()

        # --- 1. Подготовка структуры и путей ---
        print("Создание структуры каталогов YOLO...")
        images_base_dir = os.path.join(output_dir, "images")
        labels_base_dir = os.path.join(output_dir, "labels")

        for split_name in splits.keys():  # train, val, test
            os.makedirs(os.path.join(images_base_dir, split_name), exist_ok=True)
            os.makedirs(os.path.join(labels_base_dir, split_name), exist_ok=True)

        # --- 2. Сбор классов и создание dataset.yaml ---
        print("Сбор классов и создание dataset.yaml...")
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

        yaml_data = {
            "path": os.path.abspath(output_dir),
            "train": os.path.join("images", "train"),
            "val": os.path.join("images", "val"),
            "test": os.path.join("images", "test"),
            "names": {i: name for i, name in enumerate(class_names_list)},
        }

        yaml_path = os.path.join(output_dir, "dataset.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {yaml_data['path'].replace(os.sep, '/')}\n")
            f.write(f"train: {yaml_data['train'].replace(os.sep, '/')}\n")
            f.write(f"val: {yaml_data['val'].replace(os.sep, '/')}\n")
            f.write(f"test: {yaml_data['test'].replace(os.sep, '/')}\n\n")
            f.write("names:\n")
            for i, name in yaml_data["names"].items():
                f.write(f"  {i}: {name}\n")
        print(f"Файл 'dataset.yaml' создан: {yaml_path}")

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

        # В режиме обновления изображения уже экспортированы в temp_new_data/images
        # В обычном режиме они в output_dir/images
        if "temp_new_data" in output_dir:
            exported_images_dir = os.path.join(output_dir, "images")
        else:
            exported_images_dir = os.path.join(output_dir, "images")

        for i, tile_feature in enumerate(grid_layer.getFeatures()):
            if progress_reporter and progress_reporter.is_canceled():
                return False, "Операция отменена."
            if progress_reporter:
                progress_reporter.set_progress(i + 1, total_tiles)

            tile_id = tile_feature.id()
            tile_extent = tile_feature.geometry().boundingBox()
            img_name = f"tile_{tile_id}.{image_format.lower()}"
            label_name = f"tile_{tile_id}.txt"

            request = QgsFeatureRequest().setFilterExpression(
                f'"{overlay_id_field}" = {tile_id}'
            )
            features_in_tile = list(intersected_layer.getFeatures(request))

            source_image_path = os.path.join(exported_images_dir, img_name)

            if not features_in_tile and delete_void:
                if os.path.exists(source_image_path):
                    try:
                        os.remove(source_image_path)
                        print(f"Удален пустой тайл: {source_image_path}")
                    except OSError as e:
                        print(
                            f"Ошибка при удалении пустого тайла {source_image_path}: {e}"
                        )
                continue

            split = random.choices(split_names, weights=split_weights, k=1)[0]

            # --- 5. Формирование аннотаций ---
            annotations = []
            for feat in features_in_tile:
                class_id = class_map.get(feat[class_field])
                if class_id is None:
                    continue

                geom = feat.geometry()

                if task_type == "detect":
                    bbox = geom.boundingBox()
                    box_center_x = (
                        bbox.center().x() - tile_extent.xMinimum()
                    ) / tile_extent.width()
                    box_center_y = (
                        tile_extent.yMaximum() - bbox.center().y()
                    ) / tile_extent.height()
                    box_width = bbox.width() / tile_extent.width()
                    box_height = bbox.height() / tile_extent.height()

                    box_center_x = max(0.0, min(1.0, box_center_x))
                    box_center_y = max(0.0, min(1.0, box_center_y))
                    box_width = max(0.0, min(1.0, box_width))
                    box_height = max(0.0, min(1.0, box_height))

                    annotations.append(
                        f"{class_id} {box_center_x:.6f} {box_center_y:.6f} {box_width:.6f} {box_height:.6f}"
                    )

                elif task_type == "segment":
                    if geom.wkbType() not in [
                        QgsWkbTypes.Polygon,
                        QgsWkbTypes.MultiPolygon,
                    ]:
                        continue

                    polygons = (
                        geom.asMultiPolygon()
                        if geom.isMultipart()
                        else [geom.asPolygon()]
                    )

                    for poly in polygons:
                        exterior_ring = poly[0]
                        normalized_points = []
                        for point in exterior_ring:
                            norm_x = (
                                point.x() - tile_extent.xMinimum()
                            ) / tile_extent.width()
                            norm_y = (
                                tile_extent.yMaximum() - point.y()
                            ) / tile_extent.height()
                            norm_x = max(0.0, min(1.0, norm_x))
                            norm_y = max(0.0, min(1.0, norm_y))
                            normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

                        if normalized_points:
                            annotations.append(
                                f"{class_id} {' '.join(normalized_points)}"
                            )

            # --- 6. Сохранение файла разметки и перемещение изображения ---
            if annotations:
                label_path = os.path.join(labels_base_dir, split, label_name)
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(annotations))

            dest_image_path = os.path.join(images_base_dir, split, img_name)
            if os.path.exists(source_image_path):
                os.rename(source_image_path, dest_image_path)

        # --- 7. Очистка временной папки ---
        try:
            if not os.listdir(exported_images_dir):
                os.rmdir(exported_images_dir)
        except OSError:
            print(
                f"Предупреждение: не удалось удалить временную директорию {exported_images_dir}."
            )

        return True, None
    except Exception as e:
        return (
            False,
            f"Произошла критическая ошибка при форматировании нативного датасета YOLO: {e}",
        )
