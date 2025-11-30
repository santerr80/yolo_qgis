# -*- coding: utf-8 -*-
"""
Модуль для детекции объектов на основе обученных YOLO моделей
Поддерживает детекцию на растрах QGIS и изображениях
"""

import os
import logging
import tempfile
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from qgis.core import (
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsRectangle,
    QgsField,
    QgsProject,
    QgsRasterLayer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMapSettings,
    QgsMapRendererParallelJob,
)
from qgis.PyQt.QtCore import QVariant, QSize
from qgis.PyQt.QtGui import QColor

logger = logging.getLogger(__name__)


class YOLODetector:
    """Класс для детекции объектов с использованием обученных YOLO моделей"""

    def __init__(self):
        """Инициализация детектора"""
        self.model = None
        self.model_path = None
        self.class_names = {}
        self.task_type = "detect"  # detect или segment

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Загружает обученную YOLO модель

        :param model_path: Путь к файлу модели (.pt)
        :return: Словарь с результатом загрузки
        """
        try:
            if not os.path.exists(model_path):
                return {"error": f"Файл модели не найден: {model_path}"}

            # Проверяем наличие ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                return {
                    "error": "Библиотека ultralytics не установлена. Установите: pip install ultralytics"
                }

            # Загружаем модель
            self.model = YOLO(model_path)
            self.model_path = model_path

            # Получаем имена классов из модели
            if hasattr(self.model, "names"):
                self.class_names = self.model.names
            elif hasattr(self.model, "model") and hasattr(self.model.model, "names"):
                self.class_names = self.model.model.names
            else:
                # Пробуем получить из dataset.yaml, если модель была обучена
                self.class_names = self._extract_class_names_from_model()

            # Определяем тип задачи
            if hasattr(self.model, "task"):
                self.task_type = self.model.task
            elif "seg" in model_path.lower():
                self.task_type = "segment"
            else:
                self.task_type = "detect"

            logger.info(
                f"Модель загружена: {model_path}, тип задачи: {self.task_type}, "
                f"классов: {len(self.class_names)}"
            )

            return {
                "success": True,
                "model_path": model_path,
                "task_type": self.task_type,
                "num_classes": len(self.class_names),
                "class_names": self.class_names,
            }

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}", exc_info=True)
            return {"error": str(e)}

    def _extract_class_names_from_model(self) -> Dict[int, str]:
        """Извлекает имена классов из модели"""
        try:
            # Пробуем получить из атрибутов модели
            if hasattr(self.model, "overrides"):
                if "names" in self.model.overrides:
                    names = self.model.overrides["names"]
                    if isinstance(names, dict):
                        return names
                    elif isinstance(names, list):
                        return {i: name for i, name in enumerate(names)}

            # Если не нашли, возвращаем пустой словарь
            return {}
        except Exception as e:
            logger.warning(f"Не удалось извлечь имена классов: {e}")
            return {}

    def detect_on_raster(
        self,
        raster_layer: QgsRasterLayer,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: Optional[int] = None,
        device: str = "cpu",
        progress_callback=None,
        extent: Optional[QgsRectangle] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет детекцию объектов на растровом слое QGIS

        :param raster_layer: Растровый слой QGIS
        :param conf_threshold: Порог уверенности (0.0-1.0)
        :param iou_threshold: Порог IoU для NMS (0.0-1.0)
        :param image_size: Размер изображения для обработки (None = автоматически)
        :param device: Устройство (cpu/0/1 и т.д.)
        :param progress_callback: Callback для прогресса
        :return: Словарь с результатами детекции
        """
        try:
            if self.model is None:
                return {"error": "Модель не загружена. Сначала загрузите модель."}

            if not raster_layer.isValid():
                return {"error": "Растровый слой недействителен"}

            # Экспортируем растр в изображение
            if progress_callback:
                progress_callback("Экспорт растра в изображение...")

            # Используем переданный экстент или экстент всего растра
            detection_extent = extent if extent else raster_layer.extent()
            image_path, exported_extent, crs = self._raster_to_image_file(
                raster_layer, detection_extent
            )

            if image_path is None:
                return {"error": "Не удалось экспортировать растр в изображение"}

            try:
                # Выполняем детекцию
                if progress_callback:
                    progress_callback("Выполнение детекции...")

                results = self.model.predict(
                    image_path,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=image_size,
                    device=device,
                    verbose=False,
                )

                if not results or len(results) == 0:
                    return {"error": "Детекция не вернула результатов"}

                # Обрабатываем результаты
                if progress_callback:
                    progress_callback("Обработка результатов...")

                # Вычисляем размеры изображения для преобразования координат
                # Используем exported_extent, который уже в CRS растрового слоя
                if extent:
                    # Если используется ограниченный экстент, вычисляем размеры на основе соотношения
                    full_extent = raster_layer.extent()
                    # Используем exported_extent, который уже преобразован в CRS растрового слоя
                    width_ratio = (exported_extent.width() / full_extent.width()) * raster_layer.width()
                    height_ratio = (exported_extent.height() / full_extent.height()) * raster_layer.height()
                    img_width = int(width_ratio)
                    img_height = int(height_ratio)
                else:
                    img_width = raster_layer.width()
                    img_height = raster_layer.height()

                detections = self._process_detection_results(
                    results[0], exported_extent, img_width, img_height
                )

                return {
                    "success": True,
                    "detections": detections,
                    "num_detections": len(detections),
                    "extent": exported_extent,
                    "crs": crs,
                }
            finally:
                # Удаляем временный файл
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл: {e}")

        except Exception as e:
            logger.error(f"Ошибка детекции на растре: {e}", exc_info=True)
            return {"error": str(e)}

    def detect_on_image(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: Optional[int] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Выполняет детекцию объектов на изображении

        :param image_path: Путь к изображению
        :param conf_threshold: Порог уверенности (0.0-1.0)
        :param iou_threshold: Порог IoU для NMS (0.0-1.0)
        :param image_size: Размер изображения для обработки (None = автоматически)
        :param device: Устройство (cpu/0/1 и т.д.)
        :return: Словарь с результатами детекции
        """
        try:
            if self.model is None:
                return {"error": "Модель не загружена. Сначала загрузите модель."}

            if not os.path.exists(image_path):
                return {"error": f"Файл изображения не найден: {image_path}"}

            # Выполняем детекцию
            results = self.model.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=image_size,
                device=device,
                verbose=False,
            )

            if not results or len(results) == 0:
                return {"error": "Детекция не вернула результатов"}

            # Обрабатываем результаты
            detections = self._process_detection_results(results[0], None, None, None)

            return {
                "success": True,
                "detections": detections,
                "num_detections": len(detections),
            }

        except Exception as e:
            logger.error(f"Ошибка детекции на изображении: {e}", exc_info=True)
            return {"error": str(e)}

    def _raster_to_image_file(
        self, raster_layer: QgsRasterLayer, extent: Optional[QgsRectangle] = None
    ) -> Tuple[Optional[str], Optional[QgsRectangle], Optional[QgsCoordinateReferenceSystem]]:
        """
        Экспортирует растровый слой QGIS во временный файл изображения

        :param raster_layer: Растровый слой
        :param extent: Экстент для экспорта (если None, используется весь растр)
        :return: Кортеж (image_path, extent, crs) или (None, None, None) при ошибке
        """
        try:
            # Получаем экстент и CRS
            # Примечание: extent передается в CRS проекта, преобразуем в CRS растрового слоя
            if extent is None:
                extent = raster_layer.extent()
            else:
                # Преобразуем extent из CRS проекта в CRS растрового слоя
                project = QgsProject.instance()
                project_crs = project.crs()
                raster_crs = raster_layer.crs()
                
                if project_crs.isValid() and raster_crs.isValid() and project_crs != raster_crs:
                    try:
                        transform = QgsCoordinateTransform(
                            project_crs, raster_crs, QgsProject.instance()
                        )
                        extent = transform.transformBoundingBox(extent)
                        logger.debug(
                            f"Экстент преобразован из CRS проекта {project_crs.authid()} "
                            f"в CRS растрового слоя {raster_crs.authid()}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Ошибка преобразования CRS экстента из проекта в CRS растра: {e}. "
                            f"Используется экстент без преобразования."
                        )
            
            crs = raster_layer.crs()
            
            # Проверяем, что extent валиден (не пустой)
            if extent is None or extent.isEmpty():
                logger.error("Переданный экстент недействителен или пуст")
                return None, None, None

            # Вычисляем размеры изображения на основе экстента
            full_extent = raster_layer.extent()
            if extent != full_extent:
                # Если используется ограниченный экстент, вычисляем размеры пропорционально
                width_ratio = extent.width() / full_extent.width()
                height_ratio = extent.height() / full_extent.height()
                img_width = int(raster_layer.width() * width_ratio)
                img_height = int(raster_layer.height() * height_ratio)
                # Ограничиваем минимальный размер
                img_width = max(100, img_width)
                img_height = max(100, img_height)
            else:
                img_width = raster_layer.width()
                img_height = raster_layer.height()

            # Создаем временный файл
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb"
            )
            temp_file.close()
            image_path = temp_file.name

            # Используем QgsMapSettings для экспорта растра в изображение
            settings = QgsMapSettings()
            settings.setLayers([raster_layer])
            settings.setBackgroundColor(QColor(255, 255, 255))
            settings.setOutputSize(QSize(img_width, img_height))
            settings.setOutputDpi(96)  # Стандартное DPI
            settings.setDestinationCrs(crs)
            settings.setExtent(extent)

            # Рендерим изображение
            job = QgsMapRendererParallelJob(settings)
            job.start()
            job.waitForFinished()

            errors = job.errors()
            if errors:
                logger.error(f"Ошибка рендеринга растра: {', '.join(errors)}")
                os.remove(image_path)
                return None, None, None

            # Сохраняем изображение
            img = job.renderedImage()
            img.save(image_path, "PNG")

            return image_path, extent, crs

        except Exception as e:
            logger.error(f"Ошибка экспорта растра в изображение: {e}", exc_info=True)
            return None, None, None

    def _process_detection_results(
        self,
        result,
        extent: Optional[QgsRectangle],
        raster_width: Optional[int],
        raster_height: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Обрабатывает результаты детекции YOLO и преобразует в список словарей

        :param result: Результат детекции от YOLO
        :param extent: Экстент растра (для преобразования координат)
        :param raster_width: Ширина растра в пикселях
        :param raster_height: Высота растра в пикселях
        :return: Список детекций
        """
        detections = []

        try:
            # Получаем boxes из результата
            boxes = result.boxes if hasattr(result, "boxes") else None
            if boxes is None:
                return detections

            # Получаем данные из boxes
            if hasattr(boxes, "xyxy"):
                # Формат: [x1, y1, x2, y2] в пикселях
                boxes_xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            elif hasattr(boxes, "data"):
                boxes_xyxy = boxes.data[:, :4].cpu().numpy() if hasattr(boxes.data, "cpu") else boxes.data[:, :4]
            else:
                return detections

            # Получаем confidence scores
            if hasattr(boxes, "conf"):
                confidences = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            elif hasattr(boxes, "data"):
                confidences = boxes.data[:, 4].cpu().numpy() if hasattr(boxes.data, "cpu") else boxes.data[:, 4]
            else:
                confidences = np.ones(len(boxes_xyxy))

            # Получаем классы
            if hasattr(boxes, "cls"):
                classes = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else boxes.cls.astype(int)
            elif hasattr(boxes, "data"):
                classes = boxes.data[:, 5].cpu().numpy().astype(int) if hasattr(boxes.data, "cpu") else boxes.data[:, 5].astype(int)
            else:
                classes = np.zeros(len(boxes_xyxy), dtype=int)

            # Обрабатываем каждую детекцию
            for i in range(len(boxes_xyxy)):
                box = boxes_xyxy[i]
                conf = float(confidences[i])
                cls_id = int(classes[i])

                # Получаем имя класса
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                # Преобразуем координаты
                if extent and raster_width and raster_height:
                    # Преобразуем из пикселей в координаты растра
                    x_min_px, y_min_px, x_max_px, y_max_px = box

                    # Нормализуем координаты (0-1)
                    x_min_norm = x_min_px / raster_width
                    y_min_norm = y_min_px / raster_height
                    x_max_norm = x_max_px / raster_width
                    y_max_norm = y_max_px / raster_height

                    # Преобразуем в координаты растра
                    x_min = extent.xMinimum() + x_min_norm * extent.width()
                    x_max = extent.xMinimum() + x_max_norm * extent.width()
                    y_min = extent.yMaximum() - y_max_norm * extent.height()
                    y_max = extent.yMaximum() - y_min_norm * extent.height()
                else:
                    # Используем пиксельные координаты
                    x_min, y_min, x_max, y_max = box

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": {
                            "x_min": float(x_min),
                            "y_min": float(y_min),
                            "x_max": float(x_max),
                            "y_max": float(y_max),
                        },
                    }
                )

        except Exception as e:
            logger.error(f"Ошибка обработки результатов детекции: {e}", exc_info=True)

        return detections

    def create_vector_layer(
        self,
        detections: List[Dict[str, Any]],
        layer_name: str,
        crs: Optional[QgsCoordinateReferenceSystem] = None,
    ) -> QgsVectorLayer:
        """
        Создает векторный слой QGIS из результатов детекции

        :param detections: Список детекций
        :param layer_name: Имя слоя
        :param crs: Система координат
        :return: Векторный слой
        """
        try:
            # Создаем слой в памяти
            layer = QgsVectorLayer("Polygon?crs=", layer_name, "memory")
            if not layer.isValid():
                raise Exception("Не удалось создать векторный слой")

            # Устанавливаем CRS
            if crs and crs.isValid():
                layer.setCrs(crs)

            # Добавляем поля
            provider = layer.dataProvider()
            fields = [
                QgsField("class_id", QVariant.Int),
                QgsField("class_name", QVariant.String),
                QgsField("confidence", QVariant.Double),
            ]
            provider.addAttributes(fields)
            layer.updateFields()

            # Добавляем объекты
            features = []
            for det in detections:
                bbox = det["bbox"]
                # Создаем прямоугольник из bounding box
                rect = QgsRectangle(
                    bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
                )
                geom = QgsGeometry.fromRect(rect)

                feature = QgsFeature()
                feature.setGeometry(geom)
                feature.setAttributes(
                    [det["class_id"], det["class_name"], det["confidence"]]
                )
                features.append(feature)

            provider.addFeatures(features)
            layer.updateExtents()

            return layer

        except Exception as e:
            logger.error(f"Ошибка создания векторного слоя: {e}", exc_info=True)
            raise
