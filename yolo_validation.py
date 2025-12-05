# -*- coding: utf-8 -*-
"""
Модуль для валидации моделей YOLO
Поддерживает детекцию и сегментацию
"""

import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class AdvancedValidator:
    """Расширенный валидатор моделей YOLO"""

    def __init__(self):
        """Инициализация валидатора"""
        pass

    def validate(
        self,
        model_path: str,
        dataset_path: str,
        task: str = "detect",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        comprehensive: bool = True,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Выполняет валидацию модели

        :param model_path: Путь к модели (.pt файл)
        :param dataset_path: Путь к датасету
        :param task: Тип задачи (detect/segment)
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU для NMS
        :param max_detections: Максимальное количество детекций
        :param comprehensive: Выполнять комплексную валидацию
        :param device: Устройство (cpu/0/1 и т.д.)
        :return: Словарь с результатами валидации
        """
        try:
            # Настраиваем логирование ultralytics перед импортом
            # Это перенаправляет логи ultralytics в QGIS MessageLog
            try:
                import logging
                from qgis.core import QgsMessageLog, Qgis
                
                # Создаем QGIS handler для ultralytics
                class QGISLogHandler(logging.Handler):
                    def emit(self, record):
                        try:
                            msg = self.format(record)
                            if record.levelno >= logging.ERROR:
                                QgsMessageLog.logMessage(msg, 'YOLO QGIS', Qgis.Critical)
                            elif record.levelno >= logging.WARNING:
                                QgsMessageLog.logMessage(msg, 'YOLO QGIS', Qgis.Warning)
                            else:
                                QgsMessageLog.logMessage(msg, 'YOLO QGIS', Qgis.Info)
                        except Exception:
                            pass
                
                # Настраиваем логгеры ultralytics
                for logger_name in ['ultralytics', 'ultralytics.engine', 
                                   'ultralytics.engine.trainer', 'ultralytics.engine.validator',
                                   'ultralytics.engine.model', 'ultralytics.utils']:
                    ultralytics_logger = logging.getLogger(logger_name)
                    # Удаляем проблемные handlers
                    for handler in list(ultralytics_logger.handlers):
                        if isinstance(handler, logging.StreamHandler):
                            try:
                                if handler.stream is None or not hasattr(handler.stream, 'write'):
                                    ultralytics_logger.removeHandler(handler)
                            except Exception:
                                pass
                    # Добавляем QGIS handler
                    qgis_handler = QGISLogHandler()
                    qgis_handler.setFormatter(logging.Formatter('%(message)s'))
                    ultralytics_logger.addHandler(qgis_handler)
                    ultralytics_logger.setLevel(logging.INFO)
                    ultralytics_logger.propagate = False
            except Exception:
                # Если не удалось настроить, продолжаем работу
                pass
            
            # Проверяем наличие ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                return {"error": "Библиотека ultralytics не установлена"}

            if not os.path.exists(model_path):
                return {"error": f"Модель не найдена: {model_path}"}

            if not os.path.exists(dataset_path):
                return {"error": f"Датасет не найден: {dataset_path}"}

            # Определяем путь к dataset.yaml
            yaml_path = os.path.join(dataset_path, "dataset.yaml")
            if not os.path.exists(yaml_path):
                return {"error": "Не найден dataset.yaml в датасете"}

            # Загружаем модель
            model = YOLO(model_path)

            # Выполняем валидацию
            results = model.val(
                data=yaml_path,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                device=device,
                plots=True,
            )

            # Извлекаем метрики
            metrics = {}
            if hasattr(results, "results_dict"):
                metrics = results.results_dict
            elif hasattr(results, "box"):
                # Для детекции
                metrics = {
                    "mAP50": getattr(results.box, "map50", 0.0),
                    "mAP50-95": getattr(results.box, "map", 0.0),
                    "precision": getattr(results.box, "mp", 0.0),
                    "recall": getattr(results.box, "mr", 0.0),
                    "f1_score": (
                        2
                        * (
                            getattr(results.box, "mp", 0.0)
                            * getattr(results.box, "mr", 0.0)
                        )
                        / (
                            getattr(results.box, "mp", 0.0)
                            + getattr(results.box, "mr", 0.0)
                        )
                        if (
                            getattr(results.box, "mp", 0.0)
                            + getattr(results.box, "mr", 0.0)
                        )
                        > 0
                        else 0.0
                    ),
                }
            elif hasattr(results, "seg"):
                # Для сегментации
                metrics = {
                    "mAP50": getattr(results.seg, "map50", 0.0),
                    "mAP50-95": getattr(results.seg, "map", 0.0),
                    "precision": getattr(results.seg, "mp", 0.0),
                    "recall": getattr(results.seg, "mr", 0.0),
                    "f1_score": (
                        2
                        * (
                            getattr(results.seg, "mp", 0.0)
                            * getattr(results.seg, "mr", 0.0)
                        )
                        / (
                            getattr(results.seg, "mp", 0.0)
                            + getattr(results.seg, "mr", 0.0)
                        )
                        if (
                            getattr(results.seg, "mp", 0.0)
                            + getattr(results.seg, "mr", 0.0)
                        )
                        > 0
                        else 0.0
                    ),
                }

            # Комплексная валидация
            if comprehensive:
                comprehensive_results = self._comprehensive_validation(
                    model, dataset_path, task, conf_threshold, iou_threshold, device
                )
                metrics.update(comprehensive_results)

            return {
                "success": True,
                "metrics": metrics,
                "timestamp": self._get_timestamp(),
                "model_path": model_path,
                "dataset_path": dataset_path,
                "task": task,
            }

        except Exception as e:
            logger.error(f"Ошибка валидации: {e}", exc_info=True)
            return {"error": str(e)}

    def _comprehensive_validation(
        self,
        model,
        dataset_path: str,
        task: str,
        conf_threshold: float,
        iou_threshold: float,
        device: str,
    ) -> Dict[str, Any]:
        """
        Выполняет комплексную валидацию

        :param model: Загруженная модель YOLO
        :param dataset_path: Путь к датасету
        :param task: Тип задачи
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU
        :param device: Устройство
        :return: Дополнительные метрики
        """
        try:
            comprehensive = {}

            # Анализ по классам
            class_metrics = self._get_class_metrics(model, dataset_path, task, device)
            if class_metrics:
                comprehensive["class_metrics"] = class_metrics

            # Анализ по размерам объектов
            size_metrics = self._get_size_metrics(model, dataset_path, task, device)
            if size_metrics:
                comprehensive["size_metrics"] = size_metrics

            # Анализ ошибок
            error_analysis = self._analyze_errors(model, dataset_path, task, device)
            if error_analysis:
                comprehensive["error_analysis"] = error_analysis

            return comprehensive

        except Exception as e:
            logger.error(f"Ошибка комплексной валидации: {e}", exc_info=True)
            return {}

    def _get_class_metrics(
        self, model, dataset_path: str, task: str, device: str
    ) -> Dict[str, Any]:
        """Получает метрики по классам"""
        try:
            # Загружаем информацию о классах из dataset.yaml
            import yaml

            yaml_path = os.path.join(dataset_path, "dataset.yaml")
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            class_names = yaml_data.get("names", {})

            # Выполняем валидацию с детальными метриками
            results = model.val(data=yaml_path, device=device, plots=False)

            class_metrics = {}
            if hasattr(results, "names"):
                for class_id, class_name in class_names.items():
                    class_metrics[class_name] = {
                        "precision": (
                            getattr(results, f"p{class_id}", 0.0)
                            if hasattr(results, f"p{class_id}")
                            else 0.0
                        ),
                        "recall": (
                            getattr(results, f"r{class_id}", 0.0)
                            if hasattr(results, f"r{class_id}")
                            else 0.0
                        ),
                        "mAP50": (
                            getattr(results, f"map50{class_id}", 0.0)
                            if hasattr(results, f"map50{class_id}")
                            else 0.0
                        ),
                    }

            return class_metrics

        except Exception as e:
            logger.error(f"Ошибка получения метрик по классам: {e}", exc_info=True)
            return {}

    def _get_size_metrics(
        self, model, dataset_path: str, task: str, device: str
    ) -> Dict[str, Any]:
        """Получает метрики по размерам объектов"""
        # Базовая реализация - можно расширить
        return {
            "small_objects": {"precision": 0.0, "recall": 0.0},
            "medium_objects": {"precision": 0.0, "recall": 0.0},
            "large_objects": {"precision": 0.0, "recall": 0.0},
        }

    def _analyze_errors(
        self, model, dataset_path: str, task: str, device: str
    ) -> Dict[str, Any]:
        """Анализирует типы ошибок"""
        # Базовая реализация - можно расширить
        return {"false_positives": 0, "false_negatives": 0, "localization_errors": 0}

    @staticmethod
    def _get_timestamp() -> str:
        """Возвращает текущую временную метку"""
        from datetime import datetime

        return datetime.now().isoformat()


class ModelComparator:
    """Класс для сравнения моделей"""

    def __init__(self):
        """Инициализация компаратора"""
        pass

    def compare(
        self,
        models: List[Dict[str, str]],
        dataset_path: str,
        task: str = "detect",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Сравнивает несколько моделей

        :param models: Список словарей с ключами 'path' и 'name'
        :param dataset_path: Путь к датасету
        :param task: Тип задачи
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU
        :param device: Устройство
        :return: Словарь с результатами сравнения
        """
        try:
            validator = AdvancedValidator()
            results = {}

            for model_info in models:
                model_path = model_info.get("path")
                model_name = model_info.get("name", os.path.basename(model_path))

                if not os.path.exists(model_path):
                    results[model_name] = {"error": "Модель не найдена"}
                    continue

                validation_result = validator.validate(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    task=task,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    device=device,
                    comprehensive=False,
                )

                results[model_name] = validation_result

            # Находим лучшие модели по каждой метрике
            comparison_metrics = self._find_best_models(results)

            return {
                "success": True,
                "results": results,
                "comparison_metrics": comparison_metrics,
                "timestamp": AdvancedValidator._get_timestamp(),
            }

        except Exception as e:
            logger.error(f"Ошибка сравнения моделей: {e}", exc_info=True)
            return {"error": str(e)}

    def _find_best_models(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Находит лучшие модели по каждой метрике

        :param results: Результаты валидации моделей
        :return: Словарь с лучшими моделями по метрикам
        """
        comparison = {}

        # Собираем все метрики
        metrics_to_compare = ["mAP50", "mAP50-95", "precision", "recall", "f1_score"]

        for metric_name in metrics_to_compare:
            best_value = -1
            best_model = None

            for model_name, result in results.items():
                if "error" in result:
                    continue

                metrics = result.get("metrics", {})
                value = metrics.get(metric_name, 0.0)

                if value > best_value:
                    best_value = value
                    best_model = model_name

            if best_model:
                comparison[metric_name] = {"model": best_model, "value": best_value}

        return comparison
