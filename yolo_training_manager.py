# -*- coding: utf-8 -*-
"""
Главный менеджер системы обучения YOLO моделей
Объединяет все компоненты и предоставляет единый интерфейс
"""

import os
import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

try:
    from qgis.PyQt.QtCore import QObject, pyqtSignal
except ImportError:
    # Заглушка для случаев, когда QGIS не доступен
    class QObject:
        pass
    
    class _SignalStub:
        """Заглушка для pyqtSignal"""
        def __init__(self, *args):
            self._args = args
        
        def connect(self, *args):
            pass
        
        def emit(self, *args):
            pass
    
    def pyqtSignal(*args):
        """Фабрика для создания заглушек сигналов"""
        return _SignalStub(*args)

from .yolo_detection_trainer import DetectionTrainer, DetectionDatasetAnalyzer
from .yolo_segmentation_trainer import SegmentationTrainer, SegmentationDatasetAnalyzer
from .yolo_validation import AdvancedValidator, ModelComparator
from .yolo_metrics_tracker import MetricsTracker, MetricsVisualizer, MetricsDatabase

logger = logging.getLogger(__name__)


class YOLOTrainingManager(QObject):
    """Главный менеджер системы обучения"""
    
    # Сигналы для UI
    training_started = pyqtSignal(str)  # experiment_id
    training_progress = pyqtSignal(int, dict)  # epoch, metrics
    training_completed = pyqtSignal(str, bool, str)  # experiment_id, success, message
    validation_completed = pyqtSignal(str, dict)  # experiment_id, results
    status_message = pyqtSignal(str)  # message
    
    def __init__(self, log_dir: str = "logs", db_path: str = "yolo_metrics.db"):
        """Инициализация менеджера
        
        Args:
            log_dir: Директория для логов
            db_path: Путь к базе данных метрик
        """
        super().__init__()
        
        self.log_dir = log_dir
        self.db_path = db_path
        self.database = MetricsDatabase(db_path)
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Словарь активных экспериментов
        self.active_experiments = {}
        self.experiment_threads = {}
    
    def start_detection_training(self, dataset_path: str, model_type: str = "yolov8n",
                                epochs: int = 100, batch_size: int = 16,
                                image_size: int = 640, learning_rate: float = 0.01,
                                device: str = "cpu", pretrained: bool = True,
                                save_dir: Optional[str] = None,
                                project_name: str = "yolo_training",
                                **augmentation_params) -> Optional[str]:
        """Запускает обучение модели детекции
        
        Args:
            dataset_path: Путь к датасету
            model_type: Тип модели (yolov8n, yolov8s, etc.)
            epochs: Количество эпох
            batch_size: Размер батча
            image_size: Размер изображения
            learning_rate: Скорость обучения
            device: Устройство ('cpu' или '0' для GPU)
            pretrained: Использовать предобученные веса
            save_dir: Директория для сохранения
            project_name: Имя проекта
            **augmentation_params: Параметры аугментации
        
        Returns:
            ID эксперимента или None при ошибке
        """
        try:
            # Создаем ID эксперимента
            experiment_id = str(uuid.uuid4())
            
            # Сохраняем конфигурацию
            config = {
                'task': 'detect',
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'learning_rate': learning_rate,
                'device': device,
                'pretrained': pretrained,
                'augmentation': augmentation_params
            }
            
            # Добавляем в базу данных
            self.database.add_experiment(
                experiment_id=experiment_id,
                name=project_name,
                task='detect',
                model_type=model_type,
                config=config
            )
            
            # Создаем трекер метрик
            metrics_tracker = MetricsTracker(
                experiment_id=experiment_id,
                db_path=self.db_path,
                log_dir=self.log_dir
            )
            
            # Запускаем обучение в отдельном потоке
            thread = threading.Thread(
                target=self._run_detection_training,
                args=(experiment_id, dataset_path, model_type, epochs, batch_size,
                     image_size, learning_rate, device, pretrained, save_dir,
                     project_name, metrics_tracker, augmentation_params),
                daemon=True
            )
            
            self.active_experiments[experiment_id] = {
                'type': 'detection',
                'status': 'running',
                'config': config
            }
            self.experiment_threads[experiment_id] = thread
            
            thread.start()
            
            # Отправляем сигнал о начале обучения
            self.training_started.emit(experiment_id)
            self.status_message.emit(f"Запущено обучение детекции: {project_name}")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Ошибка запуска обучения детекции: {e}", exc_info=True)
            return None
    
    def _run_detection_training(self, experiment_id: str, dataset_path: str,
                               model_type: str, epochs: int, batch_size: int,
                               image_size: int, learning_rate: float, device: str,
                               pretrained: bool, save_dir: Optional[str],
                               project_name: str, metrics_tracker: MetricsTracker,
                               augmentation_params: Dict):
        """Запускает обучение детекции в отдельном потоке"""
        def log_callback(msg: str):
            """Callback для логирования сообщений"""
            self.status_message.emit(msg)
            logger.info(f"[{experiment_id}] {msg}")
        
        def epoch_callback(epoch: int, metrics: Dict[str, float]):
            """Callback для отслеживания прогресса эпох"""
            try:
                # Отправляем сигнал о прогрессе
                self.training_progress.emit(epoch, metrics)
                
                # Сохраняем метрики в трекер
                if metrics_tracker:
                    metrics_tracker.log_metrics(epoch, 'validation', metrics)
                
                # Логируем информацию об эпохе
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                log_callback(f"Эпоха {epoch}/{epochs}: {metrics_str}")
            except Exception as e:
                logger.error(f"Ошибка в callback эпохи: {e}", exc_info=True)
        
        try:
            # Создаем тренер
            log_callback("Инициализация тренера...")
            trainer = DetectionTrainer(model_type=model_type, device=device)
            log_callback(f"Загрузка модели {model_type}...")
            trainer.load_model(pretrained=pretrained)
            log_callback("Модель загружена успешно")
            
            # Запускаем обучение
            result = trainer.train(
                dataset_path=dataset_path,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                learning_rate=learning_rate,
                save_dir=save_dir,
                project_name=project_name,
                callback=log_callback,
                epoch_callback=epoch_callback,
                **augmentation_params
            )
            
            if result.get('success'):
                # Обновляем статус
                self.database.update_experiment_status(
                    experiment_id=experiment_id,
                    status='completed',
                    completed_at=datetime.now().isoformat()
                )
                self.active_experiments[experiment_id]['status'] = 'completed'
                
                # Отправляем сигнал о завершении
                self.training_completed.emit(
                    experiment_id,
                    True,
                    f"Обучение завершено. Модель: {result.get('model_path', 'N/A')}"
                )
            else:
                # Обновляем статус с ошибкой
                error_msg = result.get('error', 'Неизвестная ошибка')
                self.database.update_experiment_status(
                    experiment_id=experiment_id,
                    status='failed'
                )
                self.active_experiments[experiment_id]['status'] = 'failed'
                
                # Отправляем сигнал об ошибке
                self.training_completed.emit(experiment_id, False, error_msg)
                
        except Exception as e:
            logger.error(f"Ошибка в потоке обучения детекции: {e}", exc_info=True)
            self.database.update_experiment_status(
                experiment_id=experiment_id,
                status='failed'
            )
            self.active_experiments[experiment_id]['status'] = 'failed'
            self.training_completed.emit(experiment_id, False, str(e))
    
    def start_segmentation_training(self, dataset_path: str, model_type: str = "yolov8n-seg",
                                   epochs: int = 100, batch_size: int = 16,
                                   image_size: int = 640, learning_rate: float = 0.01,
                                   device: str = "cpu", pretrained: bool = True,
                                   save_dir: Optional[str] = None,
                                   project_name: str = "yolo_training",
                                   **augmentation_params) -> Optional[str]:
        """Запускает обучение модели сегментации
        
        Args:
            dataset_path: Путь к датасету
            model_type: Тип модели (yolov8n-seg, yolov8s-seg, etc.)
            epochs: Количество эпох
            batch_size: Размер батча
            image_size: Размер изображения
            learning_rate: Скорость обучения
            device: Устройство ('cpu' или '0' для GPU)
            pretrained: Использовать предобученные веса
            save_dir: Директория для сохранения
            project_name: Имя проекта
            **augmentation_params: Параметры аугментации
        
        Returns:
            ID эксперимента или None при ошибке
        """
        try:
            # Создаем ID эксперимента
            experiment_id = str(uuid.uuid4())
            
            # Сохраняем конфигурацию
            config = {
                'task': 'segment',
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'learning_rate': learning_rate,
                'device': device,
                'pretrained': pretrained,
                'augmentation': augmentation_params
            }
            
            # Добавляем в базу данных
            self.database.add_experiment(
                experiment_id=experiment_id,
                name=project_name,
                task='segment',
                model_type=model_type,
                config=config
            )
            
            # Создаем трекер метрик
            metrics_tracker = MetricsTracker(
                experiment_id=experiment_id,
                db_path=self.db_path,
                log_dir=self.log_dir
            )
            
            # Запускаем обучение в отдельном потоке
            thread = threading.Thread(
                target=self._run_segmentation_training,
                args=(experiment_id, dataset_path, model_type, epochs, batch_size,
                     image_size, learning_rate, device, pretrained, save_dir,
                     project_name, metrics_tracker, augmentation_params),
                daemon=True
            )
            
            self.active_experiments[experiment_id] = {
                'type': 'segmentation',
                'status': 'running',
                'config': config
            }
            self.experiment_threads[experiment_id] = thread
            
            thread.start()
            
            # Отправляем сигнал о начале обучения
            self.training_started.emit(experiment_id)
            self.status_message.emit(f"Запущено обучение сегментации: {project_name}")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Ошибка запуска обучения сегментации: {e}", exc_info=True)
            return None
    
    def _run_segmentation_training(self, experiment_id: str, dataset_path: str,
                                  model_type: str, epochs: int, batch_size: int,
                                  image_size: int, learning_rate: float, device: str,
                                  pretrained: bool, save_dir: Optional[str],
                                  project_name: str, metrics_tracker: MetricsTracker,
                                  augmentation_params: Dict):
        """Запускает обучение сегментации в отдельном потоке"""
        def log_callback(msg: str):
            """Callback для логирования сообщений"""
            self.status_message.emit(msg)
            logger.info(f"[{experiment_id}] {msg}")
        
        def epoch_callback(epoch: int, metrics: Dict[str, float]):
            """Callback для отслеживания прогресса эпох"""
            try:
                # Отправляем сигнал о прогрессе
                self.training_progress.emit(epoch, metrics)
                
                # Сохраняем метрики в трекер
                if metrics_tracker:
                    metrics_tracker.log_metrics(epoch, 'validation', metrics)
                
                # Логируем информацию об эпохе
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                log_callback(f"Эпоха {epoch}/{epochs}: {metrics_str}")
            except Exception as e:
                logger.error(f"Ошибка в callback эпохи: {e}", exc_info=True)
        
        try:
            # Создаем тренер
            log_callback("Инициализация тренера...")
            trainer = SegmentationTrainer(model_type=model_type, device=device)
            log_callback(f"Загрузка модели {model_type}...")
            trainer.load_model(pretrained=pretrained)
            log_callback("Модель загружена успешно")
            
            # Запускаем обучение
            result = trainer.train(
                dataset_path=dataset_path,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                learning_rate=learning_rate,
                save_dir=save_dir,
                project_name=project_name,
                callback=log_callback,
                epoch_callback=epoch_callback,
                **augmentation_params
            )
            
            if result.get('success'):
                # Обновляем статус
                self.database.update_experiment_status(
                    experiment_id=experiment_id,
                    status='completed',
                    completed_at=datetime.now().isoformat()
                )
                self.active_experiments[experiment_id]['status'] = 'completed'
                
                # Отправляем сигнал о завершении
                self.training_completed.emit(
                    experiment_id,
                    True,
                    f"Обучение завершено. Модель: {result.get('model_path', 'N/A')}"
                )
            else:
                # Обновляем статус с ошибкой
                error_msg = result.get('error', 'Неизвестная ошибка')
                self.database.update_experiment_status(
                    experiment_id=experiment_id,
                    status='failed'
                )
                self.active_experiments[experiment_id]['status'] = 'failed'
                
                # Отправляем сигнал об ошибке
                self.training_completed.emit(experiment_id, False, error_msg)
                
        except Exception as e:
            logger.error(f"Ошибка в потоке обучения сегментации: {e}", exc_info=True)
            self.database.update_experiment_status(
                experiment_id=experiment_id,
                status='failed'
            )
            self.active_experiments[experiment_id]['status'] = 'failed'
            self.training_completed.emit(experiment_id, False, str(e))
    
    def validate_model(self, model_path: str, dataset_path: str, task: str = "detect",
                      experiment_id: Optional[str] = None, comprehensive: bool = False,
                      conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict[str, Any]:
        """Валидирует модель
        
        Args:
            model_path: Путь к модели
            dataset_path: Путь к датасету
            task: Тип задачи ('detect' или 'segment')
            experiment_id: ID эксперимента (опционально)
            comprehensive: Выполнить комплексную валидацию
            conf_threshold: Порог уверенности
            iou_threshold: Порог IoU
        
        Returns:
            Результаты валидации
        """
        try:
            validator = AdvancedValidator()
            results = validator.validate(
                model_path=model_path,
                dataset_path=dataset_path,
                task=task,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                comprehensive=comprehensive
            )
            
            # Отправляем сигнал о завершении валидации
            if experiment_id:
                self.validation_completed.emit(experiment_id, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка валидации: {e}", exc_info=True)
            return {'error': str(e)}
    
    def compare_models(self, models: List[Dict[str, str]], dataset_path: str,
                      task: str = "detect") -> Dict[str, Any]:
        """Сравнивает несколько моделей
        
        Args:
            models: Список моделей с ключами 'name' и 'path'
            dataset_path: Путь к датасету
            task: Тип задачи
        
        Returns:
            Результаты сравнения
        """
        try:
            comparator = ModelComparator()
            return comparator.compare(
                models=models,
                dataset_path=dataset_path,
                task=task
            )
        except Exception as e:
            logger.error(f"Ошибка сравнения моделей: {e}", exc_info=True)
            return {'error': str(e)}
    
    def analyze_dataset(self, dataset_path: str, task: str = "detect") -> Dict[str, Any]:
        """Анализирует датасет
        
        Args:
            dataset_path: Путь к датасету
            task: Тип задачи ('detect' или 'segment')
        
        Returns:
            Результаты анализа
        """
        try:
            if task == "segment":
                analyzer = SegmentationDatasetAnalyzer()
            else:
                analyzer = DetectionDatasetAnalyzer()
            
            return analyzer.analyze(dataset_path)
            
        except Exception as e:
            logger.error(f"Ошибка анализа датасета: {e}", exc_info=True)
            return {'error': str(e)}
    
    def cancel_training(self, experiment_id: str) -> bool:
        """Отменяет обучение
        
        Args:
            experiment_id: ID эксперимента
        
        Returns:
            True если отмена успешна
        """
        try:
            if experiment_id in self.active_experiments:
                # Обновляем статус
                self.database.update_experiment_status(
                    experiment_id=experiment_id,
                    status='cancelled'
                )
                self.active_experiments[experiment_id]['status'] = 'cancelled'
                
                # Примечание: ultralytics не поддерживает прямую отмену обучения
                # Можно только отметить статус как отмененный
                self.status_message.emit(f"Обучение {experiment_id} отмечено как отмененное")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка отмены обучения: {e}", exc_info=True)
            return False
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Получает сводку по эксперименту
        
        Args:
            experiment_id: ID эксперимента
        
        Returns:
            Сводка эксперимента
        """
        try:
            experiment = self.database.get_experiment(experiment_id)
            if not experiment:
                return {}
            
            # Получаем метрики
            metrics = self.database.get_experiment_metrics(experiment_id)
            
            # Формируем сводку
            summary = {
                'id': experiment['id'],
                'name': experiment['name'],
                'task': experiment['task'],
                'model_type': experiment['model_type'],
                'status': experiment['status'],
                'created_at': experiment['created_at'],
                'completed_at': experiment.get('completed_at'),
                'config': experiment.get('config', {})
            }
            
            # Извлекаем финальные метрики
            if metrics:
                # Группируем по эпохам
                epochs_metrics = {}
                for metric in metrics:
                    epoch = metric['epoch']
                    if epoch not in epochs_metrics:
                        epochs_metrics[epoch] = {}
                    epochs_metrics[epoch][metric['metric_name']] = metric['metric_value']
                
                # Берем последнюю эпоху
                if epochs_metrics:
                    last_epoch = max(epochs_metrics.keys())
                    summary['final_metrics'] = epochs_metrics[last_epoch]
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки: {e}", exc_info=True)
            return {}
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Получает все эксперименты
        
        Returns:
            Список экспериментов
        """
        try:
            return self.database.get_all_experiments()
        except Exception as e:
            logger.error(f"Ошибка получения экспериментов: {e}", exc_info=True)
            return []


class TrainingConfigManager:
    """Менеджер конфигураций обучения"""
    
    def __init__(self, config_dir: str = "training_configs"):
        """Инициализация менеджера конфигураций
        
        Args:
            config_dir: Директория для хранения конфигураций
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, config: Dict[str, Any], name: str) -> bool:
        """Сохраняет конфигурацию
        
        Args:
            config: Словарь с конфигурацией
            name: Имя конфигурации
        
        Returns:
            True если сохранение успешно
        """
        try:
            config_path = os.path.join(self.config_dir, f"{name}.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}", exc_info=True)
            return False
    
    def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Загружает конфигурацию
        
        Args:
            name: Имя конфигурации
        
        Returns:
            Конфигурация или None
        """
        try:
            config_path = os.path.join(self.config_dir, f"{name}.json")
            if not os.path.exists(config_path):
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}", exc_info=True)
            return None
    
    def list_configs(self) -> List[str]:
        """Получает список доступных конфигураций
        
        Returns:
            Список имен конфигураций
        """
        try:
            configs = []
            for filename in os.listdir(self.config_dir):
                if filename.endswith('.json'):
                    configs.append(filename[:-5])  # Убираем .json
            return sorted(configs)
        except Exception as e:
            logger.error(f"Ошибка получения списка конфигураций: {e}", exc_info=True)
            return []
    
    def delete_config(self, name: str) -> bool:
        """Удаляет конфигурацию
        
        Args:
            name: Имя конфигурации
        
        Returns:
            True если удаление успешно
        """
        try:
            config_path = os.path.join(self.config_dir, f"{name}.json")
            if os.path.exists(config_path):
                os.remove(config_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Ошибка удаления конфигурации: {e}", exc_info=True)
            return False

