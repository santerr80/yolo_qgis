# -*- coding: utf-8 -*-
"""
Главный менеджер для управления обучением и валидацией моделей YOLO
Объединяет все компоненты системы обучения
"""

import os
import logging
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime

from qgis.PyQt.QtCore import QObject, pyqtSignal

from .yolo_metrics_tracker import MetricsTracker
from .yolo_detection_trainer import DetectionTrainer, DetectionDatasetAnalyzer
from .yolo_segmentation_trainer import SegmentationTrainer, SegmentationDatasetAnalyzer
from .yolo_validation import AdvancedValidator, ModelComparator

logger = logging.getLogger(__name__)


class TrainingConfigManager:
    """Менеджер для сохранения и загрузки конфигураций обучения"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Инициализация менеджера конфигураций
        
        :param config_dir: Директория для хранения конфигураций
        """
        if config_dir is None:
            plugin_dir = os.path.dirname(__file__)
            config_dir = os.path.join(plugin_dir, 'training_configs')
        
        self.config_dir = config_dir
        os.makedirs(self.config_dir, exist_ok=True)
    
    def save_config(self, config: Dict, name: str) -> bool:
        """
        Сохраняет конфигурацию
        
        :param config: Словарь с конфигурацией
        :param name: Имя конфигурации
        :return: True если успешно
        """
        try:
            import json
            config_path = os.path.join(self.config_dir, f"{name}.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}", exc_info=True)
            return False
    
    def load_config(self, name: str) -> Optional[Dict]:
        """
        Загружает конфигурацию
        
        :param name: Имя конфигурации
        :return: Словарь с конфигурацией или None
        """
        try:
            import json
            config_path = os.path.join(self.config_dir, f"{name}.json")
            if not os.path.exists(config_path):
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}", exc_info=True)
            return None
    
    def list_configs(self) -> List[str]:
        """
        Возвращает список доступных конфигураций
        
        :return: Список имен конфигураций
        """
        try:
            configs = []
            for filename in os.listdir(self.config_dir):
                if filename.endswith('.json'):
                    configs.append(os.path.splitext(filename)[0])
            return sorted(configs)
        except Exception as e:
            logger.error(f"Ошибка получения списка конфигураций: {e}", exc_info=True)
            return []


class YOLOTrainingManager(QObject):
    """Главный менеджер для управления обучением YOLO моделей"""
    
    # Сигналы для связи с UI
    training_started = pyqtSignal(str)  # experiment_id
    training_progress = pyqtSignal(int, dict)  # epoch, metrics
    training_completed = pyqtSignal(str, bool, str)  # experiment_id, success, message
    validation_completed = pyqtSignal(str, dict)  # experiment_id, results
    status_message = pyqtSignal(str)  # message
    
    def __init__(self):
        """Инициализация менеджера тренировки"""
        super().__init__()
        
        # Инициализируем компоненты
        self.metrics_tracker = MetricsTracker()
        self.detection_trainer = DetectionTrainer(self.metrics_tracker)
        self.segmentation_trainer = SegmentationTrainer(self.metrics_tracker)
        self.validator = AdvancedValidator()
        self.comparator = ModelComparator()
        self.config_manager = TrainingConfigManager()
        
        # Текущие активные тренировки
        self.active_trainings = {}  # experiment_id -> trainer
        
        # Анализаторы датасетов
        self.detection_analyzer = DetectionDatasetAnalyzer()
        self.segmentation_analyzer = SegmentationDatasetAnalyzer()
    
    def analyze_dataset(self, dataset_path: str, task: str = 'detect') -> Dict[str, Any]:
        """
        Анализирует датасет
        
        :param dataset_path: Путь к датасету
        :param task: Тип задачи (detect/segment)
        :return: Результаты анализа
        """
        try:
            if task == 'detect':
                return self.detection_analyzer.analyze(dataset_path)
            elif task == 'segment':
                return self.segmentation_analyzer.analyze(dataset_path)
            else:
                return {'error': f'Неизвестный тип задачи: {task}'}
        except Exception as e:
            logger.error(f"Ошибка анализа датасета: {e}", exc_info=True)
            return {'error': str(e)}
    
    def start_detection_training(self, dataset_path: str, model_type: str = 'yolov8n',
                                epochs: int = 100, batch_size: int = 16,
                                image_size: int = 640, learning_rate: float = 0.01,
                                device: str = 'cpu', pretrained: bool = True,
                                save_dir: str = None, project_name: str = 'yolo_training',
                                **augmentation_params) -> Optional[str]:
        """
        Запускает обучение модели детекции
        
        :param dataset_path: Путь к датасету
        :param model_type: Тип модели
        :param epochs: Количество эпох
        :param batch_size: Размер батча
        :param image_size: Размер изображения
        :param learning_rate: Скорость обучения
        :param device: Устройство
        :param pretrained: Использовать предобученные веса
        :param save_dir: Директория сохранения
        :param project_name: Название проекта
        :param augmentation_params: Параметры аугментации
        :return: ID эксперимента или None
        """
        try:
            # Создаем experiment_id
            experiment_id = str(uuid.uuid4())
            
            # Запускаем обучение в отдельном потоке
            from qgis.PyQt.QtCore import QThread
            
            class TrainingThread(QThread):
                def __init__(self, trainer, params, manager, exp_id):
                    super().__init__()
                    self.trainer = trainer
                    self.params = params
                    self.manager = manager
                    self.experiment_id = exp_id
                
                def run(self):
                    try:
                        self.manager.status_message.emit("Запуск обучения...")
                        result = self.trainer.train(**self.params)
                        
                        if 'error' in result:
                            self.manager.training_completed.emit(
                                self.experiment_id,
                                False,
                                result['error']
                            )
                        else:
                            # Обновляем experiment_id в результате, если он был создан в тренере
                            if 'experiment_id' in result:
                                self.experiment_id = result['experiment_id']
                            self.manager.training_completed.emit(
                                self.experiment_id,
                                True,
                                result.get('message', 'Обучение завершено')
                            )
                    except Exception as e:
                        logger.error(f"Ошибка в потоке обучения: {e}", exc_info=True)
                        self.manager.training_completed.emit(
                            self.experiment_id,
                            False,
                            str(e)
                        )
            
            # Параметры для обучения (без experiment_id, так как он не нужен методу train)
            train_params = {
                'dataset_path': dataset_path,
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'learning_rate': learning_rate,
                'device': device,
                'pretrained': pretrained,
                'save_dir': save_dir,
                'project_name': project_name,
                'status_callback': lambda msg: self.status_message.emit(msg),
                'progress_callback': lambda epoch, metrics: self.training_progress.emit(epoch, metrics),
                **augmentation_params
            }
            
            # Создаем и запускаем поток (передаем experiment_id отдельно)
            thread = TrainingThread(self.detection_trainer, train_params, self, experiment_id)
            self.active_trainings[experiment_id] = thread
            thread.start()
            
            # Отправляем сигнал о начале обучения
            self.training_started.emit(experiment_id)
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Ошибка запуска обучения детекции: {e}", exc_info=True)
            return None
    
    def start_segmentation_training(self, dataset_path: str, model_type: str = 'yolov8n-seg',
                                   epochs: int = 100, batch_size: int = 16,
                                   image_size: int = 640, learning_rate: float = 0.01,
                                   device: str = 'cpu', pretrained: bool = True,
                                   save_dir: str = None, project_name: str = 'yolo_training',
                                   **augmentation_params) -> Optional[str]:
        """
        Запускает обучение модели сегментации
        
        :param dataset_path: Путь к датасету
        :param model_type: Тип модели
        :param epochs: Количество эпох
        :param batch_size: Размер батча
        :param image_size: Размер изображения
        :param learning_rate: Скорость обучения
        :param device: Устройство
        :param pretrained: Использовать предобученные веса
        :param save_dir: Директория сохранения
        :param project_name: Название проекта
        :param augmentation_params: Параметры аугментации
        :return: ID эксперимента или None
        """
        try:
            # Создаем experiment_id
            experiment_id = str(uuid.uuid4())
            
            # Запускаем обучение в отдельном потоке
            from qgis.PyQt.QtCore import QThread
            
            class TrainingThread(QThread):
                def __init__(self, trainer, params, manager, exp_id):
                    super().__init__()
                    self.trainer = trainer
                    self.params = params
                    self.manager = manager
                    self.experiment_id = exp_id
                
                def run(self):
                    try:
                        self.manager.status_message.emit("Запуск обучения...")
                        result = self.trainer.train(**self.params)
                        
                        if 'error' in result:
                            self.manager.training_completed.emit(
                                self.experiment_id,
                                False,
                                result['error']
                            )
                        else:
                            # Обновляем experiment_id в результате, если он был создан в тренере
                            if 'experiment_id' in result:
                                self.experiment_id = result['experiment_id']
                            self.manager.training_completed.emit(
                                self.experiment_id,
                                True,
                                result.get('message', 'Обучение завершено')
                            )
                    except Exception as e:
                        logger.error(f"Ошибка в потоке обучения: {e}", exc_info=True)
                        self.manager.training_completed.emit(
                            self.experiment_id,
                            False,
                            str(e)
                        )
            
            # Параметры для обучения (без experiment_id, так как он не нужен методу train)
            train_params = {
                'dataset_path': dataset_path,
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'learning_rate': learning_rate,
                'device': device,
                'pretrained': pretrained,
                'save_dir': save_dir,
                'project_name': project_name,
                'status_callback': lambda msg: self.status_message.emit(msg),
                'progress_callback': lambda epoch, metrics: self.training_progress.emit(epoch, metrics),
                **augmentation_params
            }
            
            # Создаем и запускаем поток (передаем experiment_id отдельно)
            thread = TrainingThread(self.segmentation_trainer, train_params, self, experiment_id)
            self.active_trainings[experiment_id] = thread
            thread.start()
            
            # Отправляем сигнал о начале обучения
            self.training_started.emit(experiment_id)
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Ошибка запуска обучения сегментации: {e}", exc_info=True)
            return None
    
    def cancel_training(self, experiment_id: str) -> bool:
        """
        Отменяет обучение
        
        :param experiment_id: ID эксперимента
        :return: True если успешно
        """
        try:
            if experiment_id in self.active_trainings:
                thread = self.active_trainings[experiment_id]
                if thread.isRunning():
                    thread.terminate()
                    thread.wait()
                del self.active_trainings[experiment_id]
                
                # Отменяем в тренере
                if hasattr(self.detection_trainer, 'cancel_training'):
                    self.detection_trainer.cancel_training()
                if hasattr(self.segmentation_trainer, 'cancel_training'):
                    self.segmentation_trainer.cancel_training()
                
                return True
            return False
        except Exception as e:
            logger.error(f"Ошибка отмены обучения: {e}", exc_info=True)
            return False
    
    def validate_model(self, model_path: str, dataset_path: str, task: str = 'detect',
                      experiment_id: Optional[str] = None, comprehensive: bool = True,
                      conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                      device: str = 'cpu') -> Dict[str, Any]:
        """
        Выполняет валидацию модели
        
        :param model_path: Путь к модели
        :param dataset_path: Путь к датасету
        :param task: Тип задачи
        :param experiment_id: ID эксперимента (опционально)
        :param comprehensive: Комплексная валидация
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU
        :param device: Устройство
        :return: Результаты валидации
        """
        try:
            results = self.validator.validate(
                model_path=model_path,
                dataset_path=dataset_path,
                task=task,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                comprehensive=comprehensive,
                device=device
            )
            
            if experiment_id:
                self.validation_completed.emit(experiment_id, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка валидации: {e}", exc_info=True)
            return {'error': str(e)}
    
    def compare_models(self, models: List[Dict[str, str]], dataset_path: str,
                      task: str = 'detect', conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45, device: str = 'cpu') -> Dict[str, Any]:
        """
        Сравнивает несколько моделей
        
        :param models: Список моделей с ключами 'path' и 'name'
        :param dataset_path: Путь к датасету
        :param task: Тип задачи
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU
        :param device: Устройство
        :return: Результаты сравнения
        """
        try:
            return self.comparator.compare(
                models=models,
                dataset_path=dataset_path,
                task=task,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device
            )
        except Exception as e:
            logger.error(f"Ошибка сравнения моделей: {e}", exc_info=True)
            return {'error': str(e)}
    
    def get_all_experiments(self) -> List[Dict]:
        """
        Получает список всех экспериментов
        
        :return: Список экспериментов
        """
        try:
            return self.metrics_tracker.database.list_experiments()
        except Exception as e:
            logger.error(f"Ошибка получения списка экспериментов: {e}", exc_info=True)
            return []
    
    def get_experiment_summary(self, experiment_id: str) -> Dict:
        """
        Получает сводку по эксперименту
        
        :param experiment_id: ID эксперимента
        :return: Словарь с информацией об эксперименте
        """
        try:
            experiment = self.metrics_tracker.database.get_experiment(experiment_id)
            if not experiment:
                return {}
            
            # Получаем метрики
            metrics = self.metrics_tracker.database.get_experiment_metrics(experiment_id)
            
            # Формируем финальные метрики
            final_metrics = experiment.get('final_metrics', {})
            
            return {
                'id': experiment.get('id'),
                'name': experiment.get('name'),
                'task': experiment.get('task'),
                'model_type': experiment.get('model_type'),
                'status': experiment.get('status'),
                'created_at': experiment.get('created_at'),
                'completed_at': experiment.get('completed_at'),
                'config': experiment.get('config', {}),
                'final_metrics': final_metrics
            }
        except Exception as e:
            logger.error(f"Ошибка получения сводки эксперимента: {e}", exc_info=True)
            return {}

