# -*- coding: utf-8 -*-
"""
Главный модуль для управления обучением и валидацией YOLO моделей
Объединяет все компоненты системы
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Fix for NumPy stderr issue in QGIS environment
from . import stderr_fix

from qgis.PyQt.QtCore import QObject, pyqtSignal, QThread
from qgis.PyQt.QtWidgets import QMessageBox

from .yolo_trainer import YOLOTrainer, TrainingProgress
from .yolo_detection_trainer import DetectionTrainer, DetectionDatasetAnalyzer
from .yolo_segmentation_trainer import SegmentationTrainer, SegmentationDatasetAnalyzer
from .yolo_validation import AdvancedValidator, ModelComparator
from .yolo_metrics_tracker import MetricsTracker, MetricsVisualizer


class YOLOTrainingManager(QObject):
    """Главный класс для управления обучением YOLO моделей"""
    
    # Сигналы для UI
    training_started = pyqtSignal(str)  # experiment_id
    training_progress = pyqtSignal(int, dict)  # epoch, metrics
    training_completed = pyqtSignal(str, bool, str)  # experiment_id, success, message
    validation_started = pyqtSignal(str)  # experiment_id
    validation_completed = pyqtSignal(str, dict)  # experiment_id, results
    
    def __init__(self, log_dir: str = None, db_path: str = None):
        super().__init__()
        
        # Инициализируем компоненты
        self.metrics_tracker = MetricsTracker(log_dir, db_path)
        self.validator = AdvancedValidator()
        self.model_comparator = ModelComparator()
        self.visualizer = MetricsVisualizer(self.metrics_tracker)
        
        # Тренажеры для разных задач
        self.detection_trainer = DetectionTrainer()
        self.segmentation_trainer = SegmentationTrainer()
        
        # Подключаем сигналы
        self._setup_signal_connections()
        
        # Текущие эксперименты
        self.active_experiments = {}
    
    def _setup_signal_connections(self):
        """Настраивает соединения сигналов"""
        # Подключаем сигналы от тренажеров
        self.detection_trainer.progress.progress_updated.connect(self._on_progress_updated)
        self.detection_trainer.progress.epoch_updated.connect(self._on_epoch_updated)
        self.detection_trainer.progress.metrics_updated.connect(self._on_metrics_updated)
        self.detection_trainer.progress.training_finished.connect(self._on_training_finished)
        
        self.segmentation_trainer.progress.progress_updated.connect(self._on_progress_updated)
        self.segmentation_trainer.progress.epoch_updated.connect(self._on_epoch_updated)
        self.segmentation_trainer.progress.metrics_updated.connect(self._on_metrics_updated)
        self.segmentation_trainer.progress.training_finished.connect(self._on_training_finished)
        
        # Подключаем сигналы от трекера метрик
        self.metrics_tracker.metrics_updated.connect(self._on_metrics_tracked)
        self.metrics_tracker.experiment_completed.connect(self._on_experiment_completed)
    
    def start_detection_training(self,
                               dataset_path: str,
                               model_type: str = 'yolov8n',
                               epochs: int = 100,
                               batch_size: int = 16,
                               image_size: int = 640,
                               learning_rate: float = 0.01,
                               device: str = 'cpu',
                               pretrained: bool = True,
                               save_dir: str = None,
                               project_name: str = None,
                               **kwargs) -> str:
        """
        Запускает обучение модели детекции
        
        :return: ID эксперимента
        """
        try:
            # Создаем ID эксперимента
            experiment_id = str(uuid.uuid4())
            
            # Подготавливаем конфигурацию
            config = {
                'task': 'detect',
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'learning_rate': learning_rate,
                'device': device,
                'pretrained': pretrained,
                'dataset_path': dataset_path,
                **kwargs
            }
            
            # Начинаем эксперимент в трекере метрик
            experiment_name = project_name or f"Detection_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.metrics_tracker.start_experiment(
                experiment_id, experiment_name, 'detect', model_type, dataset_path, config
            )
            
            # Сохраняем информацию об эксперименте
            self.active_experiments[experiment_id] = {
                'type': 'detection',
                'trainer': self.detection_trainer,
                'config': config,
                'start_time': datetime.now()
            }
            
            # Запускаем обучение
            success = self.detection_trainer.train_detection_model(
                dataset_path=dataset_path,
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                learning_rate=learning_rate,
                device=device,
                pretrained=pretrained,
                save_dir=save_dir,
                project_name=project_name,
                **kwargs
            )
            
            if success:
                self.training_started.emit(experiment_id)
                return experiment_id
            else:
                # Удаляем неудачный эксперимент
                if experiment_id in self.active_experiments:
                    del self.active_experiments[experiment_id]
                return None
                
        except Exception as e:
            print(f"Ошибка запуска обучения детекции: {e}")
            return None
    
    def start_segmentation_training(self,
                                  dataset_path: str,
                                  model_type: str = 'yolov8n-seg',
                                  epochs: int = 100,
                                  batch_size: int = 16,
                                  image_size: int = 640,
                                  learning_rate: float = 0.01,
                                  device: str = 'cpu',
                                  pretrained: bool = True,
                                  save_dir: str = None,
                                  project_name: str = None,
                                  **kwargs) -> str:
        """
        Запускает обучение модели сегментации
        
        :return: ID эксперимента
        """
        try:
            # Создаем ID эксперимента
            experiment_id = str(uuid.uuid4())
            
            # Подготавливаем конфигурацию
            config = {
                'task': 'segment',
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'learning_rate': learning_rate,
                'device': device,
                'pretrained': pretrained,
                'dataset_path': dataset_path,
                **kwargs
            }
            
            # Начинаем эксперимент в трекере метрик
            experiment_name = project_name or f"Segmentation_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.metrics_tracker.start_experiment(
                experiment_id, experiment_name, 'segment', model_type, dataset_path, config
            )
            
            # Сохраняем информацию об эксперименте
            self.active_experiments[experiment_id] = {
                'type': 'segmentation',
                'trainer': self.segmentation_trainer,
                'config': config,
                'start_time': datetime.now()
            }
            
            # Запускаем обучение
            success = self.segmentation_trainer.train_segmentation_model(
                dataset_path=dataset_path,
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                learning_rate=learning_rate,
                device=device,
                pretrained=pretrained,
                save_dir=save_dir,
                project_name=project_name,
                **kwargs
            )
            
            if success:
                self.training_started.emit(experiment_id)
                return experiment_id
            else:
                # Удаляем неудачный эксперимент
                if experiment_id in self.active_experiments:
                    del self.active_experiments[experiment_id]
                return None
                
        except Exception as e:
            print(f"Ошибка запуска обучения сегментации: {e}")
            return None
    
    def validate_model(self,
                      model_path: str,
                      dataset_path: str,
                      task: str = 'detect',
                      experiment_id: str = None,
                      comprehensive: bool = True) -> Dict:
        """
        Валидирует модель
        
        :param model_path: Путь к модели
        :param dataset_path: Путь к датасету
        :param task: Тип задачи
        :param experiment_id: ID эксперимента (если связан с обучением)
        :param comprehensive: Выполнять ли комплексную валидацию
        :return: Результаты валидации
        """
        try:
            if experiment_id:
                self.validation_started.emit(experiment_id)
            
            if comprehensive:
                # Комплексная валидация
                results = self.validator.comprehensive_validation(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    task=task
                )
            else:
                # Простая валидация
                if task == 'detect':
                    results = self.detection_trainer.validate_detection_model(
                        model_path, dataset_path
                    )
                else:
                    results = self.segmentation_trainer.validate_segmentation_model(
                        model_path, dataset_path
                    )
            
            # Логируем результаты валидации
            if experiment_id and 'error' not in results:
                # self.metrics_tracker.log_final_validation(
                #     model_path, dataset_path, results.get('performance_metrics', {})
                # )
                pass
            
            if experiment_id:
                self.validation_completed.emit(experiment_id, results)
            
            return results
            
        except Exception as e:
            error_result = {'error': f"Ошибка валидации: {e}"}
            if experiment_id:
                self.validation_completed.emit(experiment_id, error_result)
            return error_result
    
    def compare_models(self,
                      models: List[Dict],
                      dataset_path: str,
                      task: str = 'detect') -> Dict:
        """
        Сравнивает несколько моделей
        
        :param models: Список моделей для сравнения
        :param dataset_path: Путь к датасету
        :param task: Тип задачи
        :return: Результаты сравнения
        """
        try:
            return self.model_comparator.compare_models(
                models=models,
                dataset_path=dataset_path,
                task=task
            )
        except Exception as e:
            return {'error': f"Ошибка сравнения моделей: {e}"}
    
    def analyze_dataset(self, dataset_path: str, task: str = 'detect') -> Dict:
        """
        Анализирует датасет
        
        :param dataset_path: Путь к датасету
        :param task: Тип задачи
        :return: Результаты анализа
        """
        try:
            if task == 'detect':
                analyzer = DetectionDatasetAnalyzer(dataset_path)
            else:
                analyzer = SegmentationDatasetAnalyzer(dataset_path)
            
            return analyzer.analyze_dataset()
            
        except Exception as e:
            return {'error': f"Ошибка анализа датасета: {e}"}
    
    def get_experiment_summary(self, experiment_id: str) -> Dict:
        """Получает сводку по эксперименту"""
        return self.metrics_tracker.get_experiment_summary(experiment_id)
    
    def get_all_experiments(self) -> List[Dict]:
        """Получает список всех экспериментов"""
        return self.metrics_tracker.get_all_experiments()
    
    def create_training_plots(self, experiment_id: str, output_dir: str) -> bool:
        """Создает графики обучения"""
        return self.visualizer.create_training_plots(experiment_id, output_dir)
    
    def export_experiment_data(self, experiment_id: str, output_path: str, format: str = 'json') -> bool:
        """Экспортирует данные эксперимента"""
        return self.metrics_tracker.export_metrics(experiment_id, output_path, format)
    
    def cancel_training(self, experiment_id: str) -> bool:
        """Отменяет обучение"""
        try:
            if experiment_id in self.active_experiments:
                experiment = self.active_experiments[experiment_id]
                trainer = experiment['trainer']
                trainer.cancel_training()
                
                # Обновляем статус эксперимента
                self.metrics_tracker.database.update_experiment_status(experiment_id, 'cancelled')
                
                # Удаляем из активных
                del self.active_experiments[experiment_id]
                
                return True
            return False
        except Exception as e:
            print(f"Ошибка отмены обучения: {e}")
            return False
    
    def get_active_experiments(self) -> Dict:
        """Получает список активных экспериментов"""
        return self.active_experiments.copy()
    
    # Обработчики сигналов
    def _on_progress_updated(self, progress: int):
        """Обработчик обновления прогресса"""
        # Находим соответствующий эксперимент
        for exp_id, exp_info in self.active_experiments.items():
            if hasattr(exp_info['trainer'], 'progress') and exp_info['trainer'].progress == self.sender():
                # Можно добавить дополнительную логику
                break
    
    def _on_epoch_updated(self, current_epoch: int, total_epochs: int):
        """Обработчик обновления эпохи"""
        # Находим соответствующий эксперимент
        for exp_id, exp_info in self.active_experiments.items():
            if hasattr(exp_info['trainer'], 'progress') and exp_info['trainer'].progress == self.sender():
                # Можно добавить дополнительную логику
                break
    
    def _on_metrics_updated(self, metrics: Dict):
        """Обработчик обновления метрик"""
        # Находим соответствующий эксперимент
        for exp_id, exp_info in self.active_experiments.items():
            if hasattr(exp_info['trainer'], 'progress') and exp_info['trainer'].progress == self.sender():
                # Логируем метрики
                current_epoch = exp_info['trainer'].progress.current_epoch
                # self.metrics_tracker.log_training_metrics(current_epoch, metrics)
                
                # Отправляем сигнал
                self.training_progress.emit(current_epoch, metrics)
                break
    
    def _on_training_finished(self, success: bool, message: str):
        """Обработчик завершения обучения"""
        # Находим соответствующий эксперимент
        for exp_id, exp_info in self.active_experiments.items():
            if hasattr(exp_info['trainer'], 'progress') and exp_info['trainer'].progress == self.sender():
                # Обновляем статус эксперимента
                status = 'completed' if success else 'failed'
                self.metrics_tracker.database.update_experiment_status(exp_id, status)
                
                # Удаляем из активных
                del self.active_experiments[exp_id]
                
                # Отправляем сигнал
                self.training_completed.emit(exp_id, success, message)
                break
    
    def _on_metrics_tracked(self, metrics_data: Dict):
        """Обработчик отслеживания метрик"""
        # Можно добавить дополнительную логику обработки
        pass
    
    def _on_experiment_completed(self, experiment_id: str, results: Dict):
        """Обработчик завершения эксперимента"""
        # Можно добавить дополнительную логику обработки
        pass


class TrainingConfigManager:
    """Класс для управления конфигурациями обучения"""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or 'training_configs'
        os.makedirs(self.config_dir, exist_ok=True)
    
    def save_config(self, config: Dict, name: str) -> bool:
        """Сохраняет конфигурацию"""
        try:
            config_path = os.path.join(self.config_dir, f"{name}.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
            return False
    
    def load_config(self, name: str) -> Dict:
        """Загружает конфигурацию"""
        try:
            config_path = os.path.join(self.config_dir, f"{name}.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return {}
    
    def list_configs(self) -> List[str]:
        """Возвращает список доступных конфигураций"""
        try:
            configs = []
            for file in os.listdir(self.config_dir):
                if file.endswith('.json'):
                    configs.append(file[:-5])  # Убираем .json
            return sorted(configs)
        except Exception as e:
            print(f"Ошибка получения списка конфигураций: {e}")
            return []
    
    def delete_config(self, name: str) -> bool:
        """Удаляет конфигурацию"""
        try:
            config_path = os.path.join(self.config_dir, f"{name}.json")
            if os.path.exists(config_path):
                os.remove(config_path)
                return True
            return False
        except Exception as e:
            print(f"Ошибка удаления конфигурации: {e}")
            return False
    
    def get_default_configs(self) -> Dict:
        """Возвращает конфигурации по умолчанию"""
        return {
            'detection_fast': {
                'task': 'detect',
                'model_type': 'yolov8n',
                'epochs': 50,
                'batch_size': 16,
                'image_size': 640,
                'learning_rate': 0.01,
                'device': 'cpu',
                'pretrained': True
            },
            'detection_accurate': {
                'task': 'detect',
                'model_type': 'yolov8l',
                'epochs': 200,
                'batch_size': 8,
                'image_size': 640,
                'learning_rate': 0.005,
                'device': 'cpu',
                'pretrained': True
            },
            'segmentation_fast': {
                'task': 'segment',
                'model_type': 'yolov8n-seg',
                'epochs': 50,
                'batch_size': 16,
                'image_size': 640,
                'learning_rate': 0.01,
                'device': 'cpu',
                'pretrained': True,
                'copy_paste': 0.3
            },
            'segmentation_accurate': {
                'task': 'segment',
                'model_type': 'yolov8l-seg',
                'epochs': 200,
                'batch_size': 8,
                'image_size': 640,
                'learning_rate': 0.005,
                'device': 'cpu',
                'pretrained': True,
                'copy_paste': 0.3
            }
        }

