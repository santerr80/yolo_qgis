# -*- coding: utf-8 -*-
"""
Модуль для обучения и валидации YOLO моделей
Поддерживает детекцию и сегментацию объектов
"""

import os
import sys
import io
import json
import yaml
import shutil
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import tempfile

# Fix for NumPy stderr issue in QGIS environment
from . import stderr_fix

from qgis.PyQt.QtCore import QObject, pyqtSignal, QThread
from qgis.PyQt.QtWidgets import QMessageBox

# Настройка для предотвращения создания новых окон
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    # Устанавливаем флаг для предотвращения создания консольного окна
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)


class TrainingProgress(QObject):
    """Класс для отслеживания прогресса обучения"""
    
    progress_updated = pyqtSignal(int)  # Процент выполнения
    epoch_updated = pyqtSignal(int, int)  # Текущая эпоха, общее количество
    metrics_updated = pyqtSignal(dict)  # Метрики обучения (объединенные)
    training_metrics_updated = pyqtSignal(int, dict)  # Эпоха, метрики обучения
    validation_metrics_updated = pyqtSignal(int, dict)  # Эпоха, метрики валидации
    status_updated = pyqtSignal(str)  # Статус обучения
    training_finished = pyqtSignal(bool, str)  # Завершение (успех, сообщение)
    training_canceled = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_metrics = {}
        self.is_canceled = False
        self.is_running = False


class YOLOTrainer(QObject):
    """Основной класс для обучения YOLO моделей"""
    
    def __init__(self):
        super().__init__()
        self.progress = TrainingProgress()
        self.training_thread = None
        self.training_process = None
        self.temp_dir = None
        self.model_config = None
        
    def train_model(self, 
                   dataset_path: str,
                   model_type: str = 'yolov8n',
                   task: str = 'detect',
                   epochs: int = 100,
                   batch_size: int = 16,
                   image_size: int = 640,
                   learning_rate: float = 0.01,
                   device: str = 'cpu',
                   pretrained: bool = True,
                   save_dir: str = None,
                   project_name: str = 'yolo_training',
                   **kwargs) -> bool:
        """
        Запускает обучение YOLO модели
        
        :param dataset_path: Путь к датасету (должен содержать dataset.yaml)
        :param model_type: Тип модели (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        :param task: Тип задачи ('detect' или 'segment')
        :param epochs: Количество эпох
        :param batch_size: Размер батча
        :param image_size: Размер изображения
        :param learning_rate: Скорость обучения
        :param device: Устройство ('cpu', '0', '1', etc.)
        :param pretrained: Использовать предобученную модель
        :param save_dir: Директория для сохранения результатов
        :param project_name: Имя проекта
        :param kwargs: Дополнительные параметры
        :return: True если обучение запущено успешно
        """
        
        try:
            # Проверяем наличие датасета
            if not self._validate_dataset(dataset_path):
                return False
            
            # Создаем временную директорию для обучения
            self.temp_dir = tempfile.mkdtemp(prefix='yolo_training_')
            
            # Подготавливаем конфигурацию
            self.model_config = self._prepare_training_config(
                dataset_path=dataset_path,
                model_type=model_type,
                task=task,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                learning_rate=learning_rate,
                device=device,
                pretrained=pretrained,
                save_dir=save_dir or self.temp_dir,
                project_name=project_name,
                **kwargs
            )
            
            # Запускаем обучение в отдельном потоке
            self.training_thread = TrainingThread(self.model_config, self.progress)
            self.training_thread.finished.connect(self._on_training_finished)
            self.training_thread.start()
            
            self.progress.is_running = True
            self.progress.status_updated.emit("Обучение запущено...")
            
            return True
            
        except Exception as e:
            self.progress.training_finished.emit(False, f"Ошибка запуска обучения: {e}")
            return False
    
    def cancel_training(self):
        """Отменяет текущее обучение"""
        if self.training_thread and self.training_thread.isRunning():
            self.progress.is_canceled = True
            self.training_thread.cancel()
            self.progress.training_canceled.emit()
    
    def _validate_dataset(self, dataset_path: str) -> bool:
        """Проверяет валидность датасета"""
        try:
            # Проверяем наличие dataset.yaml
            yaml_path = os.path.join(dataset_path, 'dataset.yaml')
            if not os.path.exists(yaml_path):
                self.progress.training_finished.emit(False, "Не найден файл dataset.yaml в датасете")
                return False
            
            # Проверяем структуру датасета
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            required_keys = ['path', 'train', 'val', 'names']
            for key in required_keys:
                if key not in config:
                    self.progress.training_finished.emit(False, f"Отсутствует ключ '{key}' в dataset.yaml")
                    return False
            
            # Проверяем наличие директорий
            for split in ['train', 'val']:
                split_path = os.path.join(dataset_path, config[split])
                if not os.path.exists(split_path):
                    self.progress.training_finished.emit(False, f"Не найдена директория {split}: {split_path}")
                    return False
            
            return True
            
        except Exception as e:
            self.progress.training_finished.emit(False, f"Ошибка валидации датасета: {e}")
            return False
    
    def _prepare_training_config(self, **kwargs) -> Dict:
        """Подготавливает конфигурацию для обучения"""
        config = {
            'dataset_path': kwargs['dataset_path'],
            'model_type': kwargs['model_type'],
            'task': kwargs['task'],
            'epochs': kwargs['epochs'],
            'batch_size': kwargs['batch_size'],
            'image_size': kwargs['image_size'],
            'learning_rate': kwargs['learning_rate'],
            'device': kwargs['device'],
            'pretrained': kwargs['pretrained'],
            'save_dir': kwargs['save_dir'],
            'project_name': kwargs['project_name'],
            'additional_params': {k: v for k, v in kwargs.items() 
                                if k not in ['dataset_path', 'model_type', 'task', 'epochs', 
                                           'batch_size', 'image_size', 'learning_rate', 
                                           'device', 'pretrained', 'save_dir', 'project_name']}
        }
        return config
    
    def _on_training_finished(self):
        """Обработчик завершения обучения"""
        self.progress.is_running = False
        if self.temp_dir and os.path.exists(self.temp_dir):
            # Очищаем временную директорию
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Ошибка очистки временной директории: {e}")


class TrainingThread(QThread):
    """Поток для выполнения обучения"""
    
    def __init__(self, config: Dict, progress: TrainingProgress):
        super().__init__()
        self.config = config
        self.progress = progress
        self.is_canceled = False
        self.current_model = None
    
    def run(self):
        """Запускает обучение"""
        try:
            # Проверяем наличие ultralytics
            if not self._check_ultralytics():
                self.progress.training_finished.emit(False, "Библиотека ultralytics не установлена")
                return
            
            # Запускаем обучение напрямую через ultralytics API
            self._run_training_directly()
            
        except Exception as e:
            self.progress.training_finished.emit(False, f"Ошибка обучения: {e}")
    
    def cancel(self):
        """Отменяет обучение"""
        self.is_canceled = True
        if self.current_model:
            # Останавливаем обучение модели
            try:
                # Разные версии ultralytics по-разному реагируют на флаги остановки
                # Попробуем несколько вариантов, чтобы гарантировать остановку
                setattr(self.current_model, 'stop', True)
                if hasattr(self.current_model, 'trainer') and self.current_model.trainer is not None:
                    setattr(self.current_model.trainer, 'stop', True)
                # Сообщим в UI
                self.progress.status_updated.emit("Остановка обучения...")
            except:
                pass
    
    def _check_ultralytics(self) -> bool:
        """Проверяет наличие библиотеки ultralytics"""
        try:
            import ultralytics
            return True
        except ImportError:
            return False
    
    def _run_training_directly(self):
        """Запускает обучение напрямую через ultralytics API"""
        try:
            # В среде QGIS stdout/stderr могут быть None — починим это перед импортами/логированием
            if sys.stdout is None:
                sys.stdout = io.StringIO()
            if sys.stderr is None:
                sys.stderr = io.StringIO()
            
            # Настраиваем logging для работы в QGIS (где потоки могут быть None)
            import logging
            
            # Создаем безопасный поток для logging
            safe_log_stream = sys.stderr if sys.stderr is not None else io.StringIO()
            
            # Сохраняем оригинальный StreamHandler для использования в обертке
            original_stream_handler = logging.StreamHandler
            
            # Создаем безопасную обертку для StreamHandler
            class SafeStreamHandler(original_stream_handler):
                def __init__(self, stream=None):
                    # Если stream None или небезопасный, используем безопасный поток
                    if stream is None or stream == sys.stdout or stream == sys.stderr:
                        if sys.stderr is not None:
                            stream = sys.stderr
                        else:
                            stream = io.StringIO()
                    super().__init__(stream)
            
            # Заменяем StreamHandler на безопасную версию
            logging.StreamHandler = SafeStreamHandler
            
            # Исправляем все существующие обработчики logging
            for handler in logging.root.handlers[:]:
                if isinstance(handler, original_stream_handler):
                    stream = getattr(handler, 'stream', None)
                    if stream is None or not hasattr(stream, 'write'):
                        handler.stream = safe_log_stream
                    else:
                        try:
                            if not callable(getattr(stream, 'write', None)):
                                handler.stream = safe_log_stream
                        except Exception:
                            handler.stream = safe_log_stream
            
            # Устанавливаем non-interactive backend для matplotlib перед импортом
            # Это предотвращает попытки использовать GUI backend в QGIS
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            # Проверяем наличие ultralytics
            if not self._check_ultralytics():
                self.progress.training_finished.emit(False, "Библиотека ultralytics не установлена")
                return
            
            # Импортируем необходимые модули
            from ultralytics import YOLO
            import torch
            
            # Настройка PyTorch для предотвращения создания новых окон в Windows
            # Устанавливаем переменные окружения перед инициализацией CUDA
            if sys.platform == "win32":
                # Отключаем создание новых окон для CUDA процессов
                os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
                # Предотвращаем создание консольных окон
                os.environ['PYTHONUNBUFFERED'] = '1'
            
            # Настраиваем логирование ultralytics
            # Получаем логгер ultralytics и настраиваем его
            ultralytics_logger = logging.getLogger('ultralytics')
            
            # Уровень логирования можно настроить через переменную окружения или оставить по умолчанию
            # Устанавливаем уровень WARNING для уменьшения вывода (можно изменить на INFO или DEBUG)
            log_level = os.environ.get('ULTRALYTICS_LOG_LEVEL', 'WARNING')
            log_level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }
            ultralytics_log_level = log_level_map.get(log_level.upper(), logging.WARNING)
            ultralytics_logger.setLevel(ultralytics_log_level)
            
            # Удаляем все существующие обработчики ultralytics и добавляем безопасный
            for handler in ultralytics_logger.handlers[:]:
                ultralytics_logger.removeHandler(handler)
            
            # Создаем безопасный обработчик для ultralytics
            ultralytics_handler = logging.StreamHandler(safe_log_stream)
            ultralytics_handler.setLevel(ultralytics_log_level)
            ultralytics_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            ultralytics_logger.addHandler(ultralytics_handler)
            ultralytics_logger.propagate = False  # Отключаем распространение в root logger
            
            # Также настраиваем логирование для подмодулей ultralytics
            for submodule in ['ultralytics.engine', 'ultralytics.utils', 'ultralytics.models']:
                sub_logger = logging.getLogger(submodule)
                sub_logger.setLevel(ultralytics_log_level)
                sub_logger.propagate = False
                # Удаляем существующие обработчики
                for handler in sub_logger.handlers[:]:
                    sub_logger.removeHandler(handler)
                # Добавляем безопасный обработчик
                sub_handler = logging.StreamHandler(safe_log_stream)
                sub_handler.setLevel(ultralytics_log_level)
                sub_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                sub_logger.addHandler(sub_handler)
            
            # Повторно исправляем обработчики logging после импорта ultralytics
            # (ultralytics может создать новые обработчики с None stream)
            for handler in logging.root.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    stream = getattr(handler, 'stream', None)
                    if stream is None or not hasattr(stream, 'write'):
                        handler.stream = safe_log_stream
                    else:
                        try:
                            if not callable(getattr(stream, 'write', None)):
                                handler.stream = safe_log_stream
                        except Exception:
                            handler.stream = safe_log_stream
            
            # Настройка multiprocessing для предотвращения создания новых окон в Windows
            # В Windows multiprocessing использует spawn, что может создавать новые процессы
            if sys.platform == "win32":
                import multiprocessing
                # Устанавливаем метод запуска процессов для Windows
                # Используем 'spawn' с защитой от создания новых окон
                try:
                    # Устанавливаем переменные окружения для предотвращения создания консольных окон
                    os.environ['PYTHONUNBUFFERED'] = '1'
                    # Отключаем создание новых окон для дочерних процессов
                    multiprocessing.set_start_method('spawn', force=True)
                except RuntimeError:
                    # Метод уже установлен, пропускаем
                    pass
            
            # Проверка доступности CUDA устройств
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"Number of GPUs: {torch.cuda.device_count()}")
                print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA version: N/A (CUDA not available)")
                print("Number of GPUs: 0")
            
            # Конфигурация
            dataset_path = self.config['dataset_path']
            model_type = self.config['model_type']
            task = self.config['task']
            epochs = self.config['epochs']
            batch_size = self.config['batch_size']
            image_size = self.config['image_size']
            learning_rate = self.config['learning_rate']
            device = self.config['device']
            pretrained = self.config['pretrained']
            save_dir = self.config['save_dir']
            project_name = self.config['project_name']
            
            # Проверяем доступность CUDA и автоматически переключаемся на CPU при необходимости
            if device != 'cpu' and not torch.cuda.is_available():
                self.progress.status_updated.emit("CUDA недоступен, переключение на CPU...")
                device = 'cpu'
            
            # Создаем директорию для сохранения
            os.makedirs(save_dir, exist_ok=True)
            
            # Расширенные параметры дообучения
            additional_params = self.config['additional_params']
            base_weights_path: Optional[str] = additional_params.get('base_weights_path')
            resume_training: bool = bool(additional_params.get('resume_training', False))
            freeze_layers = additional_params.get('freeze_layers')  # int | list | None
            strict_class_check: bool = bool(additional_params.get('strict_class_check', False))
            finetune_lr: Optional[float] = additional_params.get('finetune_lr')

            # Загружаем модель (предпочтительно из пользовательских весов для дообучения)
            if base_weights_path and os.path.exists(base_weights_path):
                model = YOLO(base_weights_path)
            else:
                model = YOLO(f"{model_type}.pt" if pretrained else f"{model_type}.yaml")
            self.current_model = model
            
            # Определяем количество workers для DataLoader
            # В Windows с CUDA лучше использовать workers=0, чтобы избежать проблем с multiprocessing
            # Это предотвратит создание новых процессов и окон
            num_workers = 0 if sys.platform == "win32" else 4
            
            # Настраиваем параметры обучения
            # verbose=False уменьшает вывод в консоль (логирование контролируется через logging)
            train_args = {
                'data': os.path.join(dataset_path, 'dataset.yaml'),
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': image_size,
                'lr0': finetune_lr if isinstance(finetune_lr, (int, float)) and finetune_lr > 0 else learning_rate,
                'device': device,
                'project': save_dir,
                'name': project_name,
                'exist_ok': True,
                'save': True,
                'save_period': 10,
                'cache': False,
                'workers': num_workers,  # Используем 0 для Windows, чтобы избежать создания новых окон
                'patience': 50,
                'verbose': False  # Отключаем verbose вывод, используем наше логирование
            }
            
            # Добавляем остальные дополнительные параметры (кроме тех, что обрабатываем вручную)
            for k, v in additional_params.items():
                if k in ['base_weights_path', 'resume_training', 'freeze_layers', 'strict_class_check', 'finetune_lr']:
                    continue
                train_args[k] = v

            # Проверка совместимости количества классов между датасетом и моделью
            try:
                dataset_yaml_path = os.path.join(dataset_path, 'dataset.yaml')
                with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
                    ds_cfg = yaml.safe_load(f)
                ds_num_classes = len(ds_cfg.get('names', []))

                model_num_classes = None
                # Ultralytics v8
                if hasattr(model, 'model'):
                    # Попытаться извлечь число классов из структуры модели
                    if hasattr(model.model, 'nc'):
                        model_num_classes = int(getattr(model.model, 'nc'))
                    elif hasattr(model.model, 'names'):
                        model_num_classes = len(getattr(model.model, 'names'))
                    elif hasattr(model.model, 'yaml') and isinstance(model.model.yaml, dict):
                        model_num_classes = int(model.model.yaml.get('nc') or ds_num_classes)
                if model_num_classes is None and hasattr(model, 'names'):
                    model_num_classes = len(getattr(model, 'names'))

                if model_num_classes is not None and ds_num_classes and model_num_classes != ds_num_classes:
                    msg = f"Несовпадение числа классов: модель={model_num_classes}, датасет={ds_num_classes}"
                    if strict_class_check:
                        self.progress.training_finished.emit(False, msg)
                        return
                    else:
                        self.progress.status_updated.emit(f"Предупреждение: {msg}. Продолжаем дообучение.")
            except Exception:
                # Не блокируем обучение при ошибках проверки
                pass

            # Возможное замораживание слоев при дообучении
            try:
                if freeze_layers is not None:
                    # Ультралитикс имеет метод freeze в некоторых версиях
                    if hasattr(model, 'freeze'):
                        model.freeze(freeze_layers)
                    else:
                        # Фоллбек: заморозка параметров вручную
                        import torch.nn as nn  # noqa: F401  (для типизации/совместимости)
                        if isinstance(freeze_layers, int):
                            frozen = 0
                            for i, (name, param) in enumerate(model.model.named_parameters() if hasattr(model, 'model') else model.named_parameters()):
                                if i < int(freeze_layers):
                                    param.requires_grad = False
                                    frozen += 1
                            self.progress.status_updated.emit(f"Заморожено параметров: {frozen}")
                        elif isinstance(freeze_layers, (list, tuple)):
                            frozen = 0
                            named_params = dict((model.model.named_parameters() if hasattr(model, 'model') else model.named_parameters()))
                            for layer_name in freeze_layers:
                                for name, param in named_params.items():
                                    if name.startswith(str(layer_name)):
                                        param.requires_grad = False
                                        frozen += 1
                            self.progress.status_updated.emit(f"Заморожено параметров: {frozen}")
            except Exception:
                # Не прерываем обучение, если не удалось корректно заморозить
                pass

            # Поддержка возобновления обучения (resume) при наличии last.pt
            if resume_training:
                last_ckpt = os.path.join(save_dir, project_name, 'weights', 'last.pt')
                if os.path.exists(last_ckpt):
                    train_args['resume'] = True
                else:
                    # Если нет last.pt для текущего проекта, но указан base_weights_path — просто продолжим с ним
                    self.progress.status_updated.emit("Режим возобновления включен, но last.pt не найден. Продолжим без resume.")
            
            # Запускаем обучение
            self.progress.total_epochs = epochs
            self.progress.status_updated.emit("Начинаем обучение...")

            # Коллбеки Ultralytics для онлайновых метрик по эпохам
            def _on_fit_epoch_end(trainer_obj):
                try:
                    current_epoch = int(getattr(trainer_obj, 'epoch', 0)) + 1
                    total_epochs = int(getattr(trainer_obj, 'args', {}).get('epochs', self.progress.total_epochs) or self.progress.total_epochs)
                    self.progress.current_epoch = current_epoch
                    self.progress.total_epochs = total_epochs

                    # Извлечь метрики: trainer_obj.metrics может быть dict или объект
                    raw_metrics = getattr(trainer_obj, 'metrics', {})
                    
                    # Разделяем метрики на training и validation
                    training_metrics = {}
                    validation_metrics = {}
                    all_metrics = {}
                    
                    # Пытаемся извлечь метрики из разных источников
                    if isinstance(raw_metrics, dict):
                        all_metrics = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
                    else:
                        # Попытка вытащить наиболее типичные атрибуты
                        for key in ['loss', 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'lr']:
                            if hasattr(raw_metrics, key):
                                try:
                                    value = getattr(raw_metrics, key)
                                    if isinstance(value, (int, float)):
                                        all_metrics[key] = float(value)
                                except Exception:
                                    pass
                    
                    # Разделяем метрики на training и validation
                    # В Ultralytics метрики обычно имеют префиксы train/ и val/ или metrics/
                    for key, value in all_metrics.items():
                        key_lower = key.lower()
                        # Training метрики (loss и т.д.)
                        if any(x in key_lower for x in ['train', 'loss', 'lr', 'box_loss', 'cls_loss', 'dfl_loss', 'seg_loss']):
                            # Убираем префикс train/ если есть
                            clean_key = key.replace('train/', '').replace('train_', '')
                            training_metrics[clean_key] = value
                        # Validation метрики (mAP, precision, recall и т.д.)
                        elif any(x in key_lower for x in ['val', 'map', 'precision', 'recall', 'f1', 'metrics']):
                            # Убираем префикс val/ или metrics/ если есть
                            clean_key = key.replace('val/', '').replace('val_', '').replace('metrics/', '').replace('metrics_', '')
                            validation_metrics[clean_key] = value
                        # Если не определено, добавляем в обе категории для совместимости
                        else:
                            training_metrics[key] = value
                    
                    # Также пытаемся извлечь метрики из результатов валидации
                    try:
                        # Способ 1: Через validator.metrics (основной способ в Ultralytics)
                        if hasattr(trainer_obj, 'validator') and trainer_obj.validator:
                            validator = trainer_obj.validator
                            if hasattr(validator, 'metrics'):
                                val_metrics = validator.metrics
                                if isinstance(val_metrics, dict):
                                    for k, v in val_metrics.items():
                                        if isinstance(v, (int, float)):
                                            validation_metrics[k] = float(v)
                                elif hasattr(val_metrics, 'box'):
                                    box = val_metrics.box
                                    if hasattr(box, 'map50'):
                                        validation_metrics['mAP50'] = float(box.map50)
                                    if hasattr(box, 'map'):
                                        validation_metrics['mAP50-95'] = float(box.map)
                                    if hasattr(box, 'mp'):
                                        validation_metrics['precision'] = float(box.mp)
                                    if hasattr(box, 'mr'):
                                        validation_metrics['recall'] = float(box.mr)
                        
                        # Способ 2: Через results (результаты последней валидации)
                        if hasattr(trainer_obj, 'results') and trainer_obj.results:
                            results = trainer_obj.results
                            if hasattr(results, 'box'):
                                box = results.box
                                if hasattr(box, 'map50'):
                                    validation_metrics['mAP50'] = float(box.map50)
                                if hasattr(box, 'map'):
                                    validation_metrics['mAP50-95'] = float(box.map)
                                if hasattr(box, 'mp'):
                                    validation_metrics['precision'] = float(box.mp)
                                if hasattr(box, 'mr'):
                                    validation_metrics['recall'] = float(box.mr)
                        
                        # Способ 3: Через results_dict
                        if hasattr(trainer_obj, 'results_dict') and trainer_obj.results_dict:
                            results_dict = trainer_obj.results_dict
                            # Извлекаем метрики валидации
                            if 'metrics' in results_dict:
                                for k, v in results_dict['metrics'].items():
                                    if isinstance(v, (int, float)):
                                        validation_metrics[k] = float(v)
                            # Извлекаем mAP метрики
                            if 'maps' in results_dict:
                                maps = results_dict['maps']
                                if isinstance(maps, (list, tuple)) and len(maps) >= 2:
                                    validation_metrics['mAP50-95'] = float(maps[0]) if maps[0] is not None else 0.0
                                    validation_metrics['mAP50'] = float(maps[1]) if maps[1] is not None else 0.0
                                elif isinstance(maps, dict):
                                    for k, v in maps.items():
                                        if isinstance(v, (int, float)):
                                            validation_metrics[f'mAP{k}'] = float(v)
                        
                        # Способ 4: Прямой доступ к метрикам из trainer.metrics (может содержать val_* префиксы)
                        if isinstance(raw_metrics, dict):
                            for key, value in raw_metrics.items():
                                key_lower = key.lower()
                                # Ищем метрики валидации с префиксами val_ или metrics/
                                if any(prefix in key_lower for prefix in ['val_map', 'val_precision', 'val_recall', 'metrics/map', 'metrics/precision', 'metrics/recall']):
                                    clean_key = key.replace('val_', '').replace('val/', '').replace('metrics/', '').replace('metrics_', '')
                                    if isinstance(value, (int, float)):
                                        validation_metrics[clean_key] = float(value)
                        
                        # Способ 5: Через last (последние результаты)
                        if hasattr(trainer_obj, 'last') and trainer_obj.last:
                            last = trainer_obj.last
                            if hasattr(last, 'box'):
                                box = last.box
                                if hasattr(box, 'map50'):
                                    validation_metrics['mAP50'] = float(box.map50)
                                if hasattr(box, 'map'):
                                    validation_metrics['mAP50-95'] = float(box.map)
                                if hasattr(box, 'mp'):
                                    validation_metrics['precision'] = float(box.mp)
                                if hasattr(box, 'mr'):
                                    validation_metrics['recall'] = float(box.mr)
                        
                        # Отладочная информация
                        if not validation_metrics:
                            print(f"Предупреждение: Метрики валидации не найдены для эпохи {current_epoch}")
                            print(f"Доступные атрибуты trainer: {dir(trainer_obj)[:20]}")
                            if hasattr(trainer_obj, 'validator'):
                                print(f"Validator доступен: {trainer_obj.validator is not None}")
                                if trainer_obj.validator:
                                    print(f"Validator атрибуты: {dir(trainer_obj.validator)[:20]}")
                    except Exception as e:
                        # Логируем ошибку для отладки
                        print(f"Ошибка извлечения метрик валидации: {e}")
                        import traceback
                        traceback.print_exc()

                    # Эмитим метрики отдельно для training и validation
                    if training_metrics:
                        self.progress.training_metrics_updated.emit(current_epoch, training_metrics)
                    
                    if validation_metrics:
                        self.progress.validation_metrics_updated.emit(current_epoch, validation_metrics)

                    # Обновляем объединенные метрики для обратной совместимости
                    if training_metrics or validation_metrics:
                        combined_metrics = {**training_metrics, **validation_metrics}
                        self.progress.current_metrics.update(combined_metrics)
                        self.progress.metrics_updated.emit(combined_metrics.copy())

                    if total_epochs > 0:
                        progress_percent = max(0, min(100, int((current_epoch / total_epochs) * 100)))
                        self.progress.progress_updated.emit(progress_percent)

                    # Строка статуса для инфо-окна
                    status_parts = []
                    if training_metrics:
                        train_parts = [f"{k}={v:.4f}" for k, v in list(training_metrics.items())[:3]]
                        status_parts.append(f"Train: {' '.join(train_parts)}")
                    if validation_metrics:
                        val_parts = [f"{k}={v:.4f}" for k, v in list(validation_metrics.items())[:3]]
                        status_parts.append(f"Val: {' '.join(val_parts)}")
                    
                    status_line = f"Эпоха {current_epoch}/{total_epochs} " + (" | ".join(status_parts) if status_parts else "")
                    self.progress.status_updated.emit(status_line)
                except Exception as e:
                    # Логируем ошибку, но не мешаем обучению
                    import traceback
                    print(f"Ошибка извлечения метрик: {e}")
                    print(traceback.format_exc())
                    pass

            def _on_train_start(trainer_obj):
                self.progress.status_updated.emit("Старт обучения модели")

            def _on_train_end(trainer_obj):
                self.progress.status_updated.emit("Обучение завершено, сохранение результатов...")
            
            def _on_val_end(trainer_obj):
                """Коллбек после завершения валидации - извлекаем метрики валидации"""
                try:
                    current_epoch = int(getattr(trainer_obj, 'epoch', 0)) + 1
                    validation_metrics = {}
                    
                    # Извлекаем метрики валидации из validator
                    if hasattr(trainer_obj, 'validator') and trainer_obj.validator:
                        validator = trainer_obj.validator
                        if hasattr(validator, 'metrics'):
                            val_metrics = validator.metrics
                            if isinstance(val_metrics, dict):
                                for k, v in val_metrics.items():
                                    if isinstance(v, (int, float)):
                                        validation_metrics[k] = float(v)
                            elif hasattr(val_metrics, 'box'):
                                box = val_metrics.box
                                if hasattr(box, 'map50'):
                                    validation_metrics['mAP50'] = float(box.map50)
                                if hasattr(box, 'map'):
                                    validation_metrics['mAP50-95'] = float(box.map)
                                if hasattr(box, 'mp'):
                                    validation_metrics['precision'] = float(box.mp)
                                if hasattr(box, 'mr'):
                                    validation_metrics['recall'] = float(box.mr)
                    
                    # Также проверяем results
                    if hasattr(trainer_obj, 'results') and trainer_obj.results:
                        results = trainer_obj.results
                        if hasattr(results, 'box'):
                            box = results.box
                            if 'mAP50' not in validation_metrics and hasattr(box, 'map50'):
                                validation_metrics['mAP50'] = float(box.map50)
                            if 'mAP50-95' not in validation_metrics and hasattr(box, 'map'):
                                validation_metrics['mAP50-95'] = float(box.map)
                            if 'precision' not in validation_metrics and hasattr(box, 'mp'):
                                validation_metrics['precision'] = float(box.mp)
                            if 'recall' not in validation_metrics and hasattr(box, 'mr'):
                                validation_metrics['recall'] = float(box.mr)
                    
                    # Эмитим метрики валидации
                    if validation_metrics:
                        self.progress.validation_metrics_updated.emit(current_epoch, validation_metrics)
                        print(f"Метрики валидации для эпохи {current_epoch}: {validation_metrics}")
                    else:
                        print(f"Предупреждение: Метрики валидации не найдены в on_val_end для эпохи {current_epoch}")
                except Exception as e:
                    print(f"Ошибка в on_val_end: {e}")
                    import traceback
                    traceback.print_exc()

            # Коллбек с частой проверкой отмены перед эпохой и перед батчем
            def _on_fit_epoch_start(trainer_obj):
                try:
                    if self.is_canceled or self.progress.is_canceled:
                        try:
                            setattr(trainer_obj, 'stop', True)
                        except Exception:
                            pass
                        try:
                            if hasattr(self.current_model, 'trainer') and self.current_model.trainer is not None:
                                setattr(self.current_model.trainer, 'stop', True)
                        except Exception:
                            pass
                        self.progress.status_updated.emit("Остановка обучения по запросу пользователя")
                except Exception:
                    pass

            def _on_train_batch_start(trainer_obj):
                try:
                    if self.is_canceled or self.progress.is_canceled:
                        try:
                            setattr(trainer_obj, 'stop', True)
                        except Exception:
                            pass
                        try:
                            if hasattr(self.current_model, 'trainer') and self.current_model.trainer is not None:
                                setattr(self.current_model.trainer, 'stop', True)
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                model.add_callback('on_fit_epoch_end', _on_fit_epoch_end)
                model.add_callback('on_fit_epoch_start', _on_fit_epoch_start)
                model.add_callback('on_train_batch_start', _on_train_batch_start)
                model.add_callback('on_train_start', _on_train_start)
                model.add_callback('on_train_end', _on_train_end)
                model.add_callback('on_val_end', _on_val_end)  # Коллбек после валидации
            except Exception as e:
                # Если API коллбеков недоступен, просто продолжаем без онлайновых метрик
                print(f"Предупреждение: Не удалось добавить некоторые коллбеки: {e}")
                pass
            
            # Простой запуск без сложных callback'ов
            results = model.train(**train_args)
            
            # Если была запрошена отмена — завершаем корректно с соответствующим сообщением
            if self.is_canceled or self.progress.is_canceled:
                self.progress.progress_updated.emit(100)
                self.progress.status_updated.emit("Обучение остановлено пользователем")
                self.progress.training_finished.emit(False, "Обучение остановлено пользователем")
                return

            # Обновляем прогресс после завершения
            self.progress.progress_updated.emit(100)
            self.progress.status_updated.emit("Обучение завершено")
            
            # Сохраняем результаты
            try:
                results_path = os.path.join(save_dir, project_name, 'training_results.json')
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, 'w') as f:
                    import json
                    import time
                    json.dump({
                        'model_type': model_type,
                        'task': task,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'image_size': image_size,
                        'learning_rate': learning_rate,
                        'device': device,
                        'pretrained': pretrained,
                        'dataset_path': dataset_path,
                        'training_completed': True,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }, f, indent=2)
            except Exception as e:
                print(f"Ошибка сохранения результатов: {e}")
            
            self.progress.training_finished.emit(True, "Обучение завершено успешно")
            
        except Exception as e:
            self.progress.training_finished.emit(False, f"Ошибка обучения: {e}")
    
class ModelValidator:
    """Класс для валидации обученных моделей"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_model(self, 
                      model_path: str,
                      dataset_path: str,
                      task: str = 'detect',
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45,
                      max_det: int = 300) -> Dict:
        """
        Валидирует обученную модель
        
        :param model_path: Путь к обученной модели
        :param dataset_path: Путь к датасету
        :param task: Тип задачи ('detect' или 'segment')
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU
        :param max_det: Максимальное количество детекций
        :return: Словарь с результатами валидации
        """
        try:
            # В среде QGIS stdout/stderr могут быть None — починим это
            if sys.stdout is None:
                sys.stdout = io.StringIO()
            if sys.stderr is None:
                sys.stderr = io.StringIO()
            
            # Отключаем любые GUI backend'ы Qt/Matplotlib
            try:
                os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            except Exception:
                pass
            
            # Гарантируем неинтерактивный backend для matplotlib
            try:
                import matplotlib
                matplotlib.use('Agg')
            except Exception:
                pass
            
            from ultralytics import YOLO
            
            # Загружаем модель
            model = YOLO(model_path)
            
            # Запускаем валидацию
            results = model.val(
                data=os.path.join(dataset_path, 'dataset.yaml'),
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                save_json=True,
                save_hybrid=True,
                plots=False,      # не создавать GUI-графики
                show=False,       # не показывать окна
                verbose=False,
                workers=0         # без дополнительных процессов/окон в Windows
            )
            
            # Извлекаем метрики
            validation_results = {
                'model_path': model_path,
                'dataset_path': dataset_path,
                'task': task,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'max_det': max_det,
                'metrics': {
                    'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
                    'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
                    'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
                    'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Добавляем метрики по классам если доступны
            if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
                validation_results['class_metrics'] = {}
                for i, class_idx in enumerate(results.box.ap_class_index):
                    if i < len(results.box.ap):
                        validation_results['class_metrics'][int(class_idx)] = {
                            'mAP50': float(results.box.ap50[i]) if hasattr(results.box, 'ap50') and i < len(results.box.ap50) else 0.0,
                            'mAP50-95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0
                        }
            
            self.validation_results = validation_results
            return validation_results
            
        except Exception as e:
            return {
                'error': str(e),
                'model_path': model_path,
                'dataset_path': dataset_path,
                'timestamp': datetime.now().isoformat()
            }
    
    def save_validation_results(self, output_path: str):
        """Сохраняет результаты валидации в файл"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Ошибка сохранения результатов валидации: {e}")
            return False


class ModelPredictor:
    """Класс для предсказаний с использованием обученной модели"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загружает модель"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model = None
    
    def predict(self, 
                source: Union[str, List[str]],
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.45,
                max_det: int = 300,
                save_results: bool = False,
                output_dir: str = None) -> List[Dict]:
        """
        Выполняет предсказания
        
        :param source: Путь к изображению или список путей
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU
        :param max_det: Максимальное количество детекций
        :param save_results: Сохранять ли результаты
        :param output_dir: Директория для сохранения
        :return: Список результатов предсказаний
        """
        if not self.model:
            return []
        
        try:
            results = self.model(
                source=source,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                save=save_results,
                project=output_dir or 'predictions',
                name='results'
            )
            
            predictions = []
            for result in results:
                pred_data = {
                    'image_path': result.path,
                    'boxes': [],
                    'masks': [],
                    'confidence_scores': []
                }
                
                if result.boxes is not None:
                    for box in result.boxes:
                        pred_data['boxes'].append({
                            'class_id': int(box.cls[0]),
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist()
                        })
                        pred_data['confidence_scores'].append(float(box.conf[0]))
                
                if result.masks is not None:
                    for mask in result.masks:
                        pred_data['masks'].append({
                            'class_id': int(mask.cls[0]),
                            'confidence': float(mask.conf[0]),
                            'segmentation': mask.xy[0].tolist()
                        })
                
                predictions.append(pred_data)
            
            return predictions
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return []

