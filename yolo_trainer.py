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
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import tempfile

# Fix for NumPy stderr issue in QGIS environment
from . import stderr_fix

from qgis.PyQt.QtCore import QObject, pyqtSignal, QThread, QTimer
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
    metrics_updated = pyqtSignal(dict)  # Метрики обучения
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
            'temp_dir': self.temp_dir,
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
        self.process = None
        self.is_canceled = False
        self.current_model = None
    
    def run(self):
        """Запускает обучение"""
        try:
            # Проверяем наличие ultralytics
            if not self._check_ultralytics():
                self.progress.training_finished.emit(False, "Библиотека ultralytics не установлена")
                return
            
            # Создаем скрипт обучения
            training_script = self._create_training_script()
            
            # Запускаем обучение
            self._run_training_script(training_script)
            
        except Exception as e:
            self.progress.training_finished.emit(False, f"Ошибка обучения: {e}")
    
    def cancel(self):
        """Отменяет обучение"""
        self.is_canceled = True
        if self.process:
            self.process.terminate()
        if self.current_model:
            # Останавливаем обучение модели
            try:
                self.current_model.stop = True
            except:
                pass
    
    def _check_ultralytics(self) -> bool:
        """Проверяет наличие библиотеки ultralytics"""
        try:
            import ultralytics
            return True
        except ImportError:
            return False
    
    def _create_training_script(self) -> str:
        """Создает Python скрипт для обучения"""
        script_content = f'''
import os
import sys
import json
import time
from pathlib import Path

# Настройка для предотвращения создания новых окон
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    # Устанавливаем флаг для предотвращения создания консольного окна
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)
    
    # Дополнительные настройки для предотвращения создания окон
    import subprocess
    import tempfile
    
    def log_print(*args, **kwargs):
        """Функция для логирования (заглушка)"""
        pass
else:
    def log_print(*args, **kwargs):
        """Функция для логирования (заглушка)"""
        pass

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    log_print(f"Ошибка импорта: {{e}}")
    sys.exit(1)

def train_model():
    try:
        # Конфигурация
        dataset_path = r"{self.config['dataset_path']}"
        model_type = "{self.config['model_type']}"
        task = "{self.config['task']}"
        epochs = {self.config['epochs']}
        batch_size = {self.config['batch_size']}
        image_size = {self.config['image_size']}
        learning_rate = {self.config['learning_rate']}
        device = "{self.config['device']}"
        pretrained = {self.config['pretrained']}
        save_dir = r"{self.config['save_dir']}"
        project_name = "{self.config['project_name']}"
        
        # Создаем директорию для сохранения
        os.makedirs(save_dir, exist_ok=True)
        
        # Загружаем модель
        model = YOLO(f"{{model_type}}.pt" if pretrained else f"{{model_type}}.yaml")
        
        # Настраиваем параметры обучения
        train_args = {{
            'data': os.path.join(dataset_path, 'dataset.yaml'),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': image_size,
            'lr0': learning_rate,
            'device': device,
            'project': save_dir,
            'name': project_name,
            'exist_ok': True,
            'save': True,
            'save_period': 10,
            'cache': False,
            'workers': 4,
            'patience': 50,
            'verbose': True
        }}
        
        # Добавляем дополнительные параметры
        additional_params = {json.dumps(self.config['additional_params'])}
        train_args.update(additional_params)
        
        # Запускаем обучение
        results = model.train(**train_args)
        
        # Сохраняем результаты
        results_path = os.path.join(save_dir, project_name, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump({{
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
            }}, f, indent=2)
        
        log_print("Обучение завершено успешно")
        
    except Exception as e:
        log_print(f"Ошибка обучения: {{e}}")
        sys.exit(1)
    finally:
        # Закрываем файл лога
        if 'log_fd' in locals() and log_fd is not None:
            try:
                log_fd.close()
            except Exception:
                pass

if __name__ == "__main__":
    train_model()
'''
        
        script_path = os.path.join(self.config['temp_dir'], 'train_model.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return script_path
    
    def _run_training_directly(self):
        """Запускает обучение напрямую через ultralytics API"""
        try:
            # В среде QGIS stdout/stderr могут быть None — починим это перед импортами/логированием
            if sys.stdout is None:
                sys.stdout = io.StringIO()
            if sys.stderr is None:
                sys.stderr = io.StringIO()

            # Проверяем наличие ultralytics
            if not self._check_ultralytics():
                self.progress.training_finished.emit(False, "Библиотека ultralytics не установлена")
                return
            
            # Импортируем необходимые модули
            from ultralytics import YOLO
            import torch
            
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
            
            # Создаем директорию для сохранения
            os.makedirs(save_dir, exist_ok=True)
            
            # Загружаем модель
            model = YOLO(f"{model_type}.pt" if pretrained else f"{model_type}.yaml")
            self.current_model = model
            
            # Настраиваем параметры обучения
            train_args = {
                'data': os.path.join(dataset_path, 'dataset.yaml'),
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': image_size,
                'lr0': learning_rate,
                'device': device,
                'project': save_dir,
                'name': project_name,
                'exist_ok': True,
                'save': True,
                'save_period': 10,
                'cache': False,
                'workers': 4,
                'patience': 50,
                'verbose': True
            }
            
            # Добавляем дополнительные параметры
            additional_params = self.config['additional_params']
            train_args.update(additional_params)
            
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
                    metrics: dict = {}
                    if isinstance(raw_metrics, dict):
                        metrics = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
                    else:
                        # Попытка вытащить наиболее типичные атрибуты
                        for key in ['loss', 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'lr']:
                            if hasattr(raw_metrics, key):
                                try:
                                    metrics[key] = float(getattr(raw_metrics, key))
                                except Exception:
                                    pass

                    # Обновляем и эмитим
                    if metrics:
                        self.progress.current_metrics.update(metrics)
                        self.progress.metrics_updated.emit(self.progress.current_metrics.copy())

                    if total_epochs > 0:
                        progress_percent = max(0, min(100, int((current_epoch / total_epochs) * 100)))
                        self.progress.progress_updated.emit(progress_percent)

                    # Строка статуса для инфо-окна
                    summary_parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
                    status_line = f"Эпоха {current_epoch}/{total_epochs} " + (" ".join(summary_parts) if summary_parts else "")
                    self.progress.status_updated.emit(status_line)
                except Exception:
                    # Не мешаем обучению, если парсинг метрик не удался
                    pass

            def _on_train_start(trainer_obj):
                self.progress.status_updated.emit("Старт обучения модели")

            def _on_train_end(trainer_obj):
                self.progress.status_updated.emit("Обучение завершено, сохранение результатов...")

            try:
                model.add_callback('on_fit_epoch_end', _on_fit_epoch_end)
                model.add_callback('on_train_start', _on_train_start)
                model.add_callback('on_train_end', _on_train_end)
            except Exception:
                # Если API коллбеков недоступен, просто продолжаем без онлайновых метрик
                pass
            
            # Простой запуск без сложных callback'ов
            results = model.train(**train_args)
            
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
    
    def _run_training_subprocess(self, script_path: str):
        """Запускает обучение через subprocess с улучшенными настройками"""
        try:
            # Настройки для Windows
            startupinfo = None
            creationflags = 0
            
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creationflags = subprocess.CREATE_NO_WINDOW
            
            # Попробуем использовать pythonw.exe вместо python.exe для Windows
            python_executable = sys.executable
            if sys.platform == "win32" and python_executable.endswith("python.exe"):
                python_executable = python_executable.replace("python.exe", "pythonw.exe")
            
            self.process = subprocess.Popen(
                [python_executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=self.config['temp_dir'],
                startupinfo=startupinfo,
                creationflags=creationflags,
                shell=False
            )
            
            # Читаем вывод в реальном времени
            while True:
                if self.is_canceled:
                    self.process.terminate()
                    self.progress.training_finished.emit(False, "Обучение отменено")
                    return
                
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                
                if output:
                    self._parse_training_output(output.strip())
            
            # Проверяем результат
            return_code = self.process.poll()
            if return_code == 0:
                self.progress.training_finished.emit(True, "Обучение завершено успешно")
            else:
                self.progress.training_finished.emit(False, f"Обучение завершилось с ошибкой (код: {return_code})")
                
        except Exception as e:
            self.progress.training_finished.emit(False, f"Ошибка запуска обучения: {e}")
    
    def _run_training_script(self, script_path: str):
        """Запускает скрипт обучения"""
        try:
            # Всегда пытаемся запустить обучение напрямую через ultralytics API
            # вместо создания отдельного процесса
            self._run_training_directly()
                
        except Exception as e:
            self.progress.training_finished.emit(False, f"Ошибка запуска обучения: {e}")
    
    def _parse_training_output(self, output: str):
        """Парсит вывод обучения для извлечения метрик"""
        try:
            # Ищем информацию об эпохах
            if 'epoch' in output.lower() and '/' in output:
                parts = output.split()
                for i, part in enumerate(parts):
                    if 'epoch' in part.lower() and i + 1 < len(parts):
                        epoch_info = parts[i + 1]
                        if '/' in epoch_info:
                            current, total = epoch_info.split('/')
                            try:
                                self.progress.current_epoch = int(current)
                                self.progress.total_epochs = int(total)
                                self.progress.epoch_updated.emit(int(current), int(total))
                                
                                # Вычисляем прогресс
                                progress_percent = int((int(current) / int(total)) * 100)
                                self.progress.progress_updated.emit(progress_percent)
                            except ValueError:
                                pass
            
            # Ищем метрики
            if any(metric in output.lower() for metric in ['loss', 'mAP', 'precision', 'recall']):
                metrics = {}
                parts = output.split()
                for i, part in enumerate(parts):
                    if part.lower() in ['loss', 'map', 'precision', 'recall'] and i + 1 < len(parts):
                        try:
                            value = float(parts[i + 1])
                            metrics[part.lower()] = value
                        except ValueError:
                            pass
                
                if metrics:
                    self.progress.current_metrics.update(metrics)
                    self.progress.metrics_updated.emit(self.progress.current_metrics.copy())
            
            # Обновляем статус
            if 'training' in output.lower():
                self.progress.status_updated.emit(output)
                
        except Exception as e:
            # Игнорируем ошибки парсинга
            pass


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
                plots=True
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

