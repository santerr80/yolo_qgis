# -*- coding: utf-8 -*-
"""
Тренер для обучения моделей детекции объектов YOLOv8 и YOLOv11
Использует стандартные методы ultralytics
"""

import os
import sys
import logging
from typing import Dict, Optional, Any, Callable
from pathlib import Path

# Настройка PyTorch для предотвращения создания новых окон в Windows
# Устанавливаем переменные окружения перед инициализацией CUDA
if sys.platform == "win32":
    # Отключаем создание новых окон для CUDA процессов
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    # Предотвращаем создание консольных окон
    os.environ['PYTHONUNBUFFERED'] = '1'

# Настройка multiprocessing для предотвращения конфликтов с QGIS
# Отключаем использование multiprocessing на уровне окружения
# Это предотвращает попытки QGIS интерпретировать аргументы multiprocessing как пути к файлам
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# Предотвращаем использование multiprocessing в ultralytics
os.environ['YOLO_WORKERS'] = '0'

# Настройка matplotlib на неинтерактивный режим для предотвращения открытия окон
try:
    import matplotlib
    matplotlib.use('Agg')  # Неинтерактивный бэкенд
except ImportError:
    pass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics не установлен")

logger = logging.getLogger(__name__)


class DetectionTrainer:
    """Тренер для моделей детекции объектов"""
    
    def __init__(self, model_type: str = "yolov8n", device: str = "cpu"):
        """Инициализация тренера
        
        Args:
            model_type: Тип модели (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x,
                       yolov11n, yolov11s, yolov11m, yolov11l, yolov11x)
            device: Устройство для обучения ('cpu' или '0' для GPU)
        """
        if YOLO is None:
            raise ImportError("ultralytics не установлен. Установите: pip install ultralytics")
        
        self.model_type = model_type
        self.device = device
        self.model = None
    
    def load_model(self, pretrained: bool = True, weights_path: Optional[str] = None):
        """Загружает модель
        
        Args:
            pretrained: Использовать предобученные веса
            weights_path: Путь к файлу весов (.pt)
        """
        try:
            if weights_path and os.path.exists(weights_path):
                # Загружаем из файла
                self.model = YOLO(weights_path)
                logger.info(f"Модель загружена из {weights_path}")
            elif pretrained:
                # Загружаем предобученную модель
                self.model = YOLO(f"{self.model_type}.pt")
                logger.info(f"Предобученная модель {self.model_type} загружена")
            else:
                # Создаем модель с нуля
                self.model = YOLO(f"{self.model_type}.yaml")
                logger.info(f"Модель {self.model_type} создана с нуля")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}", exc_info=True)
            raise
    
    def train(self, dataset_path: str, epochs: int = 100, batch_size: int = 16,
              image_size: int = 640, learning_rate: float = 0.01,
              save_dir: Optional[str] = None, project_name: str = "yolo_training",
              callback: Optional[Callable[[str], None]] = None,
              epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
              **augmentation_params) -> Dict[str, Any]:
        """Обучает модель детекции
        
        Args:
            dataset_path: Путь к датасету (должен содержать dataset.yaml)
            epochs: Количество эпох
            batch_size: Размер батча
            image_size: Размер изображения
            learning_rate: Скорость обучения
            save_dir: Директория для сохранения результатов
            project_name: Имя проекта
            callback: Функция для логирования сообщений (принимает строку)
            **augmentation_params: Дополнительные параметры аугментации
        
        Returns:
            Словарь с результатами обучения
        """
        def log_message(msg: str, level: str = "info"):
            """Вспомогательная функция для логирования"""
            if level == "info":
                logger.info(msg)
            elif level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            
            if callback:
                try:
                    callback(msg)
                except Exception:
                    pass
        
        if self.model is None:
            raise ValueError("Модель не загружена. Вызовите load_model() сначала")
        
        log_message(f"Начало обучения модели детекции: {self.model_type}")
        log_message(f"Параметры обучения: epochs={epochs}, batch_size={batch_size}, image_size={image_size}, lr={learning_rate}")
        
        # Проверяем наличие dataset.yaml
        log_message(f"Поиск dataset.yaml в: {dataset_path}")
        dataset_yaml = os.path.join(dataset_path, "dataset.yaml")
        if not os.path.exists(dataset_yaml):
            # Пробуем найти dataset.yaml в корне датасета
            if os.path.isdir(dataset_path):
                # Если dataset_path - это директория, ищем dataset.yaml внутри
                for root, dirs, files in os.walk(dataset_path):
                    if "dataset.yaml" in files:
                        dataset_yaml = os.path.join(root, "dataset.yaml")
                        break
                else:
                    error_msg = f"dataset.yaml не найден в {dataset_path}. Убедитесь, что датасет имеет правильную структуру."
                    log_message(error_msg, "error")
                    raise FileNotFoundError(error_msg)
            else:
                error_msg = f"dataset.yaml не найден: {dataset_yaml}"
                log_message(error_msg, "error")
                raise FileNotFoundError(error_msg)
        
        log_message(f"Найден dataset.yaml: {dataset_yaml}")
        
        # Подготавливаем параметры обучения
        train_params = {
            'data': dataset_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': image_size,
            'lr0': learning_rate,
            'device': self.device,
            'project': save_dir or "runs/detect",
            'name': project_name,
            'exist_ok': True,  # Перезаписывать существующие результаты
            'workers': 0,  # Отключить multiprocessing для совместимости с QGIS
            'val': True,  # Включить валидацию во время обучения
            'resume': True,  # Возобновляет обучение с последней сохраненной контрольной точки
        }
        
        # Добавляем параметры аугментации, если они переданы
        if augmentation_params:
            # Стандартные параметры аугментации ultralytics
            if 'mosaic' in augmentation_params:
                train_params['mosaic'] = augmentation_params['mosaic']
            if 'mixup' in augmentation_params:
                train_params['mixup'] = augmentation_params['mixup']
            if 'copy_paste' in augmentation_params:
                train_params['copy_paste'] = augmentation_params['copy_paste']
            if 'fliplr' in augmentation_params:
                train_params['flipud'] = 0.0  # Отключаем вертикальное отражение
                train_params['fliplr'] = augmentation_params['fliplr']
        
        try:
            # Устанавливаем переменные окружения для предотвращения открытия окон и multiprocessing
            os.environ['MPLBACKEND'] = 'Agg'
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            # Дополнительные настройки для предотвращения multiprocessing
            os.environ['YOLO_VERBOSE'] = 'False'
            os.environ['NUMEXPR_MAX_THREADS'] = '1'
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            
            log_message(f"Устройство обучения: {self.device}")
            log_message(f"Директория сохранения: {save_dir or 'runs/detect'}/{project_name}")
            log_message("Запуск обучения...")
            
            # Добавляем callback для отслеживания эпох, если передан
            if epoch_callback:
                # Создаем callback для отслеживания прогресса обучения
                class EpochCallback:
                    def __init__(self, callback_func):
                        self.callback_func = callback_func
                        self.current_epoch = 0
                    
                    def on_train_epoch_end(self, trainer):
                        """Вызывается в конце каждой эпохи обучения"""
                        try:
                            self.current_epoch = trainer.epoch + 1  # Эпохи начинаются с 0
                        except Exception as e:
                            logger.warning(f"Ошибка в on_train_epoch_end: {e}")
                    
                    def on_val_end(self, trainer):
                        """Вызывается после валидации - здесь доступны метрики"""
                        try:
                            epoch = self.current_epoch
                            if epoch == 0:
                                epoch = trainer.epoch + 1
                            
                            metrics = {}
                            
                            # Извлекаем метрики из результатов валидации
                            if hasattr(trainer, 'metrics') and trainer.metrics:
                                metrics_dict = trainer.metrics
                                if isinstance(metrics_dict, dict):
                                    # Для детекции используем метрики с суффиксом (B)
                                    metrics['mAP50'] = float(metrics_dict.get('metrics/mAP50(B)', 0.0))
                                    metrics['mAP50-95'] = float(metrics_dict.get('metrics/mAP50-95(B)', 0.0))
                                    metrics['precision'] = float(metrics_dict.get('metrics/precision(B)', 0.0))
                                    metrics['recall'] = float(metrics_dict.get('metrics/recall(B)', 0.0))
                                    
                                    # Вычисляем F1-score
                                    precision = metrics['precision']
                                    recall = metrics['recall']
                                    if precision + recall > 0:
                                        metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
                                    else:
                                        metrics['f1_score'] = 0.0
                            
                            # Извлекаем loss метрики из trainer
                            if hasattr(trainer, 'loss_items'):
                                loss_items = trainer.loss_items
                                if isinstance(loss_items, dict):
                                    metrics['loss'] = float(loss_items.get('loss', 0.0))
                            elif hasattr(trainer, 'loss'):
                                metrics['loss'] = float(trainer.loss)
                            
                            # Если метрики пустые, пытаемся получить из других источников
                            if not metrics and hasattr(trainer, 'validator') and trainer.validator:
                                if hasattr(trainer.validator, 'metrics'):
                                    val_metrics = trainer.validator.metrics
                                    if isinstance(val_metrics, dict):
                                        metrics['mAP50'] = float(val_metrics.get('metrics/mAP50(B)', 0.0))
                                        metrics['mAP50-95'] = float(val_metrics.get('metrics/mAP50-95(B)', 0.0))
                                        metrics['precision'] = float(val_metrics.get('metrics/precision(B)', 0.0))
                                        metrics['recall'] = float(val_metrics.get('metrics/recall(B)', 0.0))
                                        
                                        # Вычисляем F1-score
                                        precision = metrics['precision']
                                        recall = metrics['recall']
                                        if precision + recall > 0:
                                            metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
                                        else:
                                            metrics['f1_score'] = 0.0
                            
                            # Вызываем callback только если есть метрики
                            if metrics:
                                self.callback_func(epoch, metrics)
                        except Exception as e:
                            logger.warning(f"Ошибка в callback валидации: {e}")
                
                # Добавляем callbacks к модели
                epoch_cb = EpochCallback(epoch_callback)
                self.model.add_callback("on_train_epoch_end", epoch_cb.on_train_epoch_end)
                self.model.add_callback("on_val_end", epoch_cb.on_val_end)
            
            # Запускаем обучение используя стандартный метод ultralytics
            results = self.model.train(**train_params)
            
            log_message("Обучение завершено успешно")
            
            # Формируем результат
            model_path = str(results.save_dir / 'weights' / 'best.pt')
            precision = float(results.results_dict.get('metrics/precision(B)', 0.0))
            recall = float(results.results_dict.get('metrics/recall(B)', 0.0))
            
            # Вычисляем F1-score
            f1_score = 0.0
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            
            result_dict = {
                'success': True,
                'model_path': model_path,
                'results_dir': str(results.save_dir),
                'metrics': {
                    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                }
            }
            
            log_message(f"Модель сохранена: {model_path}")
            log_message(f"Метрики: mAP50={result_dict['metrics']['mAP50']:.4f}, mAP50-95={result_dict['metrics']['mAP50-95']:.4f}")
            log_message(f"Precision={result_dict['metrics']['precision']:.4f}, Recall={result_dict['metrics']['recall']:.4f}, F1-score={result_dict['metrics']['f1_score']:.4f}")
            
            logger.info(f"Обучение завершено успешно. Модель сохранена: {model_path}")
            return result_dict
            
        except Exception as e:
            error_msg = f"Ошибка обучения: {e}"
            log_message(error_msg, "error")
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }


class DetectionDatasetAnalyzer:
    """Анализатор датасетов для детекции"""
    
    @staticmethod
    def analyze(dataset_path: str) -> Dict[str, Any]:
        """Анализирует датасет детекции
        
        Args:
            dataset_path: Путь к датасету
        
        Returns:
            Словарь с информацией о датасете
        """
        try:
            # Ищем dataset.yaml
            dataset_yaml = os.path.join(dataset_path, "dataset.yaml")
            if not os.path.exists(dataset_yaml):
                for root, dirs, files in os.walk(dataset_path):
                    if "dataset.yaml" in files:
                        dataset_yaml = os.path.join(root, "dataset.yaml")
                        break
                else:
                    return {'error': f"dataset.yaml не найден в {dataset_path}"}
            
            # Загружаем конфигурацию датасета
            try:
                import yaml
            except ImportError:
                # Если yaml не установлен, используем простой парсинг
                dataset_config = DetectionDatasetAnalyzer._parse_yaml_simple(dataset_yaml)
            else:
                with open(dataset_yaml, 'r', encoding='utf-8') as f:
                    dataset_config = yaml.safe_load(f)
            
            # Анализируем структуру датасета
            analysis = {
                'dataset_info': {
                    'path': dataset_path,
                    'yaml_path': dataset_yaml,
                    'nc': dataset_config.get('nc', 0),
                    'names': dataset_config.get('names', {})
                },
                'splits': {}
            }
            
            # Анализируем каждый сплит
            for split_name in ['train', 'val', 'test']:
                split_key = split_name
                if split_key in dataset_config:
                    split_path = dataset_config[split_key]
                    # Если путь относительный, делаем его абсолютным
                    if not os.path.isabs(split_path):
                        base_path = dataset_config.get('path', dataset_path)
                        split_path = os.path.join(base_path, split_path)
                    
                    split_analysis = DetectionDatasetAnalyzer._analyze_split(
                        split_path, split_name
                    )
                    analysis['splits'][split_name] = split_analysis
            
            # Подсчитываем общую статистику
            total_images = sum(
                s.get('image_count', 0) 
                for s in analysis['splits'].values() 
                if 'error' not in s
            )
            total_annotations = sum(
                s.get('annotation_count', 0) 
                for s in analysis['splits'].values() 
                if 'error' not in s
            )
            
            analysis['total_images'] = total_images
            analysis['total_annotations'] = total_annotations
            
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа датасета: {e}", exc_info=True)
            return {'error': str(e)}
    
    @staticmethod
    def _analyze_split(split_path: str, split_name: str) -> Dict[str, Any]:
        """Анализирует один сплит датасета"""
        try:
            if not os.path.exists(split_path):
                return {'error': f"Путь не существует: {split_path}"}
            
            # Подсчитываем изображения
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(split_path).glob(f"*{ext}"))
                image_files.extend(Path(split_path).glob(f"*{ext.upper()}"))
            
            image_count = len(image_files)
            
            # Подсчитываем аннотации (ищем соответствующие .txt файлы)
            # Предполагаем, что аннотации находятся в labels/ с тем же именем
            labels_path = split_path.replace('images', 'labels')
            if not os.path.exists(labels_path):
                # Пробуем найти labels рядом с images
                parent = os.path.dirname(split_path)
                labels_path = os.path.join(parent, 'labels', os.path.basename(split_path))
            
            annotation_count = 0
            total_objects = 0
            class_distribution = {}
            
            if os.path.exists(labels_path):
                txt_files = list(Path(labels_path).glob("*.txt"))
                annotation_count = len(txt_files)
                
                # Анализируем аннотации
                for txt_file in txt_files:
                    try:
                        with open(txt_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        total_objects += 1
                                        class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                    except Exception:
                        continue
            
            return {
                'image_count': image_count,
                'annotation_count': annotation_count,
                'total_objects': total_objects,
                'objects_per_image': total_objects / image_count if image_count > 0 else 0,
                'class_distribution': class_distribution
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа сплита {split_name}: {e}", exc_info=True)
            return {'error': str(e)}
    
    @staticmethod
    def _parse_yaml_simple(yaml_path: str) -> Dict:
        """Простой парсер YAML без зависимостей"""
        config = {}
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line and not line.startswith('#'):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        if key == 'names':
                            # Для names нужен специальный парсинг
                            config[key] = {}
                        else:
                            config[key] = value
        except Exception:
            pass
        return config

