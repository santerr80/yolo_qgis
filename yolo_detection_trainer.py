# -*- coding: utf-8 -*-
"""
Модуль для обучения моделей YOLO детекции объектов
Поддерживает YOLOv8 и YOLOv11
"""

import os
import logging
import uuid
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class DetectionDatasetAnalyzer:
    """Анализатор датасета для детекции"""
    
    @staticmethod
    def analyze(dataset_path: str) -> Dict[str, Any]:
        """
        Анализирует датасет для детекции
        
        :param dataset_path: Путь к датасету
        :return: Словарь с информацией о датасете
        """
        try:
            result = {
                'dataset_info': {},
                'splits': {},
                'total_images': 0,
                'total_annotations': 0,
                'error': None
            }
            
            # Проверяем формат датасета
            yaml_path = os.path.join(dataset_path, 'dataset.yaml')
            ndjson_path = os.path.join(dataset_path, 'data.ndjson')
            
            if os.path.exists(yaml_path):
                # Нативный YOLO формат
                result = DetectionDatasetAnalyzer._analyze_yolo_format(dataset_path, yaml_path)
            elif os.path.exists(ndjson_path):
                # NDJSON формат
                result = DetectionDatasetAnalyzer._analyze_ndjson_format(dataset_path, ndjson_path)
            else:
                result['error'] = "Не найден dataset.yaml или data.ndjson"
                return result
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа датасета: {e}", exc_info=True)
            return {'error': str(e)}
    
    @staticmethod
    def _analyze_yolo_format(dataset_path: str, yaml_path: str) -> Dict[str, Any]:
        """Анализирует датасет в нативном YOLO формате"""
        import yaml
        
        result = {
            'dataset_info': {},
            'splits': {},
            'total_images': 0,
            'total_annotations': 0
        }
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            result['dataset_info'] = {
                'path': dataset_path,
                'nc': yaml_data.get('nc', 0),
                'names': yaml_data.get('names', {})
            }
            
            # Анализируем каждый сплит
            images_dir = os.path.join(dataset_path, 'images')
            labels_dir = os.path.join(dataset_path, 'labels')
            
            for split in ['train', 'val', 'test']:
                split_images_dir = os.path.join(images_dir, split)
                split_labels_dir = os.path.join(labels_dir, split)
                
                if not os.path.exists(split_images_dir):
                    continue
                
                image_files = [f for f in os.listdir(split_images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                annotation_count = 0
                total_objects = 0
                class_distribution = {}
                
                for img_file in image_files:
                    # Ищем соответствующий файл аннотаций
                    base_name = os.path.splitext(img_file)[0]
                    label_file = os.path.join(split_labels_dir, f"{base_name}.txt")
                    
                    if os.path.exists(label_file):
                        annotation_count += 1
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            total_objects += len(lines)
                            for line in lines:
                                parts = line.strip().split()
                                if parts:
                                    class_id = int(parts[0])
                                    class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                
                result['splits'][split] = {
                    'image_count': len(image_files),
                    'annotation_count': annotation_count,
                    'total_objects': total_objects,
                    'objects_per_image': total_objects / len(image_files) if image_files else 0,
                    'class_distribution': class_distribution
                }
                result['total_images'] += len(image_files)
                result['total_annotations'] += annotation_count
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа YOLO формата: {e}", exc_info=True)
            return {'error': str(e)}
    
    @staticmethod
    def _analyze_ndjson_format(dataset_path: str, ndjson_path: str) -> Dict[str, Any]:
        """Анализирует датасет в NDJSON формате"""
        import json
        
        result = {
            'dataset_info': {},
            'splits': {},
            'total_images': 0,
            'total_annotations': 0
        }
        
        try:
            splits_data = {'train': [], 'val': [], 'test': []}
            all_classes = set()
            
            with open(ndjson_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get('type') == 'image':
                        split = record.get('split', 'train')
                        if split in splits_data:
                            splits_data[split].append(record)
                            
                            # Собираем классы из аннотаций
                            for ann in record.get('annotations', []):
                                if 'class' in ann:
                                    all_classes.add(ann['class'])
            
            # Формируем информацию о датасете
            class_names = {i: name for i, name in enumerate(sorted(all_classes))}
            result['dataset_info'] = {
                'path': dataset_path,
                'nc': len(all_classes),
                'names': class_names
            }
            
            # Анализируем каждый сплит
            for split, records in splits_data.items():
                if not records:
                    continue
                
                total_objects = 0
                class_distribution = {}
                
                for record in records:
                    annotations = record.get('annotations', [])
                    total_objects += len(annotations)
                    for ann in annotations:
                        class_name = ann.get('class', '')
                        class_id = list(class_names.keys())[list(class_names.values()).index(class_name)] if class_name in class_names.values() else 0
                        class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                
                result['splits'][split] = {
                    'image_count': len(records),
                    'annotation_count': len([r for r in records if r.get('annotations')]),
                    'total_objects': total_objects,
                    'objects_per_image': total_objects / len(records) if records else 0,
                    'class_distribution': class_distribution
                }
                result['total_images'] += len(records)
                result['total_annotations'] += len([r for r in records if r.get('annotations')])
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа NDJSON формата: {e}", exc_info=True)
            return {'error': str(e)}


class DetectionTrainer:
    """Тренер для моделей детекции YOLO"""
    
    def __init__(self, metrics_tracker=None):
        """
        Инициализация тренера
        
        :param metrics_tracker: Трекер метрик (опционально)
        """
        self.metrics_tracker = metrics_tracker
        self.current_model = None
        self.training_thread = None
        self.is_training = False
    
    def train(self, dataset_path: str, model_type: str = 'yolov8n',
              epochs: int = 100, batch_size: int = 16, image_size: int = 640,
              learning_rate: float = 0.01, device: str = 'cpu',
              pretrained: bool = True, save_dir: str = None,
              project_name: str = 'yolo_training',
              mosaic: float = 1.0, mixup: float = 0.0,
              copy_paste: float = 0.3, fliplr: float = 0.5,
              progress_callback=None, status_callback=None) -> Dict[str, Any]:
        """
        Запускает обучение модели детекции
        
        :param dataset_path: Путь к датасету
        :param model_type: Тип модели (yolov8n, yolov8s, yolov11n и т.д.)
        :param epochs: Количество эпох
        :param batch_size: Размер батча
        :param image_size: Размер изображения
        :param learning_rate: Скорость обучения
        :param device: Устройство (cpu/0/1 и т.д.)
        :param pretrained: Использовать предобученные веса
        :param save_dir: Директория для сохранения
        :param project_name: Название проекта
        :param mosaic: Вероятность мозаики
        :param mixup: Вероятность mixup
        :param copy_paste: Вероятность copy-paste
        :param fliplr: Вероятность горизонтального отражения
        :param progress_callback: Callback для прогресса
        :param status_callback: Callback для статуса
        :return: Словарь с результатами обучения
        """
        try:
            if self.is_training:
                return {'error': 'Обучение уже выполняется'}
            
            self.is_training = True
            
            # Проверяем наличие ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                return {'error': 'Библиотека ultralytics не установлена. Установите: pip install ultralytics'}
            
            # Определяем путь к dataset.yaml
            yaml_path = os.path.join(dataset_path, 'dataset.yaml')
            if not os.path.exists(yaml_path):
                # Пробуем создать dataset.yaml из NDJSON
                yaml_path = self._create_yaml_from_ndjson(dataset_path)
                if not yaml_path:
                    return {'error': 'Не найден dataset.yaml и не удалось создать из NDJSON'}
            
            # Создаем модель
            if status_callback:
                status_callback("Загрузка модели...")
            
            model_name = model_type
            if pretrained:
                # Загружаем предобученную модель
                self.current_model = YOLO(f"{model_name}.pt")
            else:
                # Создаем модель с нуля
                self.current_model = YOLO(f"{model_name}.yaml")
            
            # Настраиваем параметры обучения
            train_args = {
                'data': yaml_path,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': image_size,
                'lr0': learning_rate,
                'device': device,
                'project': save_dir or 'runs',
                'name': project_name,
                'save': True,
                'save_period': 10,  # Сохранять чекпоинты каждые 10 эпох
                'plots': True,  # Графики сохраняются в файлы, не открываются
                'augment': True,
                'mosaic': mosaic,
                'mixup': mixup,
                'copy_paste': copy_paste,
                'flipud': 0.0,  # Вертикальное отражение отключено по умолчанию
                'fliplr': fliplr,
            }
            
            # Создаем experiment_id для трекинга
            experiment_id = str(uuid.uuid4())
            
            if self.metrics_tracker:
                config = {
                    'model_type': model_type,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'image_size': image_size,
                    'learning_rate': learning_rate,
                    'device': device,
                    'pretrained': pretrained,
                    'augmentation': {
                        'mosaic': mosaic,
                        'mixup': mixup,
                        'copy_paste': copy_paste,
                        'fliplr': fliplr
                    }
                }
                self.metrics_tracker.start_experiment(
                    experiment_id=experiment_id,
                    name=project_name,
                    task='detect',
                    model_type=model_type,
                    dataset_path=dataset_path,
                    config=config
                )
                self.metrics_tracker.current_experiment_id = experiment_id
            
            if status_callback:
                status_callback("Начало обучения...")
            
            # Создаем кастомный callback для обновления прогресса
            if progress_callback:
                def on_fit_epoch_end(trainer):
                    """Callback вызывается в конце каждой эпохи"""
                    try:
                        epoch = trainer.epoch + 1  # Номер эпохи (начинается с 1)
                        
                        # Извлекаем метрики из trainer
                        metrics = {}
                        if hasattr(trainer, 'metrics') and trainer.metrics:
                            metrics_dict = trainer.metrics
                            # Проверяем, является ли это словарем или объектом
                            if isinstance(metrics_dict, dict):
                                # Если это словарь, извлекаем метрики напрямую
                                for key, value in metrics_dict.items():
                                    try:
                                        if isinstance(value, (int, float)):
                                            metrics[key] = float(value)
                                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                                            metrics[key] = float(value[0])
                                    except (ValueError, TypeError, IndexError):
                                        pass
                            else:
                                # Если это объект, используем getattr для основных метрик
                                for attr_name in ['loss', 'box_loss', 'cls_loss', 'dfl_loss']:
                                    try:
                                        if hasattr(metrics_dict, attr_name):
                                            value = getattr(metrics_dict, attr_name)
                                            if value is not None:
                                                metrics[attr_name] = float(value)
                                    except (ValueError, TypeError, AttributeError):
                                        pass
                        
                        # Также пытаемся получить метрики из результатов валидации
                        if hasattr(trainer, 'validator') and trainer.validator:
                            if hasattr(trainer.validator, 'metrics'):
                                val_metrics = trainer.validator.metrics
                                if val_metrics:
                                    # DetMetrics - это объект, а не словарь, используем getattr
                                    metric_names = {
                                        'map50': 'mAP50',
                                        'map': 'mAP50-95',
                                        'precision': 'precision',
                                        'recall': 'recall'
                                    }
                                    for attr_name, metric_key in metric_names.items():
                                        try:
                                            if hasattr(val_metrics, attr_name):
                                                value = getattr(val_metrics, attr_name)
                                                if value is not None:
                                                    metrics[metric_key] = float(value)
                                        except (ValueError, TypeError, AttributeError):
                                            pass
                        
                        # Вызываем callback
                        if progress_callback:
                            progress_callback(epoch, metrics)
                    except Exception as e:
                        # Используем print вместо logger, чтобы избежать проблем с логированием
                        try:
                            logger.error(f"Ошибка в callback прогресса: {e}", exc_info=False)
                        except:
                            print(f"Ошибка в callback прогресса: {e}")
                
                # Добавляем callback к модели через add_callback
                try:
                    self.current_model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
                except AttributeError:
                    # Если метод add_callback не существует, пробуем через параметр callbacks
                    if 'callbacks' not in train_args:
                        train_args['callbacks'] = {}
                    train_args['callbacks']['on_fit_epoch_end'] = on_fit_epoch_end
            
            # Запускаем обучение
            results = self.current_model.train(**train_args)
            
            # Извлекаем метрики из результатов
            final_metrics = {}
            if hasattr(results, 'results_dict'):
                final_metrics = results.results_dict
            
            # Логируем финальные метрики
            if self.metrics_tracker:
                self.metrics_tracker.complete_experiment(
                    status='completed',
                    final_metrics=final_metrics
                )
            
            self.is_training = False
            
            return {
                'success': True,
                'experiment_id': experiment_id,
                'model_path': str(results.save_dir / 'weights' / 'best.pt') if hasattr(results, 'save_dir') else None,
                'metrics': final_metrics,
                'message': 'Обучение завершено успешно'
            }
            
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}", exc_info=True)
            self.is_training = False
            
            if self.metrics_tracker:
                self.metrics_tracker.complete_experiment(status='failed')
            
            return {'error': str(e)}
    
    def _create_yaml_from_ndjson(self, dataset_path: str) -> Optional[str]:
        """
        Создает dataset.yaml из NDJSON формата
        
        :param dataset_path: Путь к датасету
        :return: Путь к созданному YAML файлу или None
        """
        try:
            import json
            import yaml
            
            ndjson_path = os.path.join(dataset_path, 'data.ndjson')
            if not os.path.exists(ndjson_path):
                return None
            
            # Собираем классы из NDJSON
            class_names = set()
            with open(ndjson_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get('type') == 'image':
                        for ann in record.get('annotations', []):
                            if 'class' in ann:
                                class_names.add(ann['class'])
            
            # Создаем dataset.yaml
            yaml_path = os.path.join(dataset_path, 'dataset.yaml')
            yaml_data = {
                'path': os.path.abspath(dataset_path),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test' if os.path.exists(os.path.join(dataset_path, 'images', 'test')) else None,
                'nc': len(class_names),
                'names': {i: name for i, name in enumerate(sorted(class_names))}
            }
            
            # Удаляем None значения
            yaml_data = {k: v for k, v in yaml_data.items() if v is not None}
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
            
            return yaml_path
            
        except Exception as e:
            logger.error(f"Ошибка создания YAML из NDJSON: {e}", exc_info=True)
            return None
    
    def cancel_training(self) -> bool:
        """
        Отменяет текущее обучение
        
        :return: True если успешно
        """
        try:
            if self.current_model and self.is_training:
                # Ultralytics не поддерживает прямую отмену, но можно попробовать
                self.is_training = False
                if self.metrics_tracker:
                    self.metrics_tracker.complete_experiment(status='cancelled')
                return True
            return False
        except Exception as e:
            logger.error(f"Ошибка отмены обучения: {e}", exc_info=True)
            return False

