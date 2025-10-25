# -*- coding: utf-8 -*-
"""
Специализированный модуль для обучения YOLO моделей детекции объектов
"""

import os
import json
import yaml
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .yolo_trainer import YOLOTrainer, TrainingProgress, ModelValidator, ModelPredictor


class DetectionTrainer(YOLOTrainer):
    """Класс для обучения YOLO моделей детекции объектов"""
    
    def __init__(self):
        super().__init__()
        self.task = 'detect'
        self.supported_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    
    def train_detection_model(self,
                            dataset_path: str,
                            model_type: str = 'yolov8n',
                            epochs: int = 100,
                            batch_size: int = 16,
                            image_size: int = 640,
                            learning_rate: float = 0.01,
                            device: str = 'cpu',
                            pretrained: bool = True,
                            save_dir: str = None,
                            project_name: str = 'detection_training',
                            # Специфичные для детекции параметры
                            mosaic: float = 1.0,
                            mixup: float = 0.0,
                            copy_paste: float = 0.0,
                            degrees: float = 0.0,
                            translate: float = 0.1,
                            scale: float = 0.5,
                            shear: float = 0.0,
                            perspective: float = 0.0,
                            flipud: float = 0.0,
                            fliplr: float = 0.5,
                            hsv_h: float = 0.015,
                            hsv_s: float = 0.7,
                            hsv_v: float = 0.4,
                            **kwargs) -> bool:
        """
        Запускает обучение модели детекции
        
        :param dataset_path: Путь к датасету
        :param model_type: Тип модели YOLO
        :param epochs: Количество эпох
        :param batch_size: Размер батча
        :param image_size: Размер изображения
        :param learning_rate: Скорость обучения
        :param device: Устройство для обучения
        :param pretrained: Использовать предобученную модель
        :param save_dir: Директория для сохранения
        :param project_name: Имя проекта
        :param mosaic: Вероятность применения мозаичной аугментации
        :param mixup: Вероятность применения mixup аугментации
        :param copy_paste: Вероятность применения copy-paste аугментации
        :param degrees: Диапазон поворота в градусах
        :param translate: Диапазон смещения
        :param scale: Диапазон масштабирования
        :param shear: Диапазон сдвига
        :param perspective: Вероятность применения перспективной трансформации
        :param flipud: Вероятность вертикального отражения
        :param fliplr: Вероятность горизонтального отражения
        :param hsv_h: Диапазон изменения оттенка HSV
        :param hsv_s: Диапазон изменения насыщенности HSV
        :param hsv_v: Диапазон изменения яркости HSV
        :param kwargs: Дополнительные параметры
        :return: True если обучение запущено успешно
        """
        
        # Проверяем тип модели
        if model_type not in self.supported_models:
            self.progress.training_finished.emit(False, f"Неподдерживаемый тип модели: {model_type}")
            return False
        
        # Специфичные для детекции параметры аугментации
        detection_params = {
            'mosaic': mosaic,
            'mixup': mixup,
            'copy_paste': copy_paste,
            'degrees': degrees,
            'translate': translate,
            'scale': scale,
            'shear': shear,
            'perspective': perspective,
            'flipud': flipud,
            'fliplr': fliplr,
            'hsv_h': hsv_h,
            'hsv_s': hsv_s,
            'hsv_v': hsv_v,
            'cls': 0.5,  # Коэффициент потерь классификации
            'box': 7.5,  # Коэффициент потерь боксов
            'dfl': 1.5,  # Коэффициент потерь DFL
        }
        
        # Объединяем с дополнительными параметрами
        detection_params.update(kwargs)
        
        return self.train_model(
            dataset_path=dataset_path,
            model_type=model_type,
            task=self.task,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            device=device,
            pretrained=pretrained,
            save_dir=save_dir,
            project_name=project_name,
            **detection_params
        )
    
    def validate_detection_model(self,
                               model_path: str,
                               dataset_path: str,
                               conf_threshold: float = 0.25,
                               iou_threshold: float = 0.45,
                               max_det: int = 300,
                               save_results: bool = True) -> Dict:
        """
        Валидирует модель детекции
        
        :param model_path: Путь к обученной модели
        :param dataset_path: Путь к датасету
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU
        :param max_det: Максимальное количество детекций
        :param save_results: Сохранять ли результаты валидации
        :return: Результаты валидации
        """
        validator = ModelValidator()
        results = validator.validate_model(
            model_path=model_path,
            dataset_path=dataset_path,
            task=self.task,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det
        )
        
        if save_results and 'error' not in results:
            # Сохраняем результаты валидации
            model_dir = os.path.dirname(model_path)
            results_path = os.path.join(model_dir, 'validation_results.json')
            validator.save_validation_results(results_path)
        
        return results
    
    def predict_detection(self,
                         model_path: str,
                         source: str,
                         conf_threshold: float = 0.25,
                         iou_threshold: float = 0.45,
                         max_det: int = 300,
                         save_results: bool = False,
                         output_dir: str = None) -> List[Dict]:
        """
        Выполняет детекцию объектов
        
        :param model_path: Путь к обученной модели
        :param source: Путь к изображению или директории
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU
        :param max_det: Максимальное количество детекций
        :param save_results: Сохранять ли результаты
        :param output_dir: Директория для сохранения
        :return: Список результатов детекции
        """
        predictor = ModelPredictor(model_path)
        return predictor.predict(
            source=source,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            save_results=save_results,
            output_dir=output_dir
        )
    
    def get_detection_metrics(self, validation_results: Dict) -> Dict:
        """
        Извлекает метрики детекции из результатов валидации
        
        :param validation_results: Результаты валидации
        :return: Словарь с метриками детекции
        """
        if 'error' in validation_results:
            return {'error': validation_results['error']}
        
        metrics = validation_results.get('metrics', {})
        
        detection_metrics = {
            'mAP50': metrics.get('mAP50', 0.0),
            'mAP50-95': metrics.get('mAP50-95', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': 0.0
        }
        
        # Вычисляем F1-score
        precision = detection_metrics['precision']
        recall = detection_metrics['recall']
        if precision + recall > 0:
            detection_metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
        
        # Добавляем метрики по классам
        if 'class_metrics' in validation_results:
            detection_metrics['class_metrics'] = validation_results['class_metrics']
        
        return detection_metrics
    
    def create_detection_config(self,
                              dataset_path: str,
                              model_type: str = 'yolov8n',
                              epochs: int = 100,
                              batch_size: int = 16,
                              image_size: int = 640,
                              learning_rate: float = 0.01,
                              device: str = 'cpu',
                              **kwargs) -> Dict:
        """
        Создает конфигурацию для обучения детекции
        
        :param dataset_path: Путь к датасету
        :param model_type: Тип модели
        :param epochs: Количество эпох
        :param batch_size: Размер батча
        :param image_size: Размер изображения
        :param learning_rate: Скорость обучения
        :param device: Устройство
        :param kwargs: Дополнительные параметры
        :return: Конфигурация обучения
        """
        config = {
            'task': self.task,
            'dataset_path': dataset_path,
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'image_size': image_size,
            'learning_rate': learning_rate,
            'device': device,
            'pretrained': True,
            'augmentation': {
                'mosaic': kwargs.get('mosaic', 1.0),
                'mixup': kwargs.get('mixup', 0.0),
                'copy_paste': kwargs.get('copy_paste', 0.0),
                'degrees': kwargs.get('degrees', 0.0),
                'translate': kwargs.get('translate', 0.1),
                'scale': kwargs.get('scale', 0.5),
                'shear': kwargs.get('shear', 0.0),
                'perspective': kwargs.get('perspective', 0.0),
                'flipud': kwargs.get('flipud', 0.0),
                'fliplr': kwargs.get('fliplr', 0.5),
                'hsv_h': kwargs.get('hsv_h', 0.015),
                'hsv_s': kwargs.get('hsv_s', 0.7),
                'hsv_v': kwargs.get('hsv_v', 0.4)
            },
            'loss_weights': {
                'cls': kwargs.get('cls', 0.5),
                'box': kwargs.get('box', 7.5),
                'dfl': kwargs.get('dfl', 1.5)
            }
        }
        
        return config
    
    def export_detection_model(self,
                             model_path: str,
                             export_format: str = 'onnx',
                             output_dir: str = None) -> str:
        """
        Экспортирует модель детекции в различные форматы
        
        :param model_path: Путь к обученной модели
        :param export_format: Формат экспорта ('onnx', 'torchscript', 'tflite', 'pb')
        :param output_dir: Директория для сохранения
        :return: Путь к экспортированной модели
        """
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            if output_dir is None:
                output_dir = os.path.dirname(model_path)
            
            # Экспортируем модель
            exported_model = model.export(
                format=export_format,
                imgsz=640,
                optimize=True,
                half=False,
                int8=False,
                dynamic=False,
                simplify=True,
                opset=None,
                workspace=4,
                nms=False
            )
            
            # Перемещаем в нужную директорию если необходимо
            if output_dir != os.path.dirname(exported_model):
                new_path = os.path.join(output_dir, os.path.basename(exported_model))
                os.rename(exported_model, new_path)
                exported_model = new_path
            
            return exported_model
            
        except Exception as e:
            print(f"Ошибка экспорта модели: {e}")
            return None


class DetectionDatasetAnalyzer:
    """Класс для анализа датасетов детекции"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_info = self._load_dataset_info()
    
    def _load_dataset_info(self) -> Dict:
        """Загружает информацию о датасете"""
        try:
            yaml_path = os.path.join(self.dataset_path, 'dataset.yaml')
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            return {
                'path': config.get('path', ''),
                'train': config.get('train', ''),
                'val': config.get('val', ''),
                'test': config.get('test', ''),
                'names': config.get('names', {}),
                'nc': config.get('nc', len(config.get('names', {})))
            }
        except Exception as e:
            print(f"Ошибка загрузки информации о датасете: {e}")
            return {}
    
    def analyze_dataset(self) -> Dict:
        """
        Анализирует датасет детекции
        
        :return: Словарь с анализом датасета
        """
        analysis = {
            'dataset_info': self.dataset_info,
            'splits': {},
            'class_distribution': {},
            'image_statistics': {},
            'annotation_statistics': {}
        }
        
        # Анализируем каждый сплит
        for split in ['train', 'val', 'test']:
            if split in self.dataset_info:
                split_analysis = self._analyze_split(split)
                analysis['splits'][split] = split_analysis
        
        # Общая статистика
        analysis['total_images'] = sum(
            split_data.get('image_count', 0) 
            for split_data in analysis['splits'].values()
        )
        analysis['total_annotations'] = sum(
            split_data.get('annotation_count', 0) 
            for split_data in analysis['splits'].values()
        )
        
        return analysis
    
    def _analyze_split(self, split: str) -> Dict:
        """Анализирует конкретный сплит датасета"""
        split_path = os.path.join(self.dataset_path, self.dataset_info[split])
        labels_path = os.path.join(self.dataset_path, 'labels', split)
        
        if not os.path.exists(split_path) or not os.path.exists(labels_path):
            return {'error': f'Директории для сплита {split} не найдены'}
        
        # Подсчитываем изображения
        image_files = [f for f in os.listdir(split_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        # Подсчитываем аннотации
        label_files = [f for f in os.listdir(labels_path) 
                      if f.lower().endswith('.txt')]
        
        # Анализируем классы
        class_counts = {}
        total_objects = 0
        
        for label_file in label_files:
            label_path = os.path.join(labels_path, label_file)
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 5:  # Минимум для детекции: class_id x y w h
                                class_id = int(parts[0])
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                total_objects += 1
            except Exception as e:
                print(f"Ошибка чтения файла {label_file}: {e}")
        
        return {
            'image_count': len(image_files),
            'annotation_count': len(label_files),
            'total_objects': total_objects,
            'class_distribution': class_counts,
            'objects_per_image': total_objects / len(image_files) if image_files else 0
        }
    
    def get_class_names(self) -> Dict[int, str]:
        """Возвращает имена классов"""
        return self.dataset_info.get('names', {})
    
    def get_class_count(self) -> int:
        """Возвращает количество классов"""
        return self.dataset_info.get('nc', 0)
    
    def validate_dataset_structure(self) -> Tuple[bool, List[str]]:
        """
        Проверяет структуру датасета
        
        :return: (валидность, список ошибок)
        """
        errors = []
        
        # Проверяем наличие dataset.yaml
        yaml_path = os.path.join(self.dataset_path, 'dataset.yaml')
        if not os.path.exists(yaml_path):
            errors.append("Отсутствует файл dataset.yaml")
        
        # Проверяем наличие директорий
        required_dirs = ['images', 'labels']
        for dir_name in required_dirs:
            dir_path = os.path.join(self.dataset_path, dir_name)
            if not os.path.exists(dir_path):
                errors.append(f"Отсутствует директория {dir_name}")
        
        # Проверяем сплиты
        for split in ['train', 'val']:
            if split in self.dataset_info:
                split_path = os.path.join(self.dataset_path, self.dataset_info[split])
                if not os.path.exists(split_path):
                    errors.append(f"Отсутствует директория для сплита {split}")
        
        return len(errors) == 0, errors

