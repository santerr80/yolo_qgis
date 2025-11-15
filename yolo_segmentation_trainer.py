# -*- coding: utf-8 -*-
"""
Специализированный модуль для обучения YOLO моделей сегментации объектов
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Fix for NumPy stderr issue in QGIS environment
from . import stderr_fix

# Настройка логирования
logger = logging.getLogger(__name__)

# Опциональные импорты
try:
    import yaml
except ImportError:
    yaml = None

try:
    import numpy as np
    # Проверяем совместимость версии numpy
    if hasattr(np, '__version__'):
        version_parts = np.__version__.split('.')
        major_version = int(version_parts[0])
        if major_version >= 2:
            logger.warning(f"numpy версии {np.__version__} может быть несовместима с QGIS. Рекомендуется numpy v1.x")
except ImportError:
    np = None

from .yolo_trainer import YOLOTrainer, TrainingProgress, ModelValidator, ModelPredictor


class SegmentationTrainer(YOLOTrainer):
    """Класс для обучения YOLO моделей сегментации объектов"""
    
    def __init__(self):
        super().__init__()
        self.task = 'segment'
        self.supported_models = [
            'yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg',
            'yolov11n-seg', 'yolov11s-seg', 'yolov11m-seg', 'yolov11l-seg', 'yolov11x-seg'
        ]
    
    def train_segmentation_model(self,
                               dataset_path: str,
                               model_type: str = 'yolov8n-seg',
                               epochs: int = 100,
                               batch_size: int = 16,
                               image_size: int = 640,
                               learning_rate: float = 0.01,
                               device: str = 'cpu',
                               pretrained: bool = True,
                               save_dir: str = None,
                               project_name: str = 'segmentation_training',
                               resume_training: bool = False,
                               # Специфичные для сегментации параметры
                               mask_ratio: int = 4,
                               overlap_mask: bool = True,
                               mask_ratio_scale: float = 1.0,
                               dropout: float = 0.0,
                               # Параметры аугментации для сегментации
                               mosaic: float = 1.0,
                               mixup: float = 0.0,
                               copy_paste: float = 0.3,  # Важно для сегментации
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
        Запускает обучение модели сегментации
        
        :param dataset_path: Путь к датасету
        :param model_type: Тип модели YOLO для сегментации
        :param epochs: Количество эпох
        :param batch_size: Размер батча
        :param image_size: Размер изображения
        :param learning_rate: Скорость обучения
        :param device: Устройство для обучения
        :param pretrained: Использовать предобученную модель
        :param save_dir: Директория для сохранения
        :param project_name: Имя проекта
        :param mask_ratio: Коэффициент масштабирования масок
        :param overlap_mask: Разрешить перекрытие масок
        :param mask_ratio_scale: Масштаб коэффициента масок
        :param dropout: Коэффициент dropout
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
        
        # Специфичные для сегментации параметры
        segmentation_params = {
            'mask_ratio': mask_ratio,
            'overlap_mask': overlap_mask,
            'mask_ratio_scale': mask_ratio_scale,
            'dropout': dropout,
            # Параметры аугментации
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
            # Веса потерь для сегментации
            'cls': 0.5,  # Коэффициент потерь классификации
            'box': 7.5,  # Коэффициент потерь боксов
            'dfl': 1.5,  # Коэффициент потерь DFL
            'seg': 1.0,  # Коэффициент потерь сегментации
        }
        
        # Объединяем с дополнительными параметрами
        segmentation_params.update(kwargs)
        
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
            resume_training=resume_training,
            **segmentation_params
        )
    
    def validate_segmentation_model(self,
                                  model_path: str,
                                  dataset_path: str,
                                  conf_threshold: float = 0.25,
                                  iou_threshold: float = 0.45,
                                  max_det: int = 300,
                                  save_results: bool = True) -> Dict:
        """
        Валидирует модель сегментации
        
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
            results_path = os.path.join(model_dir, 'segmentation_validation_results.json')
            validator.save_validation_results(results_path)
        
        return results
    
    def predict_segmentation(self,
                           model_path: str,
                           source: str,
                           conf_threshold: float = 0.25,
                           iou_threshold: float = 0.45,
                           max_det: int = 300,
                           save_results: bool = False,
                           output_dir: str = None,
                           save_masks: bool = True) -> List[Dict]:
        """
        Выполняет сегментацию объектов
        
        :param model_path: Путь к обученной модели
        :param source: Путь к изображению или директории
        :param conf_threshold: Порог уверенности
        :param iou_threshold: Порог IoU
        :param max_det: Максимальное количество детекций
        :param save_results: Сохранять ли результаты
        :param output_dir: Директория для сохранения
        :param save_masks: Сохранять ли маски
        :return: Список результатов сегментации
        """
        predictor = ModelPredictor(model_path)
        results = predictor.predict(
            source=source,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            save_results=save_results,
            output_dir=output_dir
        )
        
        # Дополнительная обработка для сегментации
        if save_masks and output_dir:
            self._save_segmentation_masks(results, output_dir)
        
        return results
    
    def _save_segmentation_masks(self, results: List[Dict], output_dir: str):
        """Сохраняет маски сегментации"""
        try:
            try:
                import cv2
                from PIL import Image
            except ImportError:
                logger.warning("cv2 или PIL не установлены. Маски не будут сохранены.")
                return
            
            masks_dir = os.path.join(output_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
            
            for result in results:
                if 'masks' in result and result['masks']:
                    image_name = os.path.splitext(os.path.basename(result['image_path']))[0]
                    
                    for i, mask_data in enumerate(result['masks']):
                        # Создаем маску из полигона
                        mask = self._create_mask_from_polygon(
                            mask_data['segmentation'], 
                            result.get('image_size', (640, 640))
                        )
                        
                        # Сохраняем маску
                        mask_path = os.path.join(masks_dir, f"{image_name}_mask_{i}.png")
                        cv2.imwrite(mask_path, mask * 255)
                        
        except Exception as e:
            logger.error(f"Ошибка сохранения масок: {e}", exc_info=True)
    
    def _create_mask_from_polygon(self, polygon: List[List[float]], image_size: Tuple[int, int]):
        """Создает маску из полигона"""
        try:
            try:
                import cv2
            except ImportError:
                logger.warning("cv2 не установлен. Маска не будет создана.")
                return None
            
            if np is None:
                return None
            mask = np.zeros(image_size, dtype=np.uint8)
            
            # Преобразуем нормализованные координаты в пиксели
            points = []
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    x = int(polygon[i] * image_size[1])
                    y = int(polygon[i + 1] * image_size[0])
                    points.append([x, y])
            
            if len(points) >= 3:
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
            
            return mask
            
        except Exception as e:
            logger.error(f"Ошибка создания маски: {e}", exc_info=True)
            if np is not None:
                return np.zeros(image_size, dtype=np.uint8)
            return None
    
    def get_segmentation_metrics(self, validation_results: Dict) -> Dict:
        """
        Извлекает метрики сегментации из результатов валидации
        
        :param validation_results: Результаты валидации
        :return: Словарь с метриками сегментации
        """
        if 'error' in validation_results:
            return {'error': validation_results['error']}
        
        metrics = validation_results.get('metrics', {})
        
        segmentation_metrics = {
            'mAP50': metrics.get('mAP50', 0.0),
            'mAP50-95': metrics.get('mAP50-95', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': 0.0,
            'mask_accuracy': 0.0,  # Требует дополнительных вычислений
            'iou_mean': 0.0,  # Средний IoU для масок
        }
        
        # Вычисляем F1-score
        precision = segmentation_metrics['precision']
        recall = segmentation_metrics['recall']
        if precision + recall > 0:
            segmentation_metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
        
        # Добавляем метрики по классам
        if 'class_metrics' in validation_results:
            segmentation_metrics['class_metrics'] = validation_results['class_metrics']
        
        return segmentation_metrics
    
    def create_segmentation_config(self,
                                 dataset_path: str,
                                 model_type: str = 'yolov8n-seg',
                                 epochs: int = 100,
                                 batch_size: int = 16,
                                 image_size: int = 640,
                                 learning_rate: float = 0.01,
                                 device: str = 'cpu',
                                 **kwargs) -> Dict:
        """
        Создает конфигурацию для обучения сегментации
        
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
            'segmentation_params': {
                'mask_ratio': kwargs.get('mask_ratio', 4),
                'overlap_mask': kwargs.get('overlap_mask', True),
                'mask_ratio_scale': kwargs.get('mask_ratio_scale', 1.0),
                'dropout': kwargs.get('dropout', 0.0)
            },
            'augmentation': {
                'mosaic': kwargs.get('mosaic', 1.0),
                'mixup': kwargs.get('mixup', 0.0),
                'copy_paste': kwargs.get('copy_paste', 0.3),
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
                'dfl': kwargs.get('dfl', 1.5),
                'seg': kwargs.get('seg', 1.0)
            }
        }
        
        return config
    
    def export_segmentation_model(self,
                                model_path: str,
                                export_format: str = 'onnx',
                                output_dir: str = None) -> str:
        """
        Экспортирует модель сегментации в различные форматы
        
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
            logger.error(f"Ошибка экспорта модели: {e}", exc_info=True)
            return None


class SegmentationDatasetAnalyzer:
    """Класс для анализа датасетов сегментации"""
    
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
            logger.error(f"Ошибка загрузки информации о датасете: {e}", exc_info=True)
            return {}
    
    def analyze_dataset(self) -> Dict:
        """
        Анализирует датасет сегментации
        
        :return: Словарь с анализом датасета
        """
        analysis = {
            'dataset_info': self.dataset_info,
            'splits': {},
            'class_distribution': {},
            'image_statistics': {},
            'annotation_statistics': {},
            'segmentation_statistics': {}
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
        
        # Анализируем классы и сегментации
        class_counts = {}
        total_objects = 0
        total_segments = 0
        segment_complexity = []
        
        for label_file in label_files:
            label_path = os.path.join(labels_path, label_file)
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 7:  # Минимум для сегментации: class_id x1 y1 x2 y2 ... (полигон)
                                class_id = int(parts[0])
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                total_objects += 1
                                
                                # Анализируем сложность сегментации
                                segment_points = (len(parts) - 5) // 2  # Количество точек полигона
                                segment_complexity.append(segment_points)
                                total_segments += 1
            except Exception as e:
                logger.error(f"Ошибка чтения файла {label_file}: {e}", exc_info=True)
        
        # Статистика сегментации
        segmentation_stats = {
            'total_segments': total_segments,
            'avg_segment_complexity': np.mean(segment_complexity) if segment_complexity and np is not None else 0,
            'max_segment_complexity': max(segment_complexity) if segment_complexity else 0,
            'min_segment_complexity': min(segment_complexity) if segment_complexity else 0
        }
        
        return {
            'image_count': len(image_files),
            'annotation_count': len(label_files),
            'total_objects': total_objects,
            'class_distribution': class_counts,
            'objects_per_image': total_objects / len(image_files) if image_files else 0,
            'segmentation_statistics': segmentation_stats
        }
    
    def get_class_names(self) -> Dict[int, str]:
        """Возвращает имена классов"""
        return self.dataset_info.get('names', {})
    
    def get_class_count(self) -> int:
        """Возвращает количество классов"""
        return self.dataset_info.get('nc', 0)
    
    def validate_dataset_structure(self) -> Tuple[bool, List[str]]:
        """
        Проверяет структуру датасета сегментации
        
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
        
        # Дополнительная проверка для сегментации
        self._validate_segmentation_annotations(errors)
        
        return len(errors) == 0, errors
    
    def _validate_segmentation_annotations(self, errors: List[str]):
        """Проверяет корректность аннотаций сегментации"""
        try:
            for split in ['train', 'val']:
                if split in self.dataset_info:
                    labels_path = os.path.join(self.dataset_path, 'labels', split)
                    if os.path.exists(labels_path):
                        label_files = [f for f in os.listdir(labels_path) 
                                      if f.lower().endswith('.txt')]
                        
                        for label_file in label_files[:10]:  # Проверяем первые 10 файлов
                            label_path = os.path.join(labels_path, label_file)
                            with open(label_path, 'r') as f:
                                for line_num, line in enumerate(f, 1):
                                    if line.strip():
                                        parts = line.strip().split()
                                        if len(parts) < 7:  # Минимум для сегментации
                                            errors.append(f"Недостаточно точек в сегментации: {label_file}:{line_num}")
                                        elif len(parts) % 2 != 1:  # Должно быть нечетное количество (class_id + пары координат)
                                            errors.append(f"Некорректное количество координат: {label_file}:{line_num}")
        except Exception as e:
            errors.append(f"Ошибка проверки аннотаций сегментации: {e}")
    
    def calculate_segmentation_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Вычисляет метрики сегментации
        
        :param predictions: Предсказания модели
        :param ground_truth: Истинные значения
        :return: Словарь с метриками
        """
        try:
            try:
                import cv2
            except ImportError:
                return {'error': 'cv2 не установлен'}
            
            metrics = {
                'iou_mean': 0.0,
                'dice_coefficient': 0.0,
                'pixel_accuracy': 0.0,
                'mean_accuracy': 0.0
            }
            
            total_iou = 0.0
            total_dice = 0.0
            total_pixel_acc = 0.0
            total_mean_acc = 0.0
            valid_samples = 0
            
            for pred, gt in zip(predictions, ground_truth):
                if 'masks' in pred and 'masks' in gt:
                    for pred_mask, gt_mask in zip(pred['masks'], gt['masks']):
                        # Вычисляем IoU
                        iou = self._calculate_iou(pred_mask, gt_mask)
                        total_iou += iou
                        
                        # Вычисляем Dice coefficient
                        dice = self._calculate_dice(pred_mask, gt_mask)
                        total_dice += dice
                        
                        # Вычисляем pixel accuracy
                        pixel_acc = self._calculate_pixel_accuracy(pred_mask, gt_mask)
                        total_pixel_acc += pixel_acc
                        
                        valid_samples += 1
            
            if valid_samples > 0:
                metrics['iou_mean'] = total_iou / valid_samples
                metrics['dice_coefficient'] = total_dice / valid_samples
                metrics['pixel_accuracy'] = total_pixel_acc / valid_samples
                metrics['mean_accuracy'] = total_mean_acc / valid_samples
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка вычисления метрик сегментации: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _calculate_iou(self, mask1, mask2) -> float:
        """Вычисляет Intersection over Union для масок"""
        try:
            if np is None:
                return 0.0
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_dice(self, mask1, mask2) -> float:
        """Вычисляет Dice coefficient для масок"""
        try:
            if np is None:
                return 0.0
            intersection = np.logical_and(mask1, mask2).sum()
            return 2 * intersection / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_pixel_accuracy(self, mask1, mask2) -> float:
        """Вычисляет pixel accuracy для масок"""
        try:
            if np is None:
                return 0.0
            correct_pixels = np.equal(mask1, mask2).sum()
            total_pixels = mask1.size
            return correct_pixels / total_pixels if total_pixels > 0 else 0.0
        except Exception:
            return 0.0
