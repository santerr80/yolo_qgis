# -*- coding: utf-8 -*-
"""
Модуль валидации YOLO моделей
Использует стандартные методы ultralytics
"""

import os
import logging
from typing import Dict, List, Optional, Any

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


class AdvancedValidator:
    """Валидатор моделей YOLO"""
    
    def __init__(self, device: str = "cpu"):
        """Инициализация валидатора
        
        Args:
            device: Устройство для валидации ('cpu' или '0' для GPU)
        """
        if YOLO is None:
            raise ImportError("ultralytics не установлен. Установите: pip install ultralytics")
        
        self.device = device
    
    def validate(self, model_path: str, dataset_path: str, task: str = "detect",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 max_detections: int = 300, comprehensive: bool = False) -> Dict[str, Any]:
        """Валидирует модель
        
        Args:
            model_path: Путь к модели (.pt)
            dataset_path: Путь к датасету
            task: Тип задачи ('detect' или 'segment')
            conf_threshold: Порог уверенности
            iou_threshold: Порог IoU для NMS
            max_detections: Максимальное количество детекций
            comprehensive: Выполнить комплексную валидацию
        
        Returns:
            Словарь с результатами валидации
        """
        try:
            # Загружаем модель
            if not os.path.exists(model_path):
                return {'error': f"Модель не найдена: {model_path}"}
            
            model = YOLO(model_path)
            
            # Проверяем наличие dataset.yaml
            dataset_yaml = os.path.join(dataset_path, "dataset.yaml")
            if not os.path.exists(dataset_yaml):
                for root, dirs, files in os.walk(dataset_path):
                    if "dataset.yaml" in files:
                        dataset_yaml = os.path.join(root, "dataset.yaml")
                        break
                else:
                    return {'error': f"dataset.yaml не найден в {dataset_path}"}
            
            # Устанавливаем переменные окружения для предотвращения открытия окон и multiprocessing
            os.environ['MPLBACKEND'] = 'Agg'
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            # Дополнительные настройки для предотвращения multiprocessing
            os.environ['YOLO_VERBOSE'] = 'False'
            os.environ['NUMEXPR_MAX_THREADS'] = '1'
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            
            # Выполняем валидацию используя стандартный метод ultralytics
            results = model.val(
                data=dataset_yaml,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                device=self.device,
                workers=0  # Отключить multiprocessing для совместимости с QGIS
            )
            
            # Извлекаем метрики в зависимости от типа задачи
            if task == "segment":
                precision = float(results.results_dict.get('metrics/precision(M)', 0.0))
                recall = float(results.results_dict.get('metrics/recall(M)', 0.0))
                metrics = {
                    'mAP50': float(results.results_dict.get('metrics/mAP50(M)', 0.0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(M)', 0.0)),
                    'precision': precision,
                    'recall': recall,
                }
            else:  # detect
                precision = float(results.results_dict.get('metrics/precision(B)', 0.0))
                recall = float(results.results_dict.get('metrics/recall(B)', 0.0))
                metrics = {
                    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
                    'precision': precision,
                    'recall': recall,
                }
            
            # Вычисляем F1-score
            if precision + recall > 0:
                metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
            else:
                metrics['f1_score'] = 0.0
            
            result = {
                'success': True,
                'metrics': metrics,
                'timestamp': results.speed.get('preprocess', 0) if hasattr(results, 'speed') else None
            }
            
            # Если комплексная валидация, добавляем дополнительную информацию
            if comprehensive:
                result['comprehensive'] = {
                    'speed': results.speed if hasattr(results, 'speed') else {},
                    'confusion_matrix': results.confusion_matrix if hasattr(results, 'confusion_matrix') else None
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка валидации: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }


class ModelComparator:
    """Сравнитель моделей"""
    
    def __init__(self, device: str = "cpu"):
        """Инициализация сравнителя
        
        Args:
            device: Устройство для сравнения ('cpu' или '0' для GPU)
        """
        if YOLO is None:
            raise ImportError("ultralytics не установлен. Установите: pip install ultralytics")
        
        self.device = device
        self.validator = AdvancedValidator(device)
    
    def compare(self, models: List[Dict[str, str]], dataset_path: str,
               task: str = "detect", conf_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> Dict[str, Any]:
        """Сравнивает несколько моделей
        
        Args:
            models: Список словарей с ключами 'name' и 'path'
            dataset_path: Путь к датасету
            task: Тип задачи ('detect' или 'segment')
            conf_threshold: Порог уверенности
            iou_threshold: Порог IoU
        
        Returns:
            Словарь с результатами сравнения
        """
        try:
            if len(models) < 2:
                return {'error': "Нужно минимум 2 модели для сравнения"}
            
            results = {}
            comparison_metrics = {}
            
            # Валидируем каждую модель
            for model_info in models:
                model_name = model_info.get('name', 'Unknown')
                model_path = model_info.get('path', '')
                
                if not model_path:
                    logger.warning(f"Пропущена модель {model_name}: путь не указан")
                    continue
                
                validation_result = self.validator.validate(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    task=task,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                
                if 'error' in validation_result:
                    logger.warning(f"Ошибка валидации модели {model_name}: {validation_result['error']}")
                    continue
                
                results[model_name] = validation_result
                
                # Собираем метрики для сравнения
                if 'metrics' in validation_result:
                    metrics = validation_result['metrics']
                    for metric_name, metric_value in metrics.items():
                        if metric_name not in comparison_metrics:
                            comparison_metrics[metric_name] = {
                                'model': model_name,
                                'value': metric_value
                            }
                        else:
                            # Обновляем, если текущая модель лучше
                            if metric_value > comparison_metrics[metric_name]['value']:
                                comparison_metrics[metric_name] = {
                                    'model': model_name,
                                    'value': metric_value
                                }
            
            return {
                'success': True,
                'results': results,
                'comparison_metrics': comparison_metrics,
                'best_model': self._determine_best_model(results)
            }
            
        except Exception as e:
            logger.error(f"Ошибка сравнения моделей: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _determine_best_model(self, results: Dict[str, Dict]) -> Optional[str]:
        """Определяет лучшую модель на основе метрик"""
        if not results:
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, result in results.items():
            if 'metrics' not in result:
                continue
            
            metrics = result['metrics']
            # Простая оценка: среднее mAP50 и mAP50-95
            score = (metrics.get('mAP50', 0) + metrics.get('mAP50-95', 0)) / 2
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model

