# -*- coding: utf-8 -*-
"""
Модуль для расширенной валидации и анализа YOLO моделей
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

# Fix for NumPy stderr issue in QGIS environment
from . import stderr_fix

# Настройка логирования
logger = logging.getLogger(__name__)

# Опциональные импорты с обработкой ошибок
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

try:
    # Используем неинтерактивный backend, чтобы не открывались окна
    import os as _os_for_backend
    _os_for_backend.environ['QT_QPA_PLATFORM'] = 'offscreen'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Suppress matplotlib font manager DEBUG messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from qgis.PyQt.QtCore import QObject, pyqtSignal
except ImportError:
    # Fallback для случаев вне QGIS
    class QObject:
        pass
    def pyqtSignal(*args, **kwargs):
        return None


class ValidationResults(QObject):
    """Класс для хранения и обработки результатов валидации"""
    
    results_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.results = {}
        self.metrics_history = []
        self.class_metrics = {}
        self.confusion_matrix = None
    
    def add_validation_result(self, result: Dict):
        """Добавляет результат валидации"""
        timestamp = datetime.now().isoformat()
        self.results[timestamp] = result
        self.metrics_history.append({
            'timestamp': timestamp,
            'metrics': result.get('metrics', {})
        })
        self.results_updated.emit(result)
    
    def get_latest_metrics(self) -> Dict:
        """Возвращает последние метрики"""
        if self.metrics_history:
            return self.metrics_history[-1]['metrics']
        return {}
    
    def get_metrics_trend(self) -> Dict:
        """Возвращает тренд метрик"""
        if len(self.metrics_history) < 2:
            return {}
        
        trends = {}
        latest = self.metrics_history[-1]['metrics']
        previous = self.metrics_history[-2]['metrics']
        
        for metric in latest:
            if metric in previous:
                trends[metric] = latest[metric] - previous[metric]
        
        return trends


class AdvancedValidator:
    """Расширенный валидатор YOLO моделей"""
    
    def __init__(self):
        self.validation_results = ValidationResults()
    
    def comprehensive_validation(self,
                               model_path: str,
                               dataset_path: str,
                               task: str = 'detect',
                               conf_thresholds: List[float] = None,
                               iou_thresholds: List[float] = None,
                               save_plots: bool = True,
                               output_dir: str = None) -> Dict:
        """
        Выполняет комплексную валидацию модели
        
        :param model_path: Путь к модели
        :param dataset_path: Путь к датасету
        :param task: Тип задачи
        :param conf_thresholds: Список порогов уверенности для анализа
        :param iou_thresholds: Список порогов IoU для анализа
        :param save_plots: Сохранять ли графики
        :param output_dir: Директория для сохранения результатов
        :return: Результаты валидации
        """
        if conf_thresholds is None:
            conf_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        if iou_thresholds is None:
            iou_thresholds = [0.3, 0.45, 0.6, 0.75, 0.9]
        
        if output_dir is None:
            output_dir = os.path.dirname(model_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        validation_results = {
            'model_path': model_path,
            'dataset_path': dataset_path,
            'task': task,
            'timestamp': datetime.now().isoformat(),
            'threshold_analysis': {},
            'class_analysis': {},
            'error_analysis': {},
            'performance_metrics': {}
        }
        
        try:
            # 1. Анализ по порогам уверенности
            validation_results['threshold_analysis'] = self._analyze_confidence_thresholds(
                model_path, dataset_path, task, conf_thresholds, output_dir
            )
            
            # 2. Анализ по порогам IoU
            validation_results['iou_analysis'] = self._analyze_iou_thresholds(
                model_path, dataset_path, task, iou_thresholds, output_dir
            )
            
            # 3. Анализ по классам
            validation_results['class_analysis'] = self._analyze_class_performance(
                model_path, dataset_path, task, output_dir
            )
            
            # 4. Анализ ошибок
            validation_results['error_analysis'] = self._analyze_errors(
                model_path, dataset_path, task, output_dir
            )
            
            # 5. Общие метрики производительности
            validation_results['performance_metrics'] = self._calculate_performance_metrics(
                validation_results
            )
            
            # 6. Создание графиков
            if save_plots:
                self._create_validation_plots(validation_results, output_dir)
            
            # Сохраняем результаты
            results_path = os.path.join(output_dir, 'comprehensive_validation.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, indent=2, ensure_ascii=False)
            
            self.validation_results.add_validation_result(validation_results)
            
            return validation_results
            
        except Exception as e:
            return {'error': f"Ошибка комплексной валидации: {e}"}
    
    def _analyze_confidence_thresholds(self, model_path: str, dataset_path: str, 
                                     task: str, thresholds: List[float], output_dir: str) -> Dict:
        """Анализирует влияние порогов уверенности на метрики используя стандартные методы Ultralytics"""
        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                return {'error': 'ultralytics не установлен'}
            
            model = YOLO(model_path)
            results = {}
            
            for conf in thresholds:
                # Используем стандартный метод валидации Ultralytics
                val_results = model.val(
                    data=os.path.join(dataset_path, 'dataset.yaml'),
                    conf=conf,
                    iou=0.45,
                    save_json=True,
                    verbose=False,
                    plots=False,
                    show=False,
                    workers=0
                )
                
                # Извлекаем метрики из стандартных результатов
                metrics = {}
                if task == 'detect' and hasattr(val_results, 'box') and val_results.box:
                    box = val_results.box
                    metrics = {
                        'mAP50': float(box.map50) if hasattr(box, 'map50') else 0.0,
                        'mAP50-95': float(box.map) if hasattr(box, 'map') else 0.0,
                        'precision': float(box.mp) if hasattr(box, 'mp') else 0.0,
                        'recall': float(box.mr) if hasattr(box, 'mr') else 0.0,
                    }
                elif task == 'segment' and hasattr(val_results, 'seg') and val_results.seg:
                    seg = val_results.seg
                    metrics = {
                        'mAP50': float(seg.map50) if hasattr(seg, 'map50') else 0.0,
                        'mAP50-95': float(seg.map) if hasattr(seg, 'map') else 0.0,
                        'precision': float(seg.mp) if hasattr(seg, 'mp') else 0.0,
                        'recall': float(seg.mr) if hasattr(seg, 'mr') else 0.0,
                    }
                
                results[conf] = metrics
            
            return results
            
        except Exception as e:
            return {'error': f"Ошибка анализа порогов уверенности: {e}"}
    
    def _analyze_iou_thresholds(self, model_path: str, dataset_path: str, 
                              task: str, thresholds: List[float], output_dir: str) -> Dict:
        """Анализирует влияние порогов IoU на метрики используя стандартные методы Ultralytics"""
        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                return {'error': 'ultralytics не установлен'}
            
            model = YOLO(model_path)
            results = {}
            
            for iou in thresholds:
                # Используем стандартный метод валидации Ultralytics
                val_results = model.val(
                    data=os.path.join(dataset_path, 'dataset.yaml'),
                    conf=0.25,
                    iou=iou,
                    save_json=True,
                    verbose=False,
                    plots=False,
                    show=False,
                    workers=0
                )
                
                # Извлекаем метрики из стандартных результатов
                metrics = {}
                if task == 'detect' and hasattr(val_results, 'box') and val_results.box:
                    box = val_results.box
                    metrics = {
                        'mAP50': float(box.map50) if hasattr(box, 'map50') else 0.0,
                        'mAP50-95': float(box.map) if hasattr(box, 'map') else 0.0,
                        'precision': float(box.mp) if hasattr(box, 'mp') else 0.0,
                        'recall': float(box.mr) if hasattr(box, 'mr') else 0.0,
                    }
                elif task == 'segment' and hasattr(val_results, 'seg') and val_results.seg:
                    seg = val_results.seg
                    metrics = {
                        'mAP50': float(seg.map50) if hasattr(seg, 'map50') else 0.0,
                        'mAP50-95': float(seg.map) if hasattr(seg, 'map') else 0.0,
                        'precision': float(seg.mp) if hasattr(seg, 'mp') else 0.0,
                        'recall': float(seg.mr) if hasattr(seg, 'mr') else 0.0,
                    }
                
                results[iou] = metrics
            
            return results
            
        except Exception as e:
            return {'error': f"Ошибка анализа порогов IoU: {e}"}
    
    def _analyze_class_performance(self, model_path: str, dataset_path: str, 
                                 task: str, output_dir: str) -> Dict:
        """Анализирует производительность по классам используя стандартные методы Ultralytics"""
        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                return {'error': 'ultralytics не установлен'}
            
            model = YOLO(model_path)
            # Используем стандартный метод валидации Ultralytics
            val_results = model.val(
                data=os.path.join(dataset_path, 'dataset.yaml'),
                conf=0.25,
                iou=0.45,
                save_json=True,
                verbose=False,
                plots=False,
                show=False,
                workers=0
            )
            
            class_analysis = {}
            
            # Извлекаем метрики по классам из стандартных результатов
            if task == 'detect' and hasattr(val_results, 'box') and val_results.box:
                box = val_results.box
                if hasattr(box, 'ap_class_index') and hasattr(box, 'ap'):
                    for i, class_idx in enumerate(box.ap_class_index):
                        if i < len(box.ap):
                            class_analysis[int(class_idx)] = {
                                'mAP50': float(box.ap50[i]) if hasattr(box, 'ap50') and i < len(box.ap50) else 0.0,
                                'mAP50-95': float(box.ap[i]) if i < len(box.ap) else 0.0,
                            }
            elif task == 'segment' and hasattr(val_results, 'seg') and val_results.seg:
                seg = val_results.seg
                if hasattr(seg, 'ap_class_index') and hasattr(seg, 'ap'):
                    for i, class_idx in enumerate(seg.ap_class_index):
                        if i < len(seg.ap):
                            class_analysis[int(class_idx)] = {
                                'mAP50': float(seg.ap50[i]) if hasattr(seg, 'ap50') and i < len(seg.ap50) else 0.0,
                                'mAP50-95': float(seg.ap[i]) if i < len(seg.ap) else 0.0,
                            }
            
            return class_analysis
            
        except Exception as e:
            return {'error': f"Ошибка анализа классов: {e}"}
    
    def _analyze_errors(self, model_path: str, dataset_path: str, 
                       task: str, output_dir: str) -> Dict:
        """Анализирует типы ошибок модели"""
        try:
            # Здесь можно добавить детальный анализ ошибок
            # Например, анализ ложных срабатываний, пропусков и т.д.
            error_analysis = {
                'false_positives': 0,
                'false_negatives': 0,
                'true_positives': 0,
                'true_negatives': 0,
                'error_rate': 0.0,
                'precision_errors': 0.0,
                'recall_errors': 0.0
            }
            
            return error_analysis
            
        except Exception as e:
            return {'error': f"Ошибка анализа ошибок: {e}"}
    
    def _calculate_performance_metrics(self, validation_results: Dict) -> Dict:
        """Вычисляет общие метрики производительности"""
        try:
            performance_metrics = {
                'overall_mAP50': 0.0,
                'overall_mAP50-95': 0.0,
                'overall_precision': 0.0,
                'overall_recall': 0.0,
                'overall_f1': 0.0,
                'model_efficiency': 0.0,
                'inference_speed': 0.0
            }
            
            # Извлекаем метрики из анализа порогов (используем conf=0.25)
            if 'threshold_analysis' in validation_results and 0.25 in validation_results['threshold_analysis']:
                metrics = validation_results['threshold_analysis'][0.25]
                performance_metrics.update({
                    'overall_mAP50': metrics.get('mAP50', 0.0),
                    'overall_mAP50-95': metrics.get('mAP50-95', 0.0),
                    'overall_precision': metrics.get('precision', 0.0),
                    'overall_recall': metrics.get('recall', 0.0)
                })
                
                # Вычисляем F1-score
                precision = performance_metrics['overall_precision']
                recall = performance_metrics['overall_recall']
                if precision + recall > 0:
                    performance_metrics['overall_f1'] = 2 * (precision * recall) / (precision + recall)
            
            return performance_metrics
            
        except Exception as e:
            return {'error': f"Ошибка вычисления метрик производительности: {e}"}
    
    def _create_validation_plots(self, validation_results: Dict, output_dir: str):
        """Создает графики для анализа валидации"""
        try:
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. График метрик по порогам уверенности
            if 'threshold_analysis' in validation_results:
                self._plot_confidence_thresholds(validation_results['threshold_analysis'], plots_dir)
            
            # 2. График метрик по порогам IoU
            if 'iou_analysis' in validation_results:
                self._plot_iou_thresholds(validation_results['iou_analysis'], plots_dir)
            
            # 3. График производительности по классам
            if 'class_analysis' in validation_results:
                self._plot_class_performance(validation_results['class_analysis'], plots_dir)
            
            # 4. Матрица путаницы (если доступна)
            if 'confusion_matrix' in validation_results:
                self._plot_confusion_matrix(validation_results['confusion_matrix'], plots_dir)
            
        except Exception as e:
            logger.error(f"Ошибка создания графиков: {e}", exc_info=True)
    
    def _plot_confidence_thresholds(self, threshold_data: Dict, plots_dir: str):
        """Создает график метрик по порогам уверенности"""
        try:
            if plt is None:
                logger.warning("matplotlib не установлен. График не будет создан.")
                return
            
            thresholds = list(threshold_data.keys())
            metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                values = [threshold_data[t].get(metric, 0.0) for t in thresholds]
                axes[i].plot(thresholds, values, marker='o', linewidth=2, markersize=6)
                axes[i].set_title(f'{metric} vs Confidence Threshold')
                axes[i].set_xlabel('Confidence Threshold')
                axes[i].set_ylabel(metric)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'confidence_thresholds.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка создания графика порогов уверенности: {e}", exc_info=True)
    
    def _plot_iou_thresholds(self, iou_data: Dict, plots_dir: str):
        """Создает график метрик по порогам IoU"""
        try:
            if plt is None:
                logger.warning("matplotlib не установлен. График не будет создан.")
                return
            
            ious = list(iou_data.keys())
            metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                values = [iou_data[iou].get(metric, 0.0) for iou in ious]
                axes[i].plot(ious, values, marker='s', linewidth=2, markersize=6, color='red')
                axes[i].set_title(f'{metric} vs IoU Threshold')
                axes[i].set_xlabel('IoU Threshold')
                axes[i].set_ylabel(metric)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'iou_thresholds.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка создания графика порогов IoU: {e}", exc_info=True)
    
    def _plot_class_performance(self, class_data: Dict, plots_dir: str):
        """Создает график производительности по классам"""
        try:
            if plt is None:
                logger.warning("matplotlib не установлен. График не будет создан.")
                return
            
            if not class_data:
                return
            
            classes = list(class_data.keys())
            metrics = ['mAP50', 'mAP50-95']
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for i, metric in enumerate(metrics):
                values = [class_data[cls].get(metric, 0.0) for cls in classes]
                bars = axes[i].bar(range(len(classes)), values, alpha=0.7)
                axes[i].set_title(f'{metric} by Class')
                axes[i].set_xlabel('Class ID')
                axes[i].set_ylabel(metric)
                axes[i].set_xticks(range(len(classes)))
                axes[i].set_xticklabels(classes)
                axes[i].grid(True, alpha=0.3)
                
                # Добавляем значения на столбцы
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'class_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка создания графика производительности классов: {e}", exc_info=True)
    
    def _plot_confusion_matrix(self, confusion_matrix, plots_dir: str):
        """Создает график матрицы путаницы"""
        try:
            if plt is None or sns is None:
                logger.warning("matplotlib или seaborn не установлены. График не будет создан.")
                return
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка создания матрицы путаницы: {e}", exc_info=True)


class ModelComparator:
    """Класс для сравнения нескольких моделей"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_models(self, 
                      models: List[Dict],
                      dataset_path: str,
                      task: str = 'detect',
                      output_dir: str = None) -> Dict:
        """
        Сравнивает несколько моделей
        
        :param models: Список словарей с информацией о моделях
        :param dataset_path: Путь к датасету
        :param task: Тип задачи
        :param output_dir: Директория для сохранения результатов
        :return: Результаты сравнения
        """
        if output_dir is None:
            output_dir = 'model_comparison'
        
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_results = {
            'models': [],
            'comparison_metrics': {},
            'ranking': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            validator = AdvancedValidator()
            
            # Валидируем каждую модель
            for model_info in models:
                model_path = model_info['path']
                model_name = model_info.get('name', os.path.basename(model_path))
                
                logger.info(f"Валидация модели: {model_name}")
                
                validation_result = validator.comprehensive_validation(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    task=task,
                    output_dir=os.path.join(output_dir, model_name)
                )
                
                model_result = {
                    'name': model_name,
                    'path': model_path,
                    'validation': validation_result,
                    'summary_metrics': self._extract_summary_metrics(validation_result)
                }
                
                comparison_results['models'].append(model_result)
            
            # Сравниваем модели
            comparison_results['comparison_metrics'] = self._compare_metrics(comparison_results['models'])
            comparison_results['ranking'] = self._rank_models(comparison_results['models'])
            
            # Создаем графики сравнения
            self._create_comparison_plots(comparison_results, output_dir)
            
            # Сохраняем результаты
            results_path = os.path.join(output_dir, 'model_comparison.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            
            return comparison_results
            
        except Exception as e:
            return {'error': f"Ошибка сравнения моделей: {e}"}
    
    def _extract_summary_metrics(self, validation_result: Dict) -> Dict:
        """Извлекает основные метрики из результатов валидации"""
        if 'error' in validation_result:
            return {'error': validation_result['error']}
        
        performance = validation_result.get('performance_metrics', {})
        
        return {
            'mAP50': performance.get('overall_mAP50', 0.0),
            'mAP50-95': performance.get('overall_mAP50-95', 0.0),
            'precision': performance.get('overall_precision', 0.0),
            'recall': performance.get('overall_recall', 0.0),
            'f1_score': performance.get('overall_f1', 0.0)
        }
    
    def _compare_metrics(self, models: List[Dict]) -> Dict:
        """Сравнивает метрики моделей"""
        comparison = {
            'best_mAP50': {'model': '', 'value': 0.0},
            'best_mAP50-95': {'model': '', 'value': 0.0},
            'best_precision': {'model': '', 'value': 0.0},
            'best_recall': {'model': '', 'value': 0.0},
            'best_f1': {'model': '', 'value': 0.0}
        }
        
        for model in models:
            metrics = model.get('summary_metrics', {})
            if 'error' in metrics:
                continue
            
            model_name = model['name']
            
            for metric in comparison:
                metric_key = metric.replace('best_', '')
                if metric_key in metrics:
                    if metrics[metric_key] > comparison[metric]['value']:
                        comparison[metric]['model'] = model_name
                        comparison[metric]['value'] = metrics[metric_key]
        
        return comparison
    
    def _rank_models(self, models: List[Dict]) -> Dict:
        """Ранжирует модели по общей производительности"""
        # Вычисляем общий балл для каждой модели
        model_scores = []
        
        for model in models:
            metrics = model.get('summary_metrics', {})
            if 'error' in metrics:
                continue
            
            # Взвешенная сумма метрик
            score = (
                metrics.get('mAP50', 0.0) * 0.3 +
                metrics.get('mAP50-95', 0.0) * 0.3 +
                metrics.get('precision', 0.0) * 0.2 +
                metrics.get('recall', 0.0) * 0.2
            )
            
            model_scores.append({
                'name': model['name'],
                'score': score,
                'metrics': metrics
            })
        
        # Сортируем по убыванию балла
        model_scores.sort(key=lambda x: x['score'], reverse=True)
        
        ranking = {
            'ranked_models': model_scores,
            'best_overall': model_scores[0]['name'] if model_scores else None
        }
        
        return ranking
    
    def _create_comparison_plots(self, comparison_results: Dict, output_dir: str):
        """Создает графики сравнения моделей"""
        try:
            plots_dir = os.path.join(output_dir, 'comparison_plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            models = comparison_results['models']
            model_names = [m['name'] for m in models if 'error' not in m.get('summary_metrics', {})]
            
            if not model_names:
                return
            
            # График сравнения метрик
            metrics = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1_score']
            metric_values = {metric: [] for metric in metrics}
            
            for model in models:
                summary = model.get('summary_metrics', {})
                if 'error' not in summary:
                    for metric in metrics:
                        metric_values[metric].append(summary.get(metric, 0.0))
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                if i < len(axes):
                    bars = axes[i].bar(model_names, metric_values[metric], alpha=0.7)
                    axes[i].set_title(f'{metric} Comparison')
                    axes[i].set_ylabel(metric)
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Добавляем значения на столбцы
                    for bar, value in zip(bars, metric_values[metric]):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
            
            # Удаляем лишний subplot
            if len(axes) > len(metrics):
                fig.delaxes(axes[-1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # График ранжирования
            ranking = comparison_results.get('ranking', {}).get('ranked_models', [])
            if ranking:
                model_names_ranked = [m['name'] for m in ranking]
                scores = [m['score'] for m in ranking]
                
                plt.figure(figsize=(12, 8))
                bars = plt.bar(model_names_ranked, scores, alpha=0.7, color='skyblue')
                plt.title('Model Ranking (Overall Score)')
                plt.xlabel('Model')
                plt.ylabel('Overall Score')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Добавляем значения на столбцы
                for bar, score in zip(bars, scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'model_ranking.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка создания графиков сравнения: {e}", exc_info=True)
