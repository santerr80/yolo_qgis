# -*- coding: utf-8 -*-
"""
Модуль для отслеживания метрик обучения и валидации YOLO моделей
"""

import os
import json
import csv
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Fix for NumPy stderr issue in QGIS environment
from . import stderr_fix

# Опциональные импорты с обработкой ошибок
try:
    import numpy as np
    # Проверяем совместимость версии numpy
    if hasattr(np, '__version__'):
        version_parts = np.__version__.split('.')
        major_version = int(version_parts[0])
        if major_version >= 2:
            print(f"Предупреждение: numpy версии {np.__version__} может быть несовместима с QGIS. Рекомендуется numpy v1.x")
except ImportError:
    np = None

try:
    from qgis.PyQt.QtCore import QObject, pyqtSignal, QTimer
    from qgis.PyQt.QtWidgets import QApplication
except ImportError:
    # Fallback для случаев вне QGIS
    class QObject:
        pass
    def pyqtSignal(*args, **kwargs):
        return None
    class QTimer:
        def __init__(self):
            pass
        def start(self, ms):
            pass
        def timeout(self):
            pass
    class QApplication:
        pass




class MetricsDatabase:
    """Класс для работы с базой данных метрик"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or 'yolo_metrics.db'
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Инициализирует базу данных"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Создаем таблицу для метрик обучения
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    experiment_id TEXT,
                    epoch INTEGER,
                    phase TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Создаем таблицу для экспериментов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    task TEXT,
                    model_type TEXT,
                    dataset_path TEXT,
                    config TEXT,
                    status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME
                )
            ''')
            
            # Создаем таблицу для валидации
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    model_path TEXT,
                    dataset_path TEXT,
                    mAP50 REAL,
                    mAP50_95 REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def create_experiment(self, experiment_id: str, name: str, task: str, 
                         model_type: str, dataset_path: str, config: Dict) -> bool:
        """Создает новый эксперимент"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO experiments 
                    (id, name, task, model_type, dataset_path, config, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (experiment_id, name, task, model_type, dataset_path, 
                     json.dumps(config), 'running'))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            print(f"Ошибка создания эксперимента: {e}")
            return False
    
    def update_experiment_status(self, experiment_id: str, status: str):
        """Обновляет статус эксперимента"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                completed_at = datetime.now().isoformat() if status == 'completed' else None
                
                cursor.execute('''
                    UPDATE experiments 
                    SET status = ?, completed_at = ?
                    WHERE id = ?
                ''', (status, completed_at, experiment_id))
                
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Ошибка обновления статуса эксперимента: {e}")
    
    def log_metrics(self, experiment_id: str, epoch: int, phase: str, metrics: Dict):
        """Логирует метрики в базу данных"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                timestamp = datetime.now(timezone.utc).isoformat()
                
                for metric_name, metric_value in metrics.items():
                    cursor.execute('''
                        INSERT INTO training_metrics 
                        (timestamp, experiment_id, epoch, phase, metric_name, metric_value)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (timestamp, experiment_id, epoch, phase, metric_name, metric_value))
                
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Ошибка логирования метрик: {e}")
    
    def log_validation_results(self, experiment_id: str, model_path: str, 
                             dataset_path: str, results: Dict):
        """Логирует результаты валидации"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                timestamp = datetime.now(timezone.utc).isoformat()
                
                cursor.execute('''
                    INSERT INTO validation_results 
                    (experiment_id, model_path, dataset_path, mAP50, mAP50_95, 
                     precision, recall, f1_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (experiment_id, model_path, dataset_path,
                     results.get('mAP50', 0.0), results.get('mAP50-95', 0.0),
                     results.get('precision', 0.0), results.get('recall', 0.0),
                     results.get('f1_score', 0.0), timestamp))
                
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Ошибка логирования результатов валидации: {e}")
    
    def get_experiment_metrics(self, experiment_id: str) -> List[Dict]:
        """Получает метрики эксперимента"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT epoch, phase, metric_name, metric_value, timestamp
                    FROM training_metrics
                    WHERE experiment_id = ?
                    ORDER BY epoch, timestamp
                ''', (experiment_id,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'epoch': row[0],
                        'phase': row[1],
                        'metric_name': row[2],
                        'metric_value': row[3],
                        'timestamp': row[4]
                    })
                
                conn.close()
                return results
        except Exception as e:
            print(f"Ошибка получения метрик эксперимента: {e}")
            return []
    
    def get_experiments_list(self) -> List[Dict]:
        """Получает список всех экспериментов"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, name, task, model_type, status, created_at, completed_at
                    FROM experiments
                    ORDER BY created_at DESC
                ''')
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'name': row[1],
                        'task': row[2],
                        'model_type': row[3],
                        'status': row[4],
                        'created_at': row[5],
                        'completed_at': row[6]
                    })
                
                conn.close()
                return results
        except Exception as e:
            print(f"Ошибка получения списка экспериментов: {e}")
            return []


class MetricsTracker(QObject):
    """Основной класс для отслеживания метрик"""
    
    metrics_updated = pyqtSignal(dict)
    experiment_started = pyqtSignal(str)
    experiment_completed = pyqtSignal(str, dict)
    
    def __init__(self, log_dir: str = None, db_path: str = None):
        super().__init__()
        self.database = MetricsDatabase(db_path)
        self.current_experiment = None
        self.metrics_history = []
        self.timer = QTimer()
        self.timer.timeout.connect(self._periodic_save)
        self.timer.start(30000)  # Сохраняем каждые 30 секунд
    
    def start_experiment(self, experiment_id: str, name: str, task: str,
                        model_type: str, dataset_path: str, config: Dict):
        """Начинает новый эксперимент"""
        self.current_experiment = experiment_id
        
        success = self.database.create_experiment(
            experiment_id, name, task, model_type, dataset_path, config
        )
        
        if success:
            self.experiment_started.emit(experiment_id)
        
        return success
    
    def log_training_metrics(self, epoch: int, metrics: Dict):
        """Логирует метрики обучения"""
        if not self.current_experiment:
            return
        
        # Логируем в файлы
        
        # Логируем в базу данных
        self.database.log_metrics(self.current_experiment, epoch, 'training', metrics)
        
        # Сохраняем в историю
        self.metrics_history.append({
            'epoch': epoch,
            'phase': 'training',
            'metrics': metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Отправляем сигнал
        self.metrics_updated.emit({
            'experiment_id': self.current_experiment,
            'epoch': epoch,
            'phase': 'training',
            'metrics': metrics
        })
    
    def log_validation_metrics(self, epoch: int, metrics: Dict):
        """Логирует метрики валидации"""
        if not self.current_experiment:
            return
        
        # Логируем в файлы
        
        # Логируем в базу данных
        self.database.log_metrics(self.current_experiment, epoch, 'validation', metrics)
        
        # Сохраняем в историю
        self.metrics_history.append({
            'epoch': epoch,
            'phase': 'validation',
            'metrics': metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Отправляем сигнал
        self.metrics_updated.emit({
            'experiment_id': self.current_experiment,
            'epoch': epoch,
            'phase': 'validation',
            'metrics': metrics
        })
    
    def log_final_validation(self, model_path: str, dataset_path: str, results: Dict):
        """Логирует финальные результаты валидации"""
        if not self.current_experiment:
            return
        
        # Логируем в базу данных
        self.database.log_validation_results(
            self.current_experiment, model_path, dataset_path, results
        )
        
        # Завершаем эксперимент
        self.database.update_experiment_status(self.current_experiment, 'completed')
        
        # Отправляем сигнал
        self.experiment_completed.emit(self.current_experiment, results)
        
    
    def get_experiment_summary(self, experiment_id: str = None) -> Dict:
        """Получает сводку по эксперименту"""
        if experiment_id is None:
            experiment_id = self.current_experiment
        
        if not experiment_id:
            return {}
        
        # Получаем метрики из базы данных
        metrics = self.database.get_experiment_metrics(experiment_id)
        
        # Анализируем метрики
        summary = {
            'experiment_id': experiment_id,
            'total_epochs': 0,
            'best_metrics': {},
            'final_metrics': {},
            'training_curve': {},
            'validation_curve': {}
        }
        
        if metrics:
            # Группируем по эпохам
            epochs = {}
            for metric in metrics:
                epoch = metric['epoch']
                if epoch not in epochs:
                    epochs[epoch] = {'training': {}, 'validation': {}}
                epochs[epoch][metric['phase']][metric['metric_name']] = metric['metric_value']
            
            summary['total_epochs'] = max(epochs.keys()) if epochs else 0
            
            # Находим лучшие метрики
            best_metrics = {}
            for epoch_data in epochs.values():
                for phase, phase_metrics in epoch_data.items():
                    for metric_name, metric_value in phase_metrics.items():
                        if metric_name not in best_metrics or metric_value > best_metrics[metric_name]:
                            best_metrics[metric_name] = metric_value
            
            summary['best_metrics'] = best_metrics
            
            # Получаем финальные метрики
            if summary['total_epochs'] > 0:
                final_epoch = epochs[summary['total_epochs']]
                summary['final_metrics'] = final_epoch.get('validation', {})
            
            # Строим кривые обучения
            for metric_name in best_metrics.keys():
                training_values = []
                validation_values = []
                
                for epoch in sorted(epochs.keys()):
                    training_values.append(epochs[epoch]['training'].get(metric_name, 0.0))
                    validation_values.append(epochs[epoch]['validation'].get(metric_name, 0.0))
                
                summary['training_curve'][metric_name] = training_values
                summary['validation_curve'][metric_name] = validation_values
        
        return summary
    
    def get_all_experiments(self) -> List[Dict]:
        """Получает список всех экспериментов"""
        return self.database.get_experiments_list()
    
    def _periodic_save(self):
        """Периодическое сохранение данных"""
        # Здесь можно добавить дополнительную логику сохранения
        pass
    
    def export_metrics(self, experiment_id: str, output_path: str, format: str = 'json'):
        """Экспортирует метрики в файл"""
        try:
            summary = self.get_experiment_summary(experiment_id)
            
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
            elif format.lower() == 'csv':
                # Экспортируем кривые обучения в CSV
                import pandas as pd
                
                data = []
                for metric_name in summary.get('training_curve', {}):
                    for i, (train_val, val_val) in enumerate(zip(
                        summary['training_curve'][metric_name],
                        summary['validation_curve'][metric_name]
                    )):
                        data.append({
                            'epoch': i + 1,
                            'metric': metric_name,
                            'training': train_val,
                            'validation': val_val
                        })
                
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
            
            return True
        except Exception as e:
            print(f"Ошибка экспорта метрик: {e}")
            return False


class MetricsVisualizer:
    """Класс для визуализации метрик"""
    
    def __init__(self, tracker: MetricsTracker):
        self.tracker = tracker
    
    def create_training_plots(self, experiment_id: str, output_dir: str):
        """Создает графики обучения"""
        try:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
            except ImportError:
                print("matplotlib или seaborn не установлены. Графики не будут созданы.")
                return False
            
            summary = self.tracker.get_experiment_summary(experiment_id)
            
            if not summary.get('training_curve'):
                return False
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Настраиваем стиль
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Создаем графики для каждой метрики
            for metric_name in summary['training_curve']:
                plt.figure(figsize=(12, 8))
                
                epochs = list(range(1, len(summary['training_curve'][metric_name]) + 1))
                training_values = summary['training_curve'][metric_name]
                validation_values = summary['validation_curve'][metric_name]
                
                plt.plot(epochs, training_values, label='Training', linewidth=2, marker='o')
                plt.plot(epochs, validation_values, label='Validation', linewidth=2, marker='s')
                
                plt.title(f'{metric_name} - Training Progress', fontsize=16, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(metric_name, fontsize=12)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Добавляем аннотацию с лучшим значением
                best_val = max(validation_values)
                best_epoch = validation_values.index(best_val) + 1
                plt.annotate(f'Best: {best_val:.4f} (Epoch {best_epoch})',
                           xy=(best_epoch, best_val), xytext=(best_epoch + 5, best_val + 0.01),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                           fontsize=10, color='red')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower()}_curve.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # Создаем сводный график
            self._create_summary_plot(summary, output_dir)
            
            return True
            
        except Exception as e:
            print(f"Ошибка создания графиков: {e}")
            return False
    
    def _create_summary_plot(self, summary: Dict, output_dir: str):
        """Создает сводный график всех метрик"""
        try:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                return
            
            metrics = list(summary['training_curve'].keys())
            if not metrics:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, metric_name in enumerate(metrics[:4]):  # Показываем максимум 4 метрики
                if i < len(axes):
                    epochs = list(range(1, len(summary['training_curve'][metric_name]) + 1))
                    training_values = summary['training_curve'][metric_name]
                    validation_values = summary['validation_curve'][metric_name]
                    
                    axes[i].plot(epochs, training_values, label='Training', linewidth=2)
                    axes[i].plot(epochs, validation_values, label='Validation', linewidth=2)
                    axes[i].set_title(metric_name, fontweight='bold')
                    axes[i].set_xlabel('Epoch')
                    axes[i].set_ylabel(metric_name)
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            # Удаляем лишние subplot'ы
            for i in range(len(metrics), len(axes)):
                fig.delaxes(axes[i])
            
            plt.suptitle('Training Progress Summary', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'training_summary.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Ошибка создания сводного графика: {e}")
    
    def create_comparison_plot(self, experiment_ids: List[str], output_path: str):
        """Создает график сравнения экспериментов"""
        try:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib не установлен. График не будет создан.")
                return False
            
            if np is None:
                print("numpy не установлен. График не будет создан.")
                return False
            
            plt.figure(figsize=(14, 10))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_ids)))
            
            for i, exp_id in enumerate(experiment_ids):
                summary = self.tracker.get_experiment_summary(exp_id)
                
                if 'validation_curve' in summary and 'mAP50' in summary['validation_curve']:
                    epochs = list(range(1, len(summary['validation_curve']['mAP50']) + 1))
                    values = summary['validation_curve']['mAP50']
                    
                    plt.plot(epochs, values, label=f'Experiment {exp_id}', 
                           linewidth=2, color=colors[i], marker='o', markersize=4)
            
            plt.title('Model Comparison - mAP50', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('mAP50', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Ошибка создания графика сравнения: {e}")
            return False
