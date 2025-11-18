# -*- coding: utf-8 -*-
"""
Модуль для отслеживания метрик обучения YOLO моделей
"""

import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsDatabase:
    """База данных для хранения метрик обучения"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Инициализация базы данных метрик
        
        :param db_path: Путь к файлу базы данных. Если None, используется стандартный путь
        """
        if db_path is None:
            # Используем стандартный путь в директории плагина
            plugin_dir = Path(__file__).parent
            db_path = str(plugin_dir / 'yolo_metrics.db')
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Инициализирует структуру базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица экспериментов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    dataset_path TEXT,
                    status TEXT DEFAULT 'running',
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    config TEXT,
                    final_metrics TEXT
                )
            ''')
            
            # Таблица метрик по эпохам
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')
            
            # Индексы для быстрого поиска
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_experiment 
                ON metrics(experiment_id, epoch, phase)
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}", exc_info=True)
            raise
    
    def create_experiment(self, experiment_id: str, name: str, task: str, 
                         model_type: str, dataset_path: str, config: Dict) -> bool:
        """
        Создает новую запись эксперимента
        
        :param experiment_id: Уникальный ID эксперимента
        :param name: Название эксперимента
        :param task: Тип задачи (detect/segment)
        :param model_type: Тип модели (yolov8n, yolov11s и т.д.)
        :param dataset_path: Путь к датасету
        :param config: Конфигурация обучения
        :return: True если успешно
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO experiments (id, name, task, model_type, dataset_path, 
                                       status, created_at, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                name,
                task,
                model_type,
                dataset_path,
                'running',
                datetime.now().isoformat(),
                json.dumps(config)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Ошибка создания эксперимента: {e}", exc_info=True)
            return False
    
    def add_metric(self, experiment_id: str, epoch: int, phase: str, 
                   metric_name: str, metric_value: float) -> bool:
        """
        Добавляет метрику для эксперимента
        
        :param experiment_id: ID эксперимента
        :param epoch: Номер эпохи
        :param phase: Фаза (training/validation)
        :param metric_name: Название метрики
        :param metric_value: Значение метрики
        :return: True если успешно
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (experiment_id, epoch, phase, metric_name, 
                                   metric_value, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                epoch,
                phase,
                metric_name,
                metric_value,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления метрики: {e}", exc_info=True)
            return False
    
    def get_experiment_metrics(self, experiment_id: str) -> List[Dict]:
        """
        Получает все метрики для эксперимента
        
        :param experiment_id: ID эксперимента
        :return: Список метрик
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT epoch, phase, metric_name, metric_value, timestamp
                FROM metrics
                WHERE experiment_id = ?
                ORDER BY epoch, phase, metric_name
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
            logger.error(f"Ошибка получения метрик: {e}", exc_info=True)
            return []
    
    def update_experiment_status(self, experiment_id: str, status: str, 
                                final_metrics: Optional[Dict] = None) -> bool:
        """
        Обновляет статус эксперимента
        
        :param experiment_id: ID эксперимента
        :param status: Новый статус (completed/failed/cancelled)
        :param final_metrics: Финальные метрики
        :return: True если успешно
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_data = {
                'status': status,
                'completed_at': datetime.now().isoformat()
            }
            
            if final_metrics:
                update_data['final_metrics'] = json.dumps(final_metrics)
            
            cursor.execute('''
                UPDATE experiments
                SET status = ?, completed_at = ?, final_metrics = ?
                WHERE id = ?
            ''', (
                update_data['status'],
                update_data['completed_at'],
                update_data.get('final_metrics'),
                experiment_id
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления статуса: {e}", exc_info=True)
            return False
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """
        Получает информацию об эксперименте
        
        :param experiment_id: ID эксперимента
        :return: Словарь с информацией об эксперименте или None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, task, model_type, dataset_path, status, 
                       created_at, completed_at, config, final_metrics
                FROM experiments
                WHERE id = ?
            ''', (experiment_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return {
                'id': row[0],
                'name': row[1],
                'task': row[2],
                'model_type': row[3],
                'dataset_path': row[4],
                'status': row[5],
                'created_at': row[6],
                'completed_at': row[7],
                'config': json.loads(row[8]) if row[8] else {},
                'final_metrics': json.loads(row[9]) if row[9] else {}
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения эксперимента: {e}", exc_info=True)
            return None
    
    def list_experiments(self) -> List[Dict]:
        """
        Получает список всех экспериментов
        
        :return: Список экспериментов
        """
        try:
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
            logger.error(f"Ошибка получения списка экспериментов: {e}", exc_info=True)
            return []


class MetricsTracker:
    """Класс для отслеживания метрик обучения"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Инициализация трекера метрик
        
        :param db_path: Путь к базе данных
        """
        self.database = MetricsDatabase(db_path)
        self.current_experiment_id = None
    
    def start_experiment(self, experiment_id: str, name: str, task: str,
                        model_type: str, dataset_path: str, config: Dict) -> bool:
        """
        Начинает отслеживание нового эксперимента
        
        :param experiment_id: ID эксперимента
        :param name: Название
        :param task: Тип задачи
        :param model_type: Тип модели
        :param dataset_path: Путь к датасету
        :param config: Конфигурация
        :return: True если успешно
        """
        self.current_experiment_id = experiment_id
        return self.database.create_experiment(
            experiment_id, name, task, model_type, dataset_path, config
        )
    
    def log_metric(self, epoch: int, phase: str, metric_name: str, 
                   metric_value: float) -> bool:
        """
        Логирует метрику
        
        :param epoch: Номер эпохи
        :param phase: Фаза (training/validation)
        :param metric_name: Название метрики
        :param metric_value: Значение
        :return: True если успешно
        """
        if not self.current_experiment_id:
            logger.warning("Нет активного эксперимента для логирования метрик")
            return False
        
        return self.database.add_metric(
            self.current_experiment_id, epoch, phase, metric_name, metric_value
        )
    
    def log_metrics_batch(self, epoch: int, phase: str, metrics: Dict[str, float]) -> bool:
        """
        Логирует несколько метрик одновременно
        
        :param epoch: Номер эпохи
        :param phase: Фаза
        :param metrics: Словарь метрик
        :return: True если успешно
        """
        success = True
        for metric_name, metric_value in metrics.items():
            if not self.log_metric(epoch, phase, metric_name, metric_value):
                success = False
        return success
    
    def complete_experiment(self, status: str = 'completed', 
                           final_metrics: Optional[Dict] = None) -> bool:
        """
        Завершает эксперимент
        
        :param status: Статус завершения
        :param final_metrics: Финальные метрики
        :return: True если успешно
        """
        if not self.current_experiment_id:
            return False
        
        result = self.database.update_experiment_status(
            self.current_experiment_id, status, final_metrics
        )
        self.current_experiment_id = None
        return result


class MetricsVisualizer:
    """Класс для визуализации метрик"""
    
    @staticmethod
    def plot_training_curves(metrics_data: List[Dict], output_path: Optional[str] = None):
        """
        Создает графики кривых обучения
        
        :param metrics_data: Данные метрик
        :param output_path: Путь для сохранения графика
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Используем backend без GUI
            
            # Группируем метрики по эпохам и фазам
            epochs = sorted(set(m['epoch'] for m in metrics_data))
            training_metrics = {}
            validation_metrics = {}
            
            for metric in metrics_data:
                epoch = metric['epoch']
                phase = metric['phase']
                name = metric['metric_name']
                value = metric['metric_value']
                
                if phase == 'training':
                    if name not in training_metrics:
                        training_metrics[name] = []
                    training_metrics[name].append((epoch, value))
                elif phase == 'validation':
                    if name not in validation_metrics:
                        validation_metrics[name] = []
                    validation_metrics[name].append((epoch, value))
            
            # Создаем графики
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Training Metrics', fontsize=16)
            
            # Loss
            ax = axes[0, 0]
            if 'loss' in training_metrics:
                train_loss = sorted(training_metrics['loss'], key=lambda x: x[0])
                ax.plot([x[0] for x in train_loss], [x[1] for x in train_loss], 
                       label='Train Loss', marker='o')
            if 'loss' in validation_metrics:
                val_loss = sorted(validation_metrics['loss'], key=lambda x: x[0])
                ax.plot([x[0] for x in val_loss], [x[1] for x in val_loss], 
                       label='Val Loss', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss')
            ax.legend()
            ax.grid(True)
            
            # mAP50
            ax = axes[0, 1]
            if 'mAP50' in validation_metrics:
                map50 = sorted(validation_metrics['mAP50'], key=lambda x: x[0])
                ax.plot([x[0] for x in map50], [x[1] for x in map50], 
                       label='mAP50', marker='o', color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP50')
            ax.set_title('mAP50')
            ax.legend()
            ax.grid(True)
            
            # Precision
            ax = axes[1, 0]
            if 'precision' in validation_metrics:
                precision = sorted(validation_metrics['precision'], key=lambda x: x[0])
                ax.plot([x[0] for x in precision], [x[1] for x in precision], 
                       label='Precision', marker='o', color='blue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Precision')
            ax.set_title('Precision')
            ax.legend()
            ax.grid(True)
            
            # Recall
            ax = axes[1, 1]
            if 'recall' in validation_metrics:
                recall = sorted(validation_metrics['recall'], key=lambda x: x[0])
                ax.plot([x[0] for x in recall], [x[1] for x in recall], 
                       label='Recall', marker='o', color='red')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Recall')
            ax.set_title('Recall')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib не установлен, визуализация недоступна")
        except Exception as e:
            logger.error(f"Ошибка создания графиков: {e}", exc_info=True)

