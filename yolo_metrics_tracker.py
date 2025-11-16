# -*- coding: utf-8 -*-
"""
Модуль для отслеживания метрик обучения YOLO моделей
Использует стандартные методы ultralytics для максимальной совместимости
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsDatabase:
    """Простая база данных SQLite для хранения метрик"""
    
    def __init__(self, db_path: str = "yolo_metrics.db"):
        """Инициализация базы данных"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Создает таблицы базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица экспериментов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT,
                task TEXT,
                model_type TEXT,
                status TEXT,
                created_at TEXT,
                completed_at TEXT,
                config TEXT
            )
        """)
        
        # Таблица метрик
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                epoch INTEGER,
                phase TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_experiment(self, experiment_id: str, name: str, task: str, 
                      model_type: str, config: Dict):
        """Добавляет новый эксперимент"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO experiments 
            (id, name, task, model_type, status, created_at, config)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, name, task, model_type, 'running',
            datetime.now().isoformat(), json.dumps(config)
        ))
        
        conn.commit()
        conn.close()
    
    def update_experiment_status(self, experiment_id: str, status: str, 
                                completed_at: Optional[str] = None):
        """Обновляет статус эксперимента"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if completed_at:
            cursor.execute("""
                UPDATE experiments 
                SET status = ?, completed_at = ?
                WHERE id = ?
            """, (status, completed_at, experiment_id))
        else:
            cursor.execute("""
                UPDATE experiments 
                SET status = ?
                WHERE id = ?
            """, (status, experiment_id))
        
        conn.commit()
        conn.close()
    
    def add_metric(self, experiment_id: str, epoch: int, phase: str,
                  metric_name: str, metric_value: float):
        """Добавляет метрику"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics 
            (experiment_id, epoch, phase, metric_name, metric_value, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, epoch, phase, metric_name, metric_value,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_experiment_metrics(self, experiment_id: str) -> List[Dict]:
        """Получает все метрики эксперимента"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT epoch, phase, metric_name, metric_value
            FROM metrics
            WHERE experiment_id = ?
            ORDER BY epoch, phase
        """, (experiment_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'epoch': row[0],
                'phase': row[1],
                'metric_name': row[2],
                'metric_value': row[3]
            })
        
        conn.close()
        return results
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Получает информацию об эксперименте"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, task, model_type, status, created_at, completed_at, config
            FROM experiments
            WHERE id = ?
        """, (experiment_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'task': row[2],
                'model_type': row[3],
                'status': row[4],
                'created_at': row[5],
                'completed_at': row[6],
                'config': json.loads(row[7]) if row[7] else {}
            }
        return None
    
    def get_all_experiments(self) -> List[Dict]:
        """Получает все эксперименты"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, task, model_type, status, created_at, completed_at
            FROM experiments
            ORDER BY created_at DESC
        """)
        
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


class MetricsTracker:
    """Трекер метрик для обучения YOLO моделей"""
    
    def __init__(self, experiment_id: str, db_path: str = "yolo_metrics.db",
                 log_dir: Optional[str] = None):
        """Инициализация трекера"""
        self.experiment_id = experiment_id
        self.database = MetricsDatabase(db_path)
        self.log_dir = log_dir or "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics_log = []
    
    def log_metrics(self, epoch: int, phase: str, metrics: Dict[str, float]):
        """Логирует метрики для эпохи"""
        timestamp = datetime.now().isoformat()
        
        for metric_name, metric_value in metrics.items():
            # Сохраняем в базу данных
            self.database.add_metric(
                experiment_id=self.experiment_id,
                epoch=epoch,
                phase=phase,
                metric_name=metric_name,
                metric_value=float(metric_value)
            )
            
            # Сохраняем в список для быстрого доступа
            self.metrics_log.append({
                'epoch': epoch,
                'phase': phase,
                'metric_name': metric_name,
                'metric_value': float(metric_value),
                'timestamp': timestamp
            })
    
    def save_to_json(self, filepath: Optional[str] = None):
        """Сохраняет метрики в JSON файл"""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"metrics_{self.experiment_id}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_log, f, indent=2, ensure_ascii=False)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Получает сводку метрик"""
        metrics = self.database.get_experiment_metrics(self.experiment_id)
        
        if not metrics:
            return {}
        
        # Группируем по эпохам и фазам
        summary = {}
        for metric in metrics:
            epoch = metric['epoch']
            phase = metric['phase']
            name = metric['metric_name']
            value = metric['metric_value']
            
            if epoch not in summary:
                summary[epoch] = {}
            if phase not in summary[epoch]:
                summary[epoch][phase] = {}
            
            summary[epoch][phase][name] = value
        
        return summary


class MetricsVisualizer:
    """Визуализатор метрик (заглушка для будущей реализации)"""
    
    def __init__(self, log_dir: str = "logs"):
        """Инициализация визуализатора"""
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
    
    def create_plots(self, experiment_id: str, output_dir: Optional[str] = None) -> bool:
        """Создает графики метрик (заглушка)"""
        # В будущем здесь можно добавить создание графиков с помощью matplotlib
        logger.info(f"Создание графиков для эксперимента {experiment_id}")
        return True
    
    def save_plots(self, experiment_id: str, output_path: str) -> bool:
        """Сохраняет графики (заглушка)"""
        logger.info(f"Сохранение графиков для эксперимента {experiment_id}")
        return True

