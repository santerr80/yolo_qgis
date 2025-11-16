# -*- coding: utf-8 -*-
"""
Менеджер истории путей для хранения ранее использованных путей к датасетам и проектам
"""

import os
import logging
from typing import List, Optional
from qgis.PyQt.QtCore import QSettings

logger = logging.getLogger(__name__)


class PathHistoryManager:
    """Менеджер для хранения и управления историей путей"""
    
    def __init__(self, max_history: int = 10):
        """Инициализация менеджера истории путей
        
        Args:
            max_history: Максимальное количество путей в истории
        """
        self.max_history = max_history
        self.settings = QSettings()
        self.dataset_paths_key = "yolo_qgis/history/dataset_paths"
        self.project_paths_key = "yolo_qgis/history/project_paths"
    
    def add_dataset_path(self, path: str) -> None:
        """Добавляет путь к датасету в историю
        
        Args:
            path: Путь к датасету
        """
        if not path or not os.path.exists(path):
            return
        
        # Нормализуем путь
        normalized_path = os.path.normpath(path)
        
        # Получаем текущую историю
        history = self.get_dataset_paths()
        
        # Удаляем путь, если он уже есть (чтобы переместить его в начало)
        if normalized_path in history:
            history.remove(normalized_path)
        
        # Добавляем в начало
        history.insert(0, normalized_path)
        
        # Ограничиваем размер истории
        history = history[:self.max_history]
        
        # Сохраняем
        self.settings.setValue(self.dataset_paths_key, history)
        self.settings.sync()
    
    def add_project_path(self, path: str) -> None:
        """Добавляет путь к проекту в историю
        
        Args:
            path: Путь к директории проекта
        """
        if not path:
            return
        
        # Нормализуем путь
        normalized_path = os.path.normpath(path)
        
        # Создаем директорию, если её нет
        try:
            os.makedirs(normalized_path, exist_ok=True)
        except Exception as e:
            logger.warning(f"Не удалось создать директорию {normalized_path}: {e}")
            return
        
        # Получаем текущую историю
        history = self.get_project_paths()
        
        # Удаляем путь, если он уже есть (чтобы переместить его в начало)
        if normalized_path in history:
            history.remove(normalized_path)
        
        # Добавляем в начало
        history.insert(0, normalized_path)
        
        # Ограничиваем размер истории
        history = history[:self.max_history]
        
        # Сохраняем
        self.settings.setValue(self.project_paths_key, history)
        self.settings.sync()
    
    def get_dataset_paths(self) -> List[str]:
        """Получает список путей к датасетам из истории
        
        Returns:
            Список путей к датасетам
        """
        history = self.settings.value(self.dataset_paths_key, [])
        if not isinstance(history, list):
            return []
        
        # Фильтруем несуществующие пути
        valid_paths = [path for path in history if os.path.exists(path)]
        
        # Если есть невалидные пути, обновляем историю
        if len(valid_paths) != len(history):
            self.settings.setValue(self.dataset_paths_key, valid_paths)
            self.settings.sync()
        
        return valid_paths
    
    def get_project_paths(self) -> List[str]:
        """Получает список путей к проектам из истории
        
        Returns:
            Список путей к проектам
        """
        history = self.settings.value(self.project_paths_key, [])
        if not isinstance(history, list):
            return []
        
        # Фильтруем несуществующие пути (для проектов это не критично, но проверим)
        valid_paths = []
        for path in history:
            try:
                # Проверяем, можем ли мы создать директорию или она существует
                if os.path.exists(path) or os.path.exists(os.path.dirname(path)):
                    valid_paths.append(path)
            except Exception:
                continue
        
        # Если есть невалидные пути, обновляем историю
        if len(valid_paths) != len(history):
            self.settings.setValue(self.project_paths_key, valid_paths)
            self.settings.sync()
        
        return valid_paths
    
    def clear_dataset_history(self) -> None:
        """Очищает историю путей к датасетам"""
        self.settings.setValue(self.dataset_paths_key, [])
        self.settings.sync()
    
    def clear_project_history(self) -> None:
        """Очищает историю путей к проектам"""
        self.settings.setValue(self.project_paths_key, [])
        self.settings.sync()
    
    def remove_dataset_path(self, path: str) -> None:
        """Удаляет путь из истории датасетов
        
        Args:
            path: Путь для удаления
        """
        history = self.get_dataset_paths()
        normalized_path = os.path.normpath(path)
        if normalized_path in history:
            history.remove(normalized_path)
            self.settings.setValue(self.dataset_paths_key, history)
            self.settings.sync()
    
    def remove_project_path(self, path: str) -> None:
        """Удаляет путь из истории проектов
        
        Args:
            path: Путь для удаления
        """
        history = self.get_project_paths()
        normalized_path = os.path.normpath(path)
        if normalized_path in history:
            history.remove(normalized_path)
            self.settings.setValue(self.project_paths_key, history)
            self.settings.sync()

