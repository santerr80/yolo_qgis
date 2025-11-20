# -*- coding: utf-8 -*-
"""
Менеджер для хранения истории использованных путей
"""

import os
import logging
from typing import List, Optional
from qgis.core import QgsSettings

logger = logging.getLogger(__name__)


class PathHistoryManager:
    """Менеджер для управления историей путей к файлам и директориям"""

    # Ключи для QSettings
    KEY_DATASET_PATHS = "yolo_qgis/paths/datasets"
    KEY_MODEL_PATHS = "yolo_qgis/paths/models"
    KEY_SAVE_DIR_PATHS = "yolo_qgis/paths/save_dirs"
    KEY_DATASET_CREATION_PATHS = "yolo_qgis/paths/dataset_creation"

    # Максимальное количество путей в истории
    MAX_HISTORY_SIZE = 20

    def __init__(self):
        """Инициализация менеджера истории путей"""
        self.settings = QgsSettings()

    def add_dataset_path(self, path: str) -> bool:
        """
        Добавляет путь к датасету в историю

        :param path: Путь к датасету
        :return: True если успешно
        """
        return self._add_path(self.KEY_DATASET_PATHS, path)

    def add_model_path(self, path: str) -> bool:
        """
        Добавляет путь к модели в историю

        :param path: Путь к модели
        :return: True если успешно
        """
        return self._add_path(self.KEY_MODEL_PATHS, path)

    def add_save_dir_path(self, path: str) -> bool:
        """
        Добавляет путь к директории сохранения в историю

        :param path: Путь к директории
        :return: True если успешно
        """
        return self._add_path(self.KEY_SAVE_DIR_PATHS, path)

    def add_dataset_creation_path(self, path: str) -> bool:
        """
        Добавляет путь для создания датасета в историю

        :param path: Путь к директории
        :return: True если успешно
        """
        return self._add_path(self.KEY_DATASET_CREATION_PATHS, path)

    def _add_path(self, key: str, path: str) -> bool:
        """
        Добавляет путь в историю

        :param key: Ключ для QSettings
        :param path: Путь для добавления
        :return: True если успешно
        """
        try:
            if not path or not os.path.exists(path):
                return False

            # Нормализуем путь
            normalized_path = os.path.normpath(path)

            # Получаем текущую историю
            history = self.get_paths(key)

            # Удаляем путь, если он уже есть (чтобы переместить в начало)
            if normalized_path in history:
                history.remove(normalized_path)

            # Добавляем в начало
            history.insert(0, normalized_path)

            # Ограничиваем размер истории
            history = history[: self.MAX_HISTORY_SIZE]

            # Сохраняем
            self.settings.setValue(key, history)

            return True

        except Exception as e:
            logger.error(f"Ошибка добавления пути в историю: {e}", exc_info=True)
            return False

    def get_dataset_paths(self) -> List[str]:
        """
        Получает историю путей к датасетам

        :return: Список путей
        """
        return self.get_paths(self.KEY_DATASET_PATHS)

    def get_model_paths(self) -> List[str]:
        """
        Получает историю путей к моделям

        :return: Список путей
        """
        return self.get_paths(self.KEY_MODEL_PATHS)

    def get_save_dir_paths(self) -> List[str]:
        """
        Получает историю путей к директориям сохранения

        :return: Список путей
        """
        return self.get_paths(self.KEY_SAVE_DIR_PATHS)

    def get_dataset_creation_paths(self) -> List[str]:
        """
        Получает историю путей для создания датасетов

        :return: Список путей
        """
        return self.get_paths(self.KEY_DATASET_CREATION_PATHS)

    def get_paths(self, key: str) -> List[str]:
        """
        Получает историю путей по ключу

        :param key: Ключ для QSettings
        :return: Список путей
        """
        try:
            paths = self.settings.value(key, [])
            if not isinstance(paths, list):
                return []

            # Фильтруем только существующие пути
            valid_paths = []
            for path in paths:
                if path and os.path.exists(path):
                    valid_paths.append(path)

            return valid_paths

        except Exception as e:
            logger.error(f"Ошибка получения истории путей: {e}", exc_info=True)
            return []

    def clear_dataset_paths(self) -> bool:
        """
        Очищает историю путей к датасетам

        :return: True если успешно
        """
        return self._clear_paths(self.KEY_DATASET_PATHS)

    def clear_model_paths(self) -> bool:
        """
        Очищает историю путей к моделям

        :return: True если успешно
        """
        return self._clear_paths(self.KEY_MODEL_PATHS)

    def clear_save_dir_paths(self) -> bool:
        """
        Очищает историю путей к директориям сохранения

        :return: True если успешно
        """
        return self._clear_paths(self.KEY_SAVE_DIR_PATHS)

    def clear_dataset_creation_paths(self) -> bool:
        """
        Очищает историю путей для создания датасетов

        :return: True если успешно
        """
        return self._clear_paths(self.KEY_DATASET_CREATION_PATHS)

    def _clear_paths(self, key: str) -> bool:
        """
        Очищает историю путей

        :param key: Ключ для QSettings
        :return: True если успешно
        """
        try:
            self.settings.remove(key)
            return True
        except Exception as e:
            logger.error(f"Ошибка очистки истории путей: {e}", exc_info=True)
            return False

    def get_last_dataset_path(self) -> Optional[str]:
        """
        Получает последний использованный путь к датасету

        :return: Путь или None
        """
        paths = self.get_dataset_paths()
        return paths[0] if paths else None

    def get_last_model_path(self) -> Optional[str]:
        """
        Получает последний использованный путь к модели

        :return: Путь или None
        """
        paths = self.get_model_paths()
        return paths[0] if paths else None

    def get_last_save_dir_path(self) -> Optional[str]:
        """
        Получает последний использованный путь к директории сохранения

        :return: Путь или None
        """
        paths = self.get_save_dir_paths()
        return paths[0] if paths else None

    def get_last_dataset_creation_path(self) -> Optional[str]:
        """
        Получает последний использованный путь для создания датасета

        :return: Путь или None
        """
        paths = self.get_dataset_creation_paths()
        return paths[0] if paths else None
