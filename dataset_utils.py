# -*- coding: utf-8 -*-
"""
Утилиты для работы с датасетами
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from qgis.PyQt.QtWidgets import QMessageBox, QFileDialog, QWidget


class DatasetUtils:
    """Утилиты для работы с датасетами"""
    
    @staticmethod
    def validate_dataset_path(dataset_path: str) -> Tuple[bool, str]:
        """
        Проверяет валидность пути к датасету
        
        :param dataset_path: Путь к датасету
        :return: (валиден_ли, сообщение_об_ошибке)
        """
        if not dataset_path:
            return False, "Путь к датасету не указан"
        
        if not os.path.exists(dataset_path):
            return False, "Директория датасета не существует"
        
        if not os.path.isdir(dataset_path):
            return False, "Указанный путь не является директорией"
        
        # Проверяем наличие файлов датасета
        has_ndjson = os.path.exists(os.path.join(dataset_path, 'data.ndjson'))
        has_yolo = os.path.exists(os.path.join(dataset_path, 'dataset.yaml'))
        
        if not has_ndjson and not has_yolo:
            return False, "В указанной директории не найден поддерживаемый датасет"
        
        return True, "Датасет валиден"
    
    @staticmethod
    def get_dataset_info(dataset_path: str) -> Dict:
        """
        Получает информацию о датасете
        
        :param dataset_path: Путь к датасету
        :return: Словарь с информацией о датасете
        """
        info = {
            'path': dataset_path,
            'type': None,
            'metadata': None,
            'class_names': {},
            'images_count': 0,
            'splits': {},
            'size_mb': 0,
            'created_at': None,
            'updated_at': None
        }
        
        try:
            # Определяем тип датасета
            if os.path.exists(os.path.join(dataset_path, 'data.ndjson')):
                info['type'] = 'ndjson'
                info.update(DatasetUtils._get_ndjson_info(dataset_path))
            elif os.path.exists(os.path.join(dataset_path, 'dataset.yaml')):
                info['type'] = 'yolo'
                info.update(DatasetUtils._get_yolo_info(dataset_path))
            
            # Размер датасета
            info['size_mb'] = DatasetUtils._get_directory_size(dataset_path) / (1024 * 1024)
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    @staticmethod
    def _get_ndjson_info(dataset_path: str) -> Dict:
        """Получает информацию из NDJSON датасета"""
        info = {}
        ndjson_path = os.path.join(dataset_path, 'data.ndjson')
        
        try:
            with open(ndjson_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    metadata = json.loads(first_line)
                    info['metadata'] = metadata
                    info['class_names'] = metadata.get('class_names', {})
                    info['created_at'] = metadata.get('created_at')
                    info['updated_at'] = metadata.get('updated_at')
            
            # Подсчитываем изображения
            with open(ndjson_path, 'r', encoding='utf-8') as f:
                image_count = 0
                splits = {'train': 0, 'val': 0, 'test': 0}
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get('type') == 'image':
                            image_count += 1
                            split = data.get('split', 'train')
                            if split in splits:
                                splits[split] += 1
                
                info['images_count'] = image_count
                info['splits'] = splits
                
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    @staticmethod
    def _get_yolo_info(dataset_path: str) -> Dict:
        """Получает информацию из YOLO датасета"""
        info = {}
        yaml_path = os.path.join(dataset_path, 'dataset.yaml')
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Парсим YAML файл
            lines = content.split('\n')
            class_names = {}
            in_names_section = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('names:'):
                    in_names_section = True
                    continue
                elif in_names_section and line and not line.startswith('#'):
                    if ':' in line:
                        parts = line.split(':')
                        if len(parts) == 2:
                            try:
                                class_id = int(parts[0].strip())
                                class_name = parts[1].strip()
                                class_names[str(class_id)] = class_name
                            except ValueError:
                                continue
                elif in_names_section and not line:
                    in_names_section = False
            
            info['class_names'] = class_names
            
            # Подсчитываем изображения
            images_dir = os.path.join(dataset_path, 'images')
            image_count = 0
            splits = {'train': 0, 'val': 0, 'test': 0}
            
            if os.path.exists(images_dir):
                for split in ['train', 'val', 'test']:
                    split_dir = os.path.join(images_dir, split)
                    if os.path.exists(split_dir):
                        count = len([f for f in os.listdir(split_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        splits[split] = count
                        image_count += count
            
            info['images_count'] = image_count
            info['splits'] = splits
            
            # Проверяем наличие файла метаданных
            metadata_path = os.path.join(dataset_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    info['metadata'] = metadata
                    info['created_at'] = metadata.get('created_at')
                    info['updated_at'] = metadata.get('updated_at')
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    @staticmethod
    def _get_directory_size(directory: str) -> int:
        """Вычисляет размер директории в байтах"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size
    
    @staticmethod
    def create_dataset_backup(dataset_path: str, backup_dir: str = None) -> Tuple[bool, str]:
        """
        Создает резервную копию датасета
        
        :param dataset_path: Путь к датасету
        :param backup_dir: Директория для резервных копий (опционально)
        :return: (успех, сообщение)
        """
        try:
            if not backup_dir:
                backup_dir = os.path.dirname(dataset_path)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = os.path.basename(dataset_path)
            backup_path = os.path.join(backup_dir, f"{dataset_name}_backup_{timestamp}")
            
            shutil.copytree(dataset_path, backup_path)
            return True, f"Резервная копия создана: {backup_path}"
            
        except Exception as e:
            return False, f"Ошибка создания резервной копии: {e}"
    
    @staticmethod
    def restore_dataset_from_backup(backup_path: str, restore_path: str) -> Tuple[bool, str]:
        """
        Восстанавливает датасет из резервной копии
        
        :param backup_path: Путь к резервной копии
        :param restore_path: Путь для восстановления
        :return: (успех, сообщение)
        """
        try:
            if not os.path.exists(backup_path):
                return False, "Резервная копия не найдена"
            
            if os.path.exists(restore_path):
                shutil.rmtree(restore_path)
            
            shutil.copytree(backup_path, restore_path)
            return True, f"Датасет восстановлен из резервной копии: {restore_path}"
            
        except Exception as e:
            return False, f"Ошибка восстановления: {e}"
    
    @staticmethod
    def list_available_datasets(directory: str) -> List[Dict]:
        """
        Списывает доступные датасеты в директории
        
        :param directory: Директория для поиска
        :return: Список словарей с информацией о датасетах
        """
        datasets = []
        
        try:
            if not os.path.exists(directory):
                return datasets
            
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    # Проверяем, является ли это датасетом
                    is_valid, _ = DatasetUtils.validate_dataset_path(item_path)
                    if is_valid:
                        info = DatasetUtils.get_dataset_info(item_path)
                        datasets.append(info)
            
            # Сортируем по дате обновления
            datasets.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
            
        except Exception as e:
            print(f"Ошибка при поиске датасетов: {e}")
        
        return datasets
    
    @staticmethod
    def cleanup_dataset(dataset_path: str) -> Tuple[bool, str]:
        """
        Очищает датасет от временных файлов
        
        :param dataset_path: Путь к датасету
        :return: (успех, сообщение)
        """
        try:
            cleaned_files = []
            
            # Удаляем временные файлы
            temp_patterns = ['*.tmp', '*.temp', 'temp_*', '*.log']
            
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file.startswith(pattern.replace('*', '')) or file.endswith(pattern.replace('*', '')) 
                           for pattern in temp_patterns):
                        os.remove(file_path)
                        cleaned_files.append(file)
                
                # Удаляем пустые временные директории
                for dir_name in dirs[:]:  # Копия списка для безопасного удаления
                    dir_path = os.path.join(root, dir_name)
                    if dir_name.startswith('temp_') and not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        cleaned_files.append(f"{dir_name}/")
                        dirs.remove(dir_name)
            
            if cleaned_files:
                return True, f"Очищено файлов: {len(cleaned_files)}"
            else:
                return True, "Временные файлы не найдены"
                
        except Exception as e:
            return False, f"Ошибка очистки: {e}"
    
    @staticmethod
    def export_dataset_info(dataset_path: str, output_file: str) -> Tuple[bool, str]:
        """
        Экспортирует информацию о датасете в файл
        
        :param dataset_path: Путь к датасету
        :param output_file: Путь к выходному файлу
        :return: (успех, сообщение)
        """
        try:
            info = DatasetUtils.get_dataset_info(dataset_path)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            
            return True, f"Информация экспортирована в: {output_file}"
            
        except Exception as e:
            return False, f"Ошибка экспорта: {e}"
