# -*- coding: utf-8 -*-
"""
Модуль для управления датасетами - обновление и дополнение существующих датасетов
"""

import os
import json
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Union
from qgis.core import QgsVectorLayer, QgsFeatureRequest, QgsWkbTypes


class DatasetManager:
    """Класс для управления датасетами - обновление и дополнение"""
    
    def __init__(self, dataset_path: str):
        """
        Инициализация менеджера датасета
        
        :param dataset_path: Путь к датасету
        """
        self.dataset_path = dataset_path
        self.dataset_type = self._detect_dataset_type()
        self.metadata = None
        self.class_names = {}
        self.existing_images = set()
        
        if self.dataset_type:
            self._load_existing_dataset()
    
    def _detect_dataset_type(self) -> Optional[str]:
        """
        Определяет тип датасета (ndjson или yolo)
        
        :return: 'ndjson', 'yolo' или None
        """
        if os.path.exists(os.path.join(self.dataset_path, 'data.ndjson')):
            return 'ndjson'
        elif os.path.exists(os.path.join(self.dataset_path, 'dataset.yaml')):
            return 'yolo'
        return None
    
    def _load_existing_dataset(self):
        """Загружает информацию о существующем датасете"""
        if self.dataset_type == 'ndjson':
            self._load_ndjson_dataset()
        elif self.dataset_type == 'yolo':
            self._load_yolo_dataset()
    
    def _load_ndjson_dataset(self):
        """Загружает информацию из NDJSON датасета"""
        ndjson_path = os.path.join(self.dataset_path, 'data.ndjson')
        
        with open(ndjson_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                self.metadata = json.loads(first_line)
                self.class_names = self.metadata.get('class_names', {})
        
        # Собираем существующие изображения
        with open(ndjson_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('type') == 'image':
                        self.existing_images.add(data.get('file', ''))
    
    def _load_yolo_dataset(self):
        """Загружает информацию из YOLO датасета"""
        yaml_path = os.path.join(self.dataset_path, 'dataset.yaml')
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Парсим YAML файл (простая реализация)
        lines = content.split('\n')
        for line in lines:
            if line.startswith('names:'):
                continue
            elif ':' in line and not line.startswith('path') and not line.startswith('train') and not line.startswith('val') and not line.startswith('test'):
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        class_id = int(parts[0].strip())
                        class_name = parts[1].strip()
                        self.class_names[str(class_id)] = class_name
                    except ValueError:
                        continue
        
        # Собираем существующие изображения
        images_dir = os.path.join(self.dataset_path, 'images')
        if os.path.exists(images_dir):
            for split in ['train', 'val', 'test']:
                split_dir = os.path.join(images_dir, split)
                if os.path.exists(split_dir):
                    for file in os.listdir(split_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.existing_images.add(os.path.join('images', split, file).replace('\\', '/'))
    
    def can_update_dataset(self) -> Tuple[bool, str]:
        """
        Проверяет, можно ли обновить датасет
        
        :return: (можно_ли_обновить, сообщение_об_ошибке)
        """
        if not self.dataset_type:
            return False, "Не найден поддерживаемый датасет в указанной директории"
        
        if not os.path.exists(self.dataset_path):
            return False, "Директория датасета не существует"
        
        return True, "Датасет готов к обновлению"
    
    def get_dataset_info(self) -> Dict:
        """
        Возвращает информацию о датасете
        
        :return: Словарь с информацией о датасете
        """
        return {
            'type': self.dataset_type,
            'path': self.dataset_path,
            'metadata': self.metadata,
            'class_names': self.class_names,
            'existing_images_count': len(self.existing_images),
            'existing_images': list(self.existing_images)
        }
    
    def get_new_classes_from_layer(self, layer, class_field: str) -> List[str]:
        """
        Извлекает новые классы из векторного слоя
        
        :param layer: Векторный слой
        :param class_field: Поле с классами
        :return: Список новых классов
        """
        try:
            field_index = layer.fields().indexFromName(class_field)
            if field_index == -1:
                return []
            
            unique_values = layer.uniqueValues(field_index)
            all_classes = [str(val) for val in unique_values if val is not None and str(val).strip()]
            
            # Получаем существующие классы
            existing_classes = set(self.class_names.values())
            
            # Находим новые классы
            new_classes = [cls for cls in all_classes if cls not in existing_classes]
            
            return sorted(new_classes)
        except Exception as e:
            return []
    
    def create_backup(self) -> Tuple[bool, str]:
        """
        Создает резервную копию датасета
        
        :return: (успех, сообщение)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.dataset_path}_backup_{timestamp}"
            shutil.copytree(self.dataset_path, backup_path)
            return True, f"Резервная копия создана: {backup_path}"
        except Exception as e:
            return False, f"Ошибка создания резервной копии: {e}"
    
    def update_dataset_metadata(self, new_metadata: Dict) -> Tuple[bool, str]:
        """
        Обновляет метаданные датасета
        
        :param new_metadata: Новые метаданные
        :return: (успех, сообщение)
        """
        try:
            if self.dataset_type == 'ndjson':
                return self._update_ndjson_metadata(new_metadata)
            elif self.dataset_type == 'yolo':
                return self._update_yolo_metadata(new_metadata)
            return False, "Неизвестный тип датасета"
        except Exception as e:
            return False, f"Ошибка обновления метаданных: {e}"
    
    def _update_ndjson_metadata(self, new_metadata: Dict) -> Tuple[bool, str]:
        """Обновляет метаданные в NDJSON файле"""
        ndjson_path = os.path.join(self.dataset_path, 'data.ndjson')
        
        # Читаем все строки
        with open(ndjson_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Обновляем первую строку (метаданные)
        if lines:
            metadata = json.loads(lines[0])
            metadata.update(new_metadata)
            metadata['updated_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            lines[0] = json.dumps(metadata) + '\n'
        
        # Записываем обратно
        with open(ndjson_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        self.metadata = metadata
        return True, "Метаданные NDJSON обновлены"
    
    def _update_yolo_metadata(self, new_metadata: Dict) -> Tuple[bool, str]:
        """Обновляет метаданные в YOLO датасете"""
        # Для YOLO датасета создаем файл с метаданными
        metadata_path = os.path.join(self.dataset_path, 'metadata.json')
        
        metadata = {
            'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'updated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            **new_metadata
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.metadata = metadata
        return True, "Метаданные YOLO обновлены"
    
    def add_new_classes(self, new_classes: List[str]) -> Tuple[bool, str]:
        """
        Добавляет новые классы в датасет
        
        :param new_classes: Список новых классов
        :return: (успех, сообщение)
        """
        try:
            # Определяем следующий доступный ID
            existing_ids = [int(k) for k in self.class_names.keys() if k.isdigit()]
            next_id = max(existing_ids) + 1 if existing_ids else 0
            
            added_classes = {}
            for class_name in new_classes:
                if class_name not in self.class_names.values():
                    added_classes[str(next_id)] = class_name
                    self.class_names[str(next_id)] = class_name
                    next_id += 1
            
            if not added_classes:
                return True, "Все указанные классы уже существуют в датасете"
            
            # Обновляем файлы датасета
            if self.dataset_type == 'ndjson':
                self._update_ndjson_classes()
            elif self.dataset_type == 'yolo':
                self._update_yolo_classes()
            
            return True, f"Добавлены новые классы: {list(added_classes.values())}"
        except Exception as e:
            return False, f"Ошибка добавления классов: {e}"
    
    def _update_ndjson_classes(self):
        """Обновляет классы в NDJSON файле"""
        ndjson_path = os.path.join(self.dataset_path, 'data.ndjson')
        
        with open(ndjson_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if lines:
            metadata = json.loads(lines[0])
            metadata['class_names'] = self.class_names
            lines[0] = json.dumps(metadata) + '\n'
        
        with open(ndjson_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    def _update_yolo_classes(self):
        """Обновляет классы в YOLO датасете"""
        yaml_path = os.path.join(self.dataset_path, 'dataset.yaml')
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Находим секцию names и обновляем её
        lines = content.split('\n')
        new_lines = []
        in_names_section = False
        
        for line in lines:
            if line.startswith('names:'):
                new_lines.append(line)
                in_names_section = True
                # Добавляем все классы
                for class_id, class_name in sorted(self.class_names.items(), key=lambda x: int(x[0])):
                    new_lines.append(f"  {class_id}: {class_name}")
            elif in_names_section and line.strip() and not line.startswith(' '):
                in_names_section = False
                new_lines.append(line)
            elif not in_names_section:
                new_lines.append(line)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
    
    def append_new_data(self, 
                       intersected_layer: QgsVectorLayer,
                       grid_layer: QgsVectorLayer,
                       class_field: str,
                       image_format: str,
                       splits: Dict,
                       delete_void: bool = True,
                       progress_reporter=None,
                       metadata: Dict = None) -> Tuple[bool, str]:
        """
        Добавляет новые данные в существующий датасет
        
        :param intersected_layer: Слой с пересечениями
        :param grid_layer: Слой сетки
        :param class_field: Поле с классами
        :param image_format: Формат изображений
        :param splits: Разбивка на train/val/test
        :param delete_void: Удалять ли пустые изображения
        :param progress_reporter: Репортер прогресса
        :param metadata: Дополнительные метаданные
        :return: (успех, сообщение)
        """
        try:
            # Объединяем существующие метаданные с новыми
            combined_metadata = self.metadata.copy() if self.metadata else {}
            if metadata:
                combined_metadata.update(metadata)
            
            if self.dataset_type == 'ndjson':
                return self._append_ndjson_data(intersected_layer, grid_layer, class_field, 
                                              image_format, splits, delete_void, progress_reporter, combined_metadata)
            elif self.dataset_type == 'yolo':
                return self._append_yolo_data(intersected_layer, grid_layer, class_field, 
                                            image_format, splits, delete_void, progress_reporter, combined_metadata)
            return False, "Неизвестный тип датасета"
        except Exception as e:
            return False, f"Ошибка добавления данных: {e}"
    
    def _append_ndjson_data(self, intersected_layer, grid_layer, class_field, 
                           image_format, splits, delete_void, progress_reporter, metadata):
        """Добавляет данные в NDJSON датасет"""
        from .dataset_formatter import format_yolo_dataset
        
        # Создаем временную директорию для новых данных
        temp_dir = os.path.join(self.dataset_path, 'temp_new_data')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Формируем новые данные
            success, error_msg = format_yolo_dataset(
                intersected_layer=intersected_layer,
                grid_layer=grid_layer,
                class_field=class_field,
                output_dir=temp_dir,
                image_format=image_format,
                image_width=640,  # Будет обновлено из метаданных
                image_height=640,
                splits=splits,
                metadata=metadata,
                delete_void=delete_void,
                progress_reporter=progress_reporter
            )
            
            if not success:
                return False, error_msg
            
            # Объединяем с существующими данными
            self._merge_ndjson_data(temp_dir)
            
            return True, "Новые данные успешно добавлены в NDJSON датасет"
        finally:
            # Очищаем временную директорию
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _append_yolo_data(self, intersected_layer, grid_layer, class_field, 
                         image_format, splits, delete_void, progress_reporter, metadata):
        """Добавляет данные в YOLO датасет"""
        from .dataset_formatter_yolo import save_yolo_native_dataset
        
        # Создаем временную директорию для новых данных
        temp_dir = os.path.join(self.dataset_path, 'temp_new_data')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Формируем новые данные
            success, error_msg = save_yolo_native_dataset(
                intersected_layer=intersected_layer,
                grid_layer=grid_layer,
                class_field=class_field,
                output_dir=temp_dir,
                image_format=image_format,
                splits=splits,
                metadata=metadata,
                delete_void=delete_void,
                progress_reporter=progress_reporter
            )
            
            if not success:
                return False, error_msg
            
            # Объединяем с существующими данными
            self._merge_yolo_data(temp_dir)
            
            return True, "Новые данные успешно добавлены в YOLO датасет"
        finally:
            # Очищаем временную директорию
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _merge_ndjson_data(self, temp_dir: str):
        """Объединяет новые NDJSON данные с существующими"""
        existing_ndjson = os.path.join(self.dataset_path, 'data.ndjson')
        new_ndjson = os.path.join(temp_dir, 'data.ndjson')
        
        # Читаем существующие данные
        existing_lines = []
        existing_metadata = None
        with open(existing_ndjson, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                existing_metadata = json.loads(lines[0])  # Метаданные
                existing_lines.extend(lines[1:])  # Данные изображений
        
        # Читаем новые данные
        new_lines = []
        new_metadata = None
        with open(new_ndjson, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                new_metadata = json.loads(lines[0])  # Метаданные
                new_lines.extend(lines[1:])  # Данные изображений
        
        # Объединяем метаданные
        if existing_metadata and new_metadata:
            # Обновляем метаданные
            existing_metadata.update(new_metadata)
            existing_metadata['updated_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            # Обновляем классы
            if 'class_names' in new_metadata:
                existing_metadata['class_names'].update(new_metadata['class_names'])
        
        # Записываем объединенные данные
        with open(existing_ndjson, 'w', encoding='utf-8') as f:
            if existing_metadata:
                f.write(json.dumps(existing_metadata) + '\n')
            f.writelines(existing_lines)
            f.writelines(new_lines)
        
        # Перемещаем новые изображения
        new_images_dir = os.path.join(temp_dir, 'images')
        existing_images_dir = os.path.join(self.dataset_path, 'images')
        
        if os.path.exists(new_images_dir):
            for split in os.listdir(new_images_dir):
                split_dir = os.path.join(new_images_dir, split)
                if os.path.isdir(split_dir):
                    dest_split_dir = os.path.join(existing_images_dir, split)
                    os.makedirs(dest_split_dir, exist_ok=True)
                    
                    for file in os.listdir(split_dir):
                        src_file = os.path.join(split_dir, file)
                        dest_file = os.path.join(dest_split_dir, file)
                        shutil.move(src_file, dest_file)
    
    def _merge_yolo_data(self, temp_dir: str):
        """Объединяет новые YOLO данные с существующими"""
        # Перемещаем новые изображения и метки
        for data_type in ['images', 'labels']:
            new_data_dir = os.path.join(temp_dir, data_type)
            existing_data_dir = os.path.join(self.dataset_path, data_type)
            
            if os.path.exists(new_data_dir):
                for split in os.listdir(new_data_dir):
                    split_dir = os.path.join(new_data_dir, split)
                    if os.path.isdir(split_dir):
                        dest_split_dir = os.path.join(existing_data_dir, split)
                        os.makedirs(dest_split_dir, exist_ok=True)
                        
                        for file in os.listdir(split_dir):
                            src_file = os.path.join(split_dir, file)
                            dest_file = os.path.join(dest_split_dir, file)
                            shutil.move(src_file, dest_file)
        
        # Обновляем dataset.yaml с новыми классами
        self._update_yolo_classes()
        
        # Обновляем метаданные
        if self.metadata:
            metadata_path = os.path.join(self.dataset_path, 'metadata.json')
            self.metadata['updated_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
