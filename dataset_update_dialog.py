# -*- coding: utf-8 -*-
"""
Диалог для обновления и дополнения существующих датасетов
"""

import os
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QProgressBar
from qgis.core import QgsMapLayerProxyModel, QgsProject
from qgis.gui import QgsMapLayerComboBox, QgsFieldComboBox

from .dataset_manager import DatasetManager
from .grid_creator import create_grid_layer
from .intersection import perform_intersection
from .map_exporter import export_views
from .processing_utils import ProgressReporter


class DatasetUpdateDialog(QtWidgets.QDialog):
    """Диалог для обновления и дополнения датасетов"""
    
    def __init__(self, parent=None):
        super(DatasetUpdateDialog, self).__init__(parent)
        self.setupUi()
        self.setup_connections()
        self.dataset_manager = None
        
    def setupUi(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle("Обновление и дополнение датасетов")
        self.setModal(True)
        self.resize(600, 500)
        
        # Основной layout
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Создаем область прокрутки
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)  # Убираем рамку
        scroll_area.setMinimumHeight(400)  # Минимальная высота для прокрутки
        
        # Создаем виджет для содержимого
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)  # Отступы
        scroll_layout.setSpacing(10)  # Промежутки между элементами
        
        # Группа выбора датасета
        dataset_group = QtWidgets.QGroupBox("Выбор датасета для обновления")
        dataset_layout = QtWidgets.QVBoxLayout(dataset_group)
        
        # Путь к датасету
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(QtWidgets.QLabel("Путь к датасету:"))
        self.dataset_path_edit = QtWidgets.QLineEdit()
        self.dataset_path_edit.setPlaceholderText("Выберите директорию с датасетом")
        path_layout.addWidget(self.dataset_path_edit)
        self.browse_button = QtWidgets.QPushButton("Обзор...")
        path_layout.addWidget(self.browse_button)
        dataset_layout.addLayout(path_layout)
        
        # Кнопка загрузки датасета
        self.load_dataset_button = QtWidgets.QPushButton("Загрузить датасет")
        dataset_layout.addWidget(self.load_dataset_button)
        
        # Информация о датасете
        self.dataset_info_text = QtWidgets.QTextEdit()
        self.dataset_info_text.setMaximumHeight(100)
        self.dataset_info_text.setReadOnly(True)
        dataset_layout.addWidget(self.dataset_info_text)
        
        scroll_layout.addWidget(dataset_group)
        
        # Группа настроек обновления
        update_group = QtWidgets.QGroupBox("Настройки обновления")
        update_layout = QtWidgets.QVBoxLayout(update_group)
        
        # Чекбоксы для выбора операций
        self.update_metadata_check = QtWidgets.QCheckBox("Обновить метаданные")
        self.add_classes_check = QtWidgets.QCheckBox("Добавить новые классы")
        self.add_data_check = QtWidgets.QCheckBox("Добавить новые данные")
        self.create_backup_check = QtWidgets.QCheckBox("Создать резервную копию")
        self.create_backup_check.setChecked(True)
        
        update_layout.addWidget(self.update_metadata_check)
        update_layout.addWidget(self.add_classes_check)
        update_layout.addWidget(self.add_data_check)
        update_layout.addWidget(self.create_backup_check)
        
        scroll_layout.addWidget(update_group)
        
        # Группа метаданных
        metadata_group = QtWidgets.QGroupBox("Метаданные")
        metadata_layout = QtWidgets.QFormLayout(metadata_group)
        
        self.name_edit = QtWidgets.QLineEdit()
        self.description_edit = QtWidgets.QLineEdit()
        self.url_edit = QtWidgets.QLineEdit()
        
        metadata_layout.addRow("Название:", self.name_edit)
        metadata_layout.addRow("Описание:", self.description_edit)
        metadata_layout.addRow("URL:", self.url_edit)
        
        scroll_layout.addWidget(metadata_group)
        
        # Группа новых классов
        classes_group = QtWidgets.QGroupBox("Новые классы")
        classes_layout = QtWidgets.QVBoxLayout(classes_group)
        
        # Автоматическое извлечение классов
        self.auto_extract_check = QtWidgets.QCheckBox("Автоматически извлекать новые классы из векторного слоя")
        self.auto_extract_check.setChecked(True)
        classes_layout.addWidget(self.auto_extract_check)
        
        # Список найденных новых классов
        self.new_classes_list = QtWidgets.QListWidget()
        self.new_classes_list.setMaximumHeight(100)
        self.new_classes_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        classes_layout.addWidget(QtWidgets.QLabel("Найденные новые классы:"))
        classes_layout.addWidget(self.new_classes_list)
        
        # Кнопка обновления списка классов
        self.refresh_classes_button = QtWidgets.QPushButton("Обновить список классов")
        classes_layout.addWidget(self.refresh_classes_button)
        
        # Ручной ввод классов (как резервный вариант)
        self.manual_classes_edit = QtWidgets.QLineEdit()
        self.manual_classes_edit.setPlaceholderText("Или введите новые классы вручную через запятую")
        self.manual_classes_edit.setEnabled(False)
        classes_layout.addWidget(QtWidgets.QLabel("Ручной ввод:"))
        classes_layout.addWidget(self.manual_classes_edit)
        
        scroll_layout.addWidget(classes_group)
        
        # Группа новых данных
        data_group = QtWidgets.QGroupBox("Новые данные")
        data_layout = QtWidgets.QFormLayout(data_group)
        
        # Тип задачи
        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.addItems(['detect', 'segment'])
        self.task_combo.setCurrentText('detect')
        data_layout.addRow("Тип задачи:", self.task_combo)
        
        # Растровый слой
        self.raster_layer_combo = QgsMapLayerComboBox()
        self.raster_layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        data_layout.addRow("Растровый слой:", self.raster_layer_combo)
        
        # Слой объектов
        self.objects_layer_combo = QgsMapLayerComboBox()
        self.objects_layer_combo.setFilters(QgsMapLayerProxyModel.VectorLayer)
        data_layout.addRow("Слой объектов:", self.objects_layer_combo)
        
        # Поле классов
        self.class_field_combo = QgsFieldComboBox()
        data_layout.addRow("Поле классов:", self.class_field_combo)
        
        # Разбивка данных
        splits_layout = QtWidgets.QHBoxLayout()
        splits_layout.addWidget(QtWidgets.QLabel("Train:"))
        self.train_spin = QtWidgets.QSpinBox()
        self.train_spin.setRange(1, 99)
        self.train_spin.setValue(80)
        splits_layout.addWidget(self.train_spin)
        
        splits_layout.addWidget(QtWidgets.QLabel("Val:"))
        self.val_spin = QtWidgets.QSpinBox()
        self.val_spin.setRange(1, 99)
        self.val_spin.setValue(20)
        splits_layout.addWidget(self.val_spin)
        
        splits_layout.addWidget(QtWidgets.QLabel("Test:"))
        self.test_spin = QtWidgets.QSpinBox()
        self.test_spin.setRange(0, 99)
        self.test_spin.setValue(0)
        splits_layout.addWidget(self.test_spin)
        
        data_layout.addRow("Разбивка:", splits_layout)
        
        # Настройки экспорта изображений
        export_layout = QtWidgets.QFormLayout()
        
        # Размеры изображений
        size_layout = QtWidgets.QHBoxLayout()
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(64, 4096)
        self.width_spin.setValue(640)
        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(64, 4096)
        self.height_spin.setValue(640)
        size_layout.addWidget(QtWidgets.QLabel("Ширина:"))
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QtWidgets.QLabel("Высота:"))
        size_layout.addWidget(self.height_spin)
        export_layout.addRow("Размеры (px):", size_layout)
        
        # DPI
        self.dpi_combo = QtWidgets.QComboBox()
        self.dpi_combo.addItems(['96', '150', '256', '300'])
        self.dpi_combo.setCurrentText('256')
        export_layout.addRow("DPI:", self.dpi_combo)
        
        # Формат изображения
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(['png', 'jpg', 'jpeg'])
        self.format_combo.setCurrentText('png')
        export_layout.addRow("Формат:", self.format_combo)
        
        data_layout.addRow("Настройки экспорта:", export_layout)
        
        scroll_layout.addWidget(data_group)
        
        # Устанавливаем содержимое в область прокрутки
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Прогресс бар (вне области прокрутки)
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        
        # Кнопки (вне области прокрутки)
        button_layout = QtWidgets.QHBoxLayout()
        self.update_button = QtWidgets.QPushButton("Обновить датасет")
        self.cancel_button = QtWidgets.QPushButton("Отмена")
        
        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(button_layout)
        
        # Обновляем список слоев
        self.update_layers_list()
        
        # Инициализируем поле классов
        self.update_class_fields()
        
    def setup_connections(self):
        """Настройка соединений сигналов и слотов"""
        self.browse_button.clicked.connect(self.browse_dataset_path)
        self.load_dataset_button.clicked.connect(self.load_dataset)
        self.update_button.clicked.connect(self.update_dataset)
        self.cancel_button.clicked.connect(self.reject)
        
        self.objects_layer_combo.layerChanged.connect(self.update_class_fields)
        self.objects_layer_combo.layerChanged.connect(self.extract_new_classes)
        self.class_field_combo.fieldChanged.connect(self.extract_new_classes)
        self.refresh_classes_button.clicked.connect(self.extract_new_classes)
        self.auto_extract_check.toggled.connect(self.toggle_manual_input)
        
        # Соединения для автоматического пересчета разбивки
        self.train_spin.valueChanged.connect(self.update_splits)
        self.val_spin.valueChanged.connect(self.update_splits)
        self.test_spin.valueChanged.connect(self.update_splits)
        
    def update_layers_list(self):
        """Обновляет список доступных слоев"""
        # QgsMapLayerComboBox автоматически обновляется
        pass
    
    def update_class_fields(self):
        """Обновляет список полей классов при смене слоя"""
        layer = self.objects_layer_combo.currentLayer()
        self.class_field_combo.setLayer(layer)
    
    def toggle_manual_input(self, checked):
        """Переключает режим ввода классов"""
        self.manual_classes_edit.setEnabled(not checked)
        if checked:
            self.extract_new_classes()
    
    def extract_new_classes(self):
        """Извлекает новые классы из векторного слоя"""
        if not self.auto_extract_check.isChecked():
            return
        
        layer = self.objects_layer_combo.currentLayer()
        class_field = self.class_field_combo.currentField()
        
        if not layer or not class_field:
            self.new_classes_list.clear()
            return
        
        if not self.dataset_manager:
            self.new_classes_list.clear()
            return
        
        try:
            # Получаем уникальные значения из поля классов
            field_index = layer.fields().indexFromName(class_field)
            if field_index == -1:
                self.new_classes_list.clear()
                return
            
            unique_values = layer.uniqueValues(field_index)
            all_classes = [str(val) for val in unique_values if val is not None and str(val).strip()]
            
            # Получаем существующие классы из датасета
            existing_classes = set(self.dataset_manager.class_names.values())
            
            # Находим новые классы
            new_classes = [cls for cls in all_classes if cls not in existing_classes]
            
            # Обновляем список
            self.new_classes_list.clear()
            for cls in sorted(new_classes):
                item = QtWidgets.QListWidgetItem(cls)
                item.setSelected(True)  # По умолчанию все новые классы выбраны
                self.new_classes_list.addItem(item)
            
            # Обновляем информацию о количестве
            if new_classes:
                self.new_classes_list.setToolTip(f"Найдено {len(new_classes)} новых классов")
            else:
                self.new_classes_list.setToolTip("Новых классов не найдено")
                
        except Exception as e:
            self.new_classes_list.clear()
            self.new_classes_list.setToolTip(f"Ошибка извлечения классов: {e}")
    
    def get_selected_new_classes(self):
        """Возвращает список выбранных новых классов"""
        if self.auto_extract_check.isChecked():
            # Получаем выбранные классы из списка
            selected_classes = []
            for i in range(self.new_classes_list.count()):
                item = self.new_classes_list.item(i)
                if item.isSelected():
                    selected_classes.append(item.text())
            return selected_classes
        else:
            # Получаем классы из ручного ввода
            manual_text = self.manual_classes_edit.text().strip()
            if manual_text:
                return [cls.strip() for cls in manual_text.split(',') if cls.strip()]
            return []
    
    def update_splits(self):
        """Автоматически пересчитывает разбивку train/val/test"""
        sender = self.sender()
        if not sender:
            return
            
        # Блокируем сигналы для предотвращения рекурсии
        self.train_spin.blockSignals(True)
        self.val_spin.blockSignals(True)
        self.test_spin.blockSignals(True)
        
        try:
            train = self.train_spin.value()
            val = self.val_spin.value()
            test = self.test_spin.value()
            
            remainder = 100 - sender.value()
            
            if sender == self.train_spin:
                other_sum = val + test
                if other_sum == 0:
                    new_val, new_test = remainder, 0
                else:
                    ratio = val / other_sum
                    new_val = round(remainder * ratio)
                    new_test = remainder - new_val
                self.val_spin.setValue(new_val)
                self.test_spin.setValue(new_test)
                
            elif sender == self.val_spin:
                other_sum = train + test
                if other_sum == 0:
                    new_train, new_test = remainder, 0
                else:
                    ratio = train / other_sum
                    new_train = round(remainder * ratio)
                    new_test = remainder - new_train
                self.train_spin.setValue(new_train)
                self.test_spin.setValue(new_test)
                
            elif sender == self.test_spin:
                other_sum = train + val
                if other_sum == 0:
                    new_train, new_val = remainder, 0
                else:
                    ratio = train / other_sum
                    new_train = round(remainder * ratio)
                    new_val = remainder - new_train
                self.train_spin.setValue(new_train)
                self.val_spin.setValue(new_val)
        finally:
            self.train_spin.blockSignals(False)
            self.val_spin.blockSignals(False)
            self.test_spin.blockSignals(False)
    
    def browse_dataset_path(self):
        """Открывает диалог выбора директории датасета"""
        path = QFileDialog.getExistingDirectory(
            self, 
            "Выберите директорию с датасетом",
            "",
            QFileDialog.ShowDirsOnly
        )
        if path:
            self.dataset_path_edit.setText(path)
    
    def load_dataset(self):
        """Загружает информацию о датасете"""
        dataset_path = self.dataset_path_edit.text().strip()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Укажите путь к датасету")
            return
        
        if not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Ошибка", "Указанная директория не существует")
            return
        
        try:
            self.dataset_manager = DatasetManager(dataset_path)
            can_update, message = self.dataset_manager.can_update_dataset()
            
            if not can_update:
                QMessageBox.warning(self, "Ошибка", message)
                return
            
            # Загружаем информацию о датасете
            info = self.dataset_manager.get_dataset_info()
            
            # Отображаем информацию
            info_text = f"Тип датасета: {info['type']}\n"
            info_text += f"Путь: {info['path']}\n"
            info_text += f"Количество существующих изображений: {info['existing_images_count']}\n"
            info_text += f"Классы: {', '.join(info['class_names'].values())}\n"
            
            if info['metadata']:
                info_text += f"Метаданные: {info['metadata']}\n"
            
            self.dataset_info_text.setText(info_text)
            
            # Заполняем поля метаданных
            if info['metadata']:
                self.name_edit.setText(info['metadata'].get('name', ''))
                self.description_edit.setText(info['metadata'].get('description', ''))
                self.url_edit.setText(info['metadata'].get('url', ''))
            
            # Автоматически извлекаем новые классы, если выбран слой
            self.extract_new_classes()
            
            QMessageBox.information(self, "Успех", "Датасет успешно загружен")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки датасета: {e}")
    
    def update_dataset(self):
        """Выполняет обновление датасета"""
        if not self.dataset_manager:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите датасет")
            return
        
        try:
            self.progress_bar.setValue(0)
            
            # Создание резервной копии
            if self.create_backup_check.isChecked():
                self.progress_bar.setValue(10)
                success, message = self.dataset_manager.create_backup()
                if not success:
                    QMessageBox.warning(self, "Предупреждение", f"Не удалось создать резервную копию: {message}")
                else:
                    QMessageBox.information(self, "Информация", message)
            
            # Обновление метаданных
            if self.update_metadata_check.isChecked():
                self.progress_bar.setValue(20)
                new_metadata = {
                    'name': self.name_edit.text().strip(),
                    'description': self.description_edit.text().strip(),
                    'url': self.url_edit.text().strip()
                }
                success, message = self.dataset_manager.update_dataset_metadata(new_metadata)
                if not success:
                    QMessageBox.warning(self, "Ошибка", f"Ошибка обновления метаданных: {message}")
                    return
            
            # Добавление новых классов
            if self.add_classes_check.isChecked():
                self.progress_bar.setValue(30)
                new_classes = self.get_selected_new_classes()
                if new_classes:
                    success, message = self.dataset_manager.add_new_classes(new_classes)
                    if not success:
                        QMessageBox.warning(self, "Ошибка", f"Ошибка добавления классов: {message}")
                        return
                    else:
                        QMessageBox.information(self, "Успех", message)
                else:
                    QMessageBox.information(self, "Информация", "Новые классы для добавления не найдены")
            
            # Добавление новых данных
            if self.add_data_check.isChecked():
                self.progress_bar.setValue(40)
                
                # Проверяем необходимые параметры
                raster_layer = self.raster_layer_combo.currentLayer()
                if not raster_layer:
                    QMessageBox.warning(self, "Ошибка", "Выберите растровый слой")
                    return
                
                objects_layer = self.objects_layer_combo.currentLayer()
                if not objects_layer:
                    QMessageBox.warning(self, "Ошибка", "Выберите слой объектов")
                    return
                
                class_field = self.class_field_combo.currentField()
                if not class_field:
                    QMessageBox.warning(self, "Ошибка", "Выберите поле классов")
                    return
                
                # Проверяем совместимость типа задачи
                task_type = self.task_combo.currentText()
                existing_task = self.dataset_manager.metadata.get('task', 'detect') if self.dataset_manager.metadata else 'detect'
                if existing_task != task_type:
                    reply = QMessageBox.question(
                        self, "Предупреждение", 
                        f"Тип задачи в существующем датасете ({existing_task}) отличается от выбранного ({task_type}). "
                        f"Продолжить обновление?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                
                # Создаем сетку на основе растрового слоя
                self.progress_bar.setValue(50)
                grid_layer, error_msg = create_grid_layer(
                    source_layer=raster_layer,
                    h_spacing=100.0,  # Можно сделать настраиваемым
                    v_spacing=100.0,
                    h_overlay=0.0,
                    v_overlay=0.0
                )
                
                if error_msg:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка создания сетки: {error_msg}")
                    return
                
                # Выполняем пересечение
                self.progress_bar.setValue(60)
                intersected_layer, error_msg = perform_intersection(
                    input_layer=objects_layer,
                    overlay_layer=grid_layer
                )
                
                if error_msg:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка пересечения: {error_msg}")
                    return
                
                # Экспортируем изображения из растрового слоя
                self.progress_bar.setValue(70)
                images_output_dir = os.path.join(self.dataset_manager.dataset_path, 'temp_images')
                progress_reporter = ProgressReporter(self.progress_bar, 70, 85)
                
                success, error_msg = export_views(
                    grid_layer=grid_layer,
                    output_dir=images_output_dir,
                    image_format=self.format_combo.currentText(),
                    width_px=self.width_spin.value(),
                    height_px=self.height_spin.value(),
                    dpi=int(self.dpi_combo.currentText()),
                    progress_reporter=progress_reporter
                )
                
                if not success:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта изображений: {error_msg}")
                    return
                
                # Добавляем данные в датасет
                self.progress_bar.setValue(85)
                splits = {
                    'train': self.train_spin.value(),
                    'val': self.val_spin.value(),
                    'test': self.test_spin.value()
                }
                
                # Обновляем метаданные с типом задачи
                new_metadata = {
                    'task': task_type,
                    'name': self.name_edit.text().strip(),
                    'description': self.description_edit.text().strip(),
                    'url': self.url_edit.text().strip()
                }
                
                progress_reporter = ProgressReporter(self.progress_bar, 85, 100)
                success, error_msg = self.dataset_manager.append_new_data(
                    intersected_layer=intersected_layer,
                    grid_layer=grid_layer,
                    class_field=class_field,
                    image_format=self.format_combo.currentText(),
                    splits=splits,
                    delete_void=True,
                    progress_reporter=progress_reporter,
                    metadata=new_metadata
                )
                
                if not success:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка добавления данных: {error_msg}")
                    return
                
                # Очищаем временные слои
                QgsProject.instance().removeMapLayer(grid_layer.id())
                QgsProject.instance().removeMapLayer(intersected_layer.id())
            
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Успех", "Датасет успешно обновлен!")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Критическая ошибка: {e}")
            self.progress_bar.setValue(0)
