# -*- coding: utf-8 -*-
"""
/***************************************************************************
 YoloQgisDialog
 This file was automatically generated with qtdesigner.py
 ***************************************************************************/
"""

import os
import json
import logging
from datetime import datetime

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import Qt

from qgis.core import QgsMapLayerProxyModel, QgsProject

# Настройка логирования
logger = logging.getLogger(__name__)

# --- ИМПОРТ ФУНКЦИЙ ИЗ ВНЕШНИХ МОДУЛЕЙ ---
from .grid_creator import create_grid_layer
from .intersection import perform_intersection
from .map_exporter import export_views
from .dataset_formatter import format_yolo_dataset 
from .dataset_formatter_yolo import save_yolo_native_dataset # <--- ДОБАВЛЕН НОВЫЙ ИМПОРТ
from .processing_utils import ProgressReporter
from .dataset_manager_dialog import DatasetManagerDialog
from .dataset_manager import DatasetManager
from .path_history_manager import PathHistoryManager

# --- ИМПОРТ МОДУЛЕЙ ТРЕНИРОВКИ ---
try:
    from .yolo_training_manager import YOLOTrainingManager, TrainingConfigManager
    from .yolo_detection_trainer import DetectionTrainer, DetectionDatasetAnalyzer
    from .yolo_segmentation_trainer import SegmentationTrainer, SegmentationDatasetAnalyzer
    from .yolo_validation import AdvancedValidator, ModelComparator
    from .yolo_metrics_tracker import MetricsTracker, MetricsVisualizer
    TRAINING_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Модули тренировки не доступны: {e}")
    TRAINING_MODULES_AVAILABLE = False
    # Создаем заглушки для избежания ошибок
    YOLOTrainingManager = None
    TrainingConfigManager = None
    DetectionTrainer = None
    DetectionDatasetAnalyzer = None
    SegmentationTrainer = None
    SegmentationDatasetAnalyzer = None
    AdvancedValidator = None
    ModelComparator = None
    MetricsTracker = None
    MetricsVisualizer = None



FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'yolo_qgis_dialog_base.ui'))


class YoloQgisDialog(QtWidgets.QDialog, FORM_CLASS):
    # ... (весь код __init__, _setup_connections, _update_size_values, update_split_values остается без изменений) ...
    def __init__(self, parent=None):
        """Конструктор."""
        super(YoloQgisDialog, self).__init__(parent)
        
        self.setupUi(self)
        
        # Инициализация компонентов тренировки
        self.training_manager = None
        self.config_manager = None
        self.current_experiment_id = None
        self.is_training = False
        
        # Инициализация менеджера истории путей
        self.path_history_manager = PathHistoryManager(max_history=10)
        
        self._setup_connections()
        self._initialize_training_components()
        self._setup_path_history_buttons()

    def _setup_connections(self):
        """Настройка всех сигналов и слотов."""
        self.buttonBox.accepted.connect(self.run_dataset_creation)
        self.buttonBox.rejected.connect(self.reject)

        self.mMapLayerComboBoxRaster.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.mMapLayerComboBoxObjects.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.mMapLayerComboBoxObjects.layerChanged.connect(self.mFieldComboBoxObjects.setLayer)
        self.mFieldComboBoxObjects.setLayer(self.mMapLayerComboBoxObjects.currentLayer())

        self.spinBox_Train.valueChanged.connect(self.update_split_values)
        self.spinBox_Val.valueChanged.connect(self.update_split_values)
        self.spinBox_Test.valueChanged.connect(self.update_split_values)
        
        self.progressBar.setValue(0)

        self.lineEdit_WidthPixel.textChanged.connect(self._update_size_values)
        self.lineEdit_HeigthPixel.textChanged.connect(self._update_size_values)
        self.lineEdit_WidthMeter.textChanged.connect(self._update_size_values)
        self.lineEdit_HeigthMeter.textChanged.connect(self._update_size_values)
        self.comboBox_Dpi.currentTextChanged.connect(self._update_size_values)
        
        # Инициализация QgsFileWidget
        try:
            # Устанавливаем начальную подсказку для QgsFileWidget
            self.mQgsFileWidget.setToolTip("Dataset directory for new dataset")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать QgsFileWidget: {e}")
        
        # Подключение кнопок управления датасетами
        self.manageDatasetsButton.clicked.connect(self.open_dataset_manager_dialog)
        
        # Подключение checkbox для обновления датасета
        self.checkBox_UpdateDataset.toggled.connect(self.toggle_update_mode)
        
        # Подключение сигналов тренировки
        self._setup_training_connections()

    def _update_size_values(self):
        """Автоматически пересчитывает размеры в метрах или пикселях."""
        sender = self.sender()
        if not sender: return
        is_pixel_source = sender in (self.lineEdit_WidthPixel, self.lineEdit_HeigthPixel, self.comboBox_Dpi)
        is_meter_source = sender in (self.lineEdit_WidthMeter, self.lineEdit_HeigthMeter)
        try:
            dpi = int(self.comboBox_Dpi.currentText())
            if is_pixel_source:
                w_px, h_px = int(self.lineEdit_WidthPixel.text()), int(self.lineEdit_HeigthPixel.text())
                w_m, h_m = (w_px / dpi) * 39.37, (h_px / dpi) * 39.37
                self.lineEdit_WidthMeter.blockSignals(True); self.lineEdit_HeigthMeter.blockSignals(True)
                self.lineEdit_WidthMeter.setText(f"{w_m:.4f}"); self.lineEdit_HeigthMeter.setText(f"{h_m:.4f}")
                self.lineEdit_WidthMeter.blockSignals(False); self.lineEdit_HeigthMeter.blockSignals(False)
            elif is_meter_source:
                w_m, h_m = float(self.lineEdit_WidthMeter.text()), float(self.lineEdit_HeigthMeter.text())
                w_px, h_px = round((w_m / 39.37) * dpi), round((h_m / 39.37) * dpi)
                self.lineEdit_WidthPixel.blockSignals(True); self.lineEdit_HeigthPixel.blockSignals(True)
                self.lineEdit_WidthPixel.setText(str(w_px)); self.lineEdit_HeigthPixel.setText(str(h_px))
                self.lineEdit_WidthPixel.blockSignals(False); self.lineEdit_HeigthPixel.blockSignals(False)
        except (ValueError, ZeroDivisionError):
            pass

    def update_split_values(self):
        """Перераспределяет значения Train/Val/Test, чтобы сумма была 100."""
        sender = self.sender()
        if not sender: return
        self.spinBox_Train.blockSignals(True); self.spinBox_Val.blockSignals(True); self.spinBox_Test.blockSignals(True)
        try:
            train, val, test = self.spinBox_Train.value(), self.spinBox_Val.value(), self.spinBox_Test.value()
            remainder = 100 - sender.value()
            if sender == self.spinBox_Train:
                other_sum = val + test
                if other_sum == 0: new_val, new_test = remainder, 0
                else: 
                    ratio = val / other_sum
                    new_val = round(remainder * ratio)
                    new_test = remainder - new_val
                self.spinBox_Val.setValue(new_val); self.spinBox_Test.setValue(new_test)
            elif sender == self.spinBox_Val:
                other_sum = train + test
                if other_sum == 0: new_train, new_test = remainder, 0
                else: 
                    ratio = train / other_sum
                    new_train = round(remainder * ratio)
                    new_test = remainder - new_train
                self.spinBox_Train.setValue(new_train); self.spinBox_Test.setValue(new_test)
            elif sender == self.spinBox_Test:
                other_sum = train + val
                if other_sum == 0: new_train, new_val = remainder, 0
                else: 
                    ratio = train / other_sum
                    new_train = round(remainder * ratio)
                    new_val = remainder - new_train
                self.spinBox_Train.setValue(new_train); self.spinBox_Val.setValue(new_val)
        finally:
            self.spinBox_Train.blockSignals(False); self.spinBox_Val.blockSignals(False); self.spinBox_Test.blockSignals(False)
    
    def toggle_update_mode(self, checked):
        """Переключает режим обновления датасета"""
        try:
            if checked:
                # В режиме обновления меняем подсказку для основного поля
                self.mQgsFileWidget.setToolTip("Dataset directory - will update existing dataset if found, otherwise create new one")
            else:
                # В обычном режиме возвращаем стандартную подсказку
                self.mQgsFileWidget.setToolTip("Dataset directory for new dataset")
        except Exception as e:
            logger.warning(f"Не удалось обновить подсказку QgsFileWidget: {e}")

    def run_dataset_creation(self):
        """Основная функция, запускающая весь процесс."""
        logger.info("--- Запуск процесса ---")
        self.progressBar.setValue(0)

        # --- Сбор и проверка данных ---
        try:
            # Безопасное получение пути из QgsFileWidget
            try:
                output_dir = self.mQgsFileWidget.filePath()
            except Exception as e:
                logger.error(f"Ошибка получения пути из QgsFileWidget: {e}", exc_info=True)
                output_dir = ""
            
            objects_layer = self.mMapLayerComboBoxObjects.currentLayer()
            classes_field = self.mFieldComboBoxObjects.currentField()
            is_update_mode = self.checkBox_UpdateDataset.isChecked()
            
            if not all([output_dir, objects_layer, classes_field]):
                raise ValueError("Необходимо заполнить все поля в группе 'Setup'.")
            
            # Сохраняем путь к датасету в историю
            self.path_history_manager.add_dataset_path(output_dir)
            
            # В режиме обновления проверяем, есть ли существующий датасет в указанной директории
            if is_update_mode:
                from .dataset_utils import DatasetUtils
                is_valid_dataset, _ = DatasetUtils.validate_dataset_path(output_dir)
                if not is_valid_dataset:
                    # Если датасета нет, переключаемся в режим создания нового
                    is_update_mode = False
                    logger.info("В указанной директории не найден существующий датасет. Создается новый датасет.")
                else:
                    logger.info(f"Найден существующий датасет в директории: {output_dir}")
            
            img_width_m = float(self.lineEdit_WidthMeter.text())
            img_height_m = float(self.lineEdit_HeigthMeter.text())
            v_overlay_m = float(self.lineEdit_VerticalOverlay.text() or 0.0)
            h_overlay_m = float(self.lineEdit_HorizontalOverlay.text() or 0.0)
            img_width_px = int(self.lineEdit_WidthPixel.text())
            img_height_px = int(self.lineEdit_HeigthPixel.text())
            img_dpi = int(self.comboBox_Dpi.currentText())
            img_format = self.comboBoxFileFormat.currentText()
            
            # --- ИЗМЕНЕНИЕ: Считываем состояние чекбокса для выбора формата ---
            # Предполагается, что в UI-файле добавлен QCheckBox с именем 'checkBox_YoloFormat'
            save_in_yolo_format = self.checkBox_YoloFormat.isChecked()
            
            if any(x <= 0 for x in [img_width_m, img_height_m, img_width_px, img_height_px]):
                raise ValueError("Все размеры должны быть больше нуля.")
        except (ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка ввода", f"Проверьте корректность входных данных:\n{e}")
            return
        
        # --- ШАГ 1: Создание сетки (0% -> 15%) ---
        logger.info("1. Создание сетки...")
        self.progressBar.setValue(5)
        grid_layer, error_msg = create_grid_layer(
            source_layer=objects_layer, h_spacing=img_width_m, v_spacing=img_height_m,
            h_overlay=h_overlay_m, v_overlay=v_overlay_m)
        if error_msg:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg); self.progressBar.setValue(0); return
        grid_layer.setName(f"Сетка для '{objects_layer.name()}'")
        QgsProject.instance().addMapLayer(grid_layer)
        self.progressBar.setValue(15)
        logger.info("Сетка создана.")

        # --- ШАГ 2: Выполнение пересечения (15% -> 25%) ---
        logger.info("2. Пересечение объектов с сеткой...")
        self.progressBar.setValue(20)
        intersected_layer, error_msg = perform_intersection(
            input_layer=objects_layer, overlay_layer=grid_layer)
        if error_msg:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg); self.progressBar.setValue(0); return
        intersected_layer.setName(f"Пересечение для '{objects_layer.name()}'")
        QgsProject.instance().addMapLayer(intersected_layer)
        self.progressBar.setValue(25)
        logger.info("Слой пересечения создан.")

        # --- ШАГ 3: Экспорт изображений (25% -> 75%) ---
        logger.info("3. Экспорт изображений тайлов...")
        
        # В режиме обновления экспортируем изображения в существующий датасет
        if is_update_mode:
            # Создаем временную структуру в существующем датасете
            temp_images_dir = os.path.join(output_dir, 'temp_new_data', 'images')
            os.makedirs(temp_images_dir, exist_ok=True)
            images_output_dir = temp_images_dir
        else:
            images_output_dir = os.path.join(output_dir, 'images')
        
        progress_reporter_export = ProgressReporter(self.progressBar, start_percentage=25, end_percentage=75)
        success, error_msg = export_views(
            grid_layer=grid_layer, output_dir=images_output_dir, image_format=img_format,
            width_px=img_width_px, height_px=img_height_px, dpi=img_dpi,
            progress_reporter=progress_reporter_export)
        if not success:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg); self.progressBar.setValue(0); return
        self.progressBar.setValue(75)
        logger.info("Экспорт изображений завершен.")

        # --- ШАГ 4: Формирование датасета (75% -> 100%) ---
        splits = {
            'train': self.spinBox_Train.value(),
            'val': self.spinBox_Val.value(),
            'test': self.spinBox_Test.value()
        }
        
        metadata = {
            'task': self.comboBox_TaskDataset.currentText(),
            'name': self.lineEdit_NameDataset.text(),
            'desc': self.lineEdit_DescriptionDataset.text(),
            'url': self.lineEdit_UrlDataset.text()
        }

        delete_void = self.voidImages.isChecked()
        progress_reporter_format = ProgressReporter(self.progressBar, start_percentage=75, end_percentage=100)
        
        # --- ИЗМЕНЕНИЕ: ВЫБОР РЕЖИМА РАБОТЫ ---
        if is_update_mode:
            logger.info("4. Обновление существующего датасета...")
            success, error_msg = self.update_existing_dataset(
                existing_dataset_path=output_dir,
                intersected_layer=intersected_layer, 
                grid_layer=grid_layer,
                class_field=classes_field, 
                image_format=img_format,
                splits=splits, 
                metadata=metadata, 
                delete_void=delete_void,
                progress_reporter=progress_reporter_format
            )
        elif save_in_yolo_format:
            logger.info("4. Формирование датасета в нативном формате YOLO...")
            success, error_msg = save_yolo_native_dataset(
                intersected_layer=intersected_layer, grid_layer=grid_layer,
                class_field=classes_field, output_dir=output_dir, image_format=img_format,
                splits=splits, metadata=metadata, delete_void=delete_void,
                progress_reporter=progress_reporter_format)
        else:
            logger.info("4. Формирование файла аннотаций data.ndjson...")
            success, error_msg = format_yolo_dataset(
                intersected_layer=intersected_layer, grid_layer=grid_layer,
                class_field=classes_field, output_dir=output_dir, image_format=img_format,
                image_width=img_width_px, image_height=img_height_px,
                splits=splits, metadata=metadata, delete_void=delete_void,
                progress_reporter=progress_reporter_format)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            
        if not success:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg); self.progressBar.setValue(0); return
        logger.info("Формирование датасета завершено.")
        
        self.progressBar.setValue(100)
        logger.info("--- Процесс успешно завершен ---")
        if is_update_mode:
            QtWidgets.QMessageBox.information(self, "Готово", f"Датасет успешно обновлен!\nОбновлен в: {output_dir}")
        else:
            QtWidgets.QMessageBox.information(self, "Готово", f"Датасет успешно создан!\nСохранено в: {output_dir}")
        # self.accept()  # Убрано, чтобы окно не закрывалось после выполнения
    
    def update_existing_dataset(self, existing_dataset_path, intersected_layer, grid_layer, 
                               class_field, image_format, splits, metadata, delete_void, progress_reporter):
        """Обновляет существующий датасет новыми данными"""
        try:
            # Инициализируем менеджер датасета
            dataset_manager = DatasetManager(existing_dataset_path)
            can_update, message = dataset_manager.can_update_dataset()
            
            if not can_update:
                return False, f"Не удается обновить датасет: {message}"
            
            # Создаем резервную копию
            progress_reporter.set_progress(1, 5)
            success, backup_message = dataset_manager.create_backup()
            if not success:
                logger.warning(backup_message)
            else:
                logger.info(backup_message)
            
            # Обновляем метаданные
            progress_reporter.set_progress(2, 5)
            success, meta_message = dataset_manager.update_dataset_metadata(metadata)
            if not success:
                return False, f"Ошибка обновления метаданных: {meta_message}"
            
            # Извлекаем новые классы из векторного слоя
            progress_reporter.set_progress(3, 5)
            field_index = intersected_layer.fields().indexFromName(class_field)
            if field_index == -1:
                return False, f"Поле '{class_field}' не найдено в слое"
            
            unique_values = intersected_layer.uniqueValues(field_index)
            all_classes = [str(val) for val in unique_values if val is not None and str(val).strip()]
            
            # Получаем существующие классы из датасета
            existing_classes = set(dataset_manager.class_names.values())
            
            # Находим новые классы
            new_classes = [cls for cls in all_classes if cls not in existing_classes]
            
            # Добавляем новые классы
            if new_classes:
                success, classes_message = dataset_manager.add_new_classes(new_classes)
                if not success:
                    return False, f"Ошибка добавления классов: {classes_message}"
                logger.info(f"Добавлены новые классы: {', '.join(new_classes)}")
            
            # Добавляем новые данные
            progress_reporter.set_progress(4, 5)
            success, data_message = dataset_manager.append_new_data(
                intersected_layer=intersected_layer,
                grid_layer=grid_layer,
                class_field=class_field,
                image_format=image_format,
                splits=splits,
                delete_void=delete_void,
                progress_reporter=progress_reporter,
                metadata=metadata
            )
            
            if not success:
                return False, f"Ошибка добавления данных: {data_message}"
            
            return True, "Датасет успешно обновлен"
            
        except Exception as e:
            return False, f"Критическая ошибка при обновлении датасета: {e}"
    
    def open_dataset_manager_dialog(self):
        """Открывает диалог управления датасетами"""
        try:
            manager_dialog = DatasetManagerDialog(self)
            manager_dialog.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка открытия диалога управления: {e}")
    
    def _initialize_training_components(self):
        """Инициализирует компоненты системы тренировки"""
        if not TRAINING_MODULES_AVAILABLE:
            logger.warning("Модули тренировки не доступны. Функции обучения будут отключены.")
            self.training_manager = None
            self.config_manager = None
            return
        
        try:
            # Создаем менеджер тренировки
            self.training_manager = YOLOTrainingManager()
            
            # Создаем менеджер конфигураций
            self.config_manager = TrainingConfigManager()
            
            # Подключаем сигналы менеджера тренировки
            self.training_manager.training_started.connect(self._on_training_started)
            self.training_manager.training_progress.connect(self._on_training_progress)
            self.training_manager.training_completed.connect(self._on_training_completed)
            self.training_manager.validation_completed.connect(self._on_validation_completed)
            # Подключаем статусные сообщения тренировки
            self.training_manager.status_message.connect(self._on_status_message)
            
            # Обновляем список экспериментов
            self._refresh_experiments_list()
            
        except Exception as e:
            logger.error(f"Ошибка инициализации компонентов тренировки: {e}", exc_info=True)
            QtWidgets.QMessageBox.warning(self, "Предупреждение", 
                                        f"Некоторые функции тренировки могут быть недоступны: {e}")
            self.training_manager = None
            self.config_manager = None
    
    def _setup_training_connections(self):
        """Настраивает соединения для интерфейса тренировки"""
        try:
            # Безопасное подключение кнопок тренировки
            if hasattr(self, 'pushButtonStartTraining'):
                self.pushButtonStartTraining.clicked.connect(self._start_training)
            if hasattr(self, 'pushButtonStopTraining'):
                self.pushButtonStopTraining.clicked.connect(self._stop_training)
            
            # Безопасное подключение кнопок анализа датасета
            if hasattr(self, 'pushButtonAnalyzeDataset'):
                self.pushButtonAnalyzeDataset.clicked.connect(self._analyze_dataset)
            
            # Безопасное подключение кнопок конфигурации
            if hasattr(self, 'pushButtonLoadConfig'):
                self.pushButtonLoadConfig.clicked.connect(self._load_training_config)
            if hasattr(self, 'pushButtonSaveConfig'):
                self.pushButtonSaveConfig.clicked.connect(self._save_training_config)
            
            # Безопасное подключение кнопок валидации
            if hasattr(self, 'pushButtonStartValidation'):
                self.pushButtonStartValidation.clicked.connect(self._start_validation)
            if hasattr(self, 'pushButtonCompareModels'):
                self.pushButtonCompareModels.clicked.connect(self._compare_models)
            if hasattr(self, 'pushButtonExportResults'):
                self.pushButtonExportResults.clicked.connect(self._export_validation_results)
            
            # Безопасное подключение кнопок метрик
            if hasattr(self, 'pushButtonRefreshExperiments'):
                self.pushButtonRefreshExperiments.clicked.connect(self._refresh_experiments_list)
            if hasattr(self, 'pushButtonDeleteExperiment'):
                self.pushButtonDeleteExperiment.clicked.connect(self._delete_experiment)
            if hasattr(self, 'pushButtonExportExperiment'):
                self.pushButtonExportExperiment.clicked.connect(self._export_experiment)
            if hasattr(self, 'pushButtonGeneratePlots'):
                self.pushButtonGeneratePlots.clicked.connect(self._generate_plots)
            if hasattr(self, 'pushButtonSavePlots'):
                self.pushButtonSavePlots.clicked.connect(self._save_plots)
            
            # Безопасное подключение списка экспериментов
            if hasattr(self, 'listWidgetExperiments'):
                self.listWidgetExperiments.itemSelectionChanged.connect(self._on_experiment_selected)
            
            # Безопасное подключение изменения типа задачи
            if hasattr(self, 'comboBoxTaskType'):
                self.comboBoxTaskType.currentTextChanged.connect(self._on_task_type_changed)
            
        except Exception as e:
            logger.error(f"Ошибка настройки соединений тренировки: {e}", exc_info=True)
    
    def _on_task_type_changed(self, task_type):
        """Обработчик изменения типа задачи"""
        try:
            if not hasattr(self, 'comboBoxModelType'):
                logger.warning("comboBoxModelType не найден")
                return
                
            if "Детекция" in task_type:
                # Обновляем список моделей для детекции
                self.comboBoxModelType.clear()
                models = [
                    "YOLOv8n (быстрая)",
                    "YOLOv8s (сбалансированная)", 
                    "YOLOv8m (средняя)",
                    "YOLOv8l (большая)",
                    "YOLOv8x (максимальная)",
                    "YOLOv11n (быстрая)",
                    "YOLOv11s (сбалансированная)",
                    "YOLOv11m (средняя)",
                    "YOLOv11l (большая)",
                    "YOLOv11x (максимальная)"
                ]
                self.comboBoxModelType.addItems(models)
            elif "Сегментация" in task_type:
                # Обновляем список моделей для сегментации
                self.comboBoxModelType.clear()
                models = [
                    "YOLOv8n-seg (быстрая)",
                    "YOLOv8s-seg (сбалансированная)",
                    "YOLOv8m-seg (средняя)", 
                    "YOLOv8l-seg (большая)",
                    "YOLOv8x-seg (максимальная)",
                    "YOLOv11n-seg (быстрая)",
                    "YOLOv11s-seg (сбалансированная)",
                    "YOLOv11m-seg (средняя)",
                    "YOLOv11l-seg (большая)",
                    "YOLOv11x-seg (максимальная)"
                ]
                self.comboBoxModelType.addItems(models)
        except Exception as e:
            logger.error(f"Ошибка обновления типа задачи: {e}", exc_info=True)
    
    def _setup_path_history_buttons(self):
        """Настраивает кнопки для выбора из истории путей"""
        try:
            from qgis.PyQt.QtWidgets import QPushButton, QMenu
            
            # Создаем кнопку для истории датасетов
            if hasattr(self, 'fileWidgetDataset') and hasattr(self, 'gridLayout_dataset'):
                self.btnDatasetHistory = QPushButton("📁")
                self.btnDatasetHistory.setToolTip("Выбрать из истории датасетов")
                self.btnDatasetHistory.setMaximumWidth(30)
                self.btnDatasetHistory.clicked.connect(self._show_dataset_history_menu)
                
                # Добавляем кнопку в layout рядом с fileWidgetDataset (строка 0, колонка 2)
                self.gridLayout_dataset.addWidget(self.btnDatasetHistory, 0, 2)
            
            # Создаем кнопку для истории проектов
            if hasattr(self, 'fileWidgetSaveDir') and hasattr(self, 'gridLayout_output'):
                self.btnProjectHistory = QPushButton("📁")
                self.btnProjectHistory.setToolTip("Выбрать из истории проектов")
                self.btnProjectHistory.setMaximumWidth(30)
                self.btnProjectHistory.clicked.connect(self._show_project_history_menu)
                
                # Добавляем кнопку в layout рядом с fileWidgetSaveDir (строка 1, колонка 2)
                self.gridLayout_output.addWidget(self.btnProjectHistory, 1, 2)
                            
        except Exception as e:
            logger.error(f"Ошибка настройки кнопок истории путей: {e}", exc_info=True)
    
    def _show_dataset_history_menu(self):
        """Показывает меню с историей путей к датасетам"""
        try:
            from qgis.PyQt.QtWidgets import QMenu, QAction
            
            history = self.path_history_manager.get_dataset_paths()
            
            if not history:
                QtWidgets.QMessageBox.information(
                    self, "История путей", 
                    "История путей к датасетам пуста.\nВыберите путь, и он будет сохранен в истории."
                )
                return
            
            menu = QMenu(self)
            
            # Добавляем действия для каждого пути
            for path in history:
                # Обрезаем путь, если он слишком длинный
                display_path = path
                if len(display_path) > 60:
                    display_path = "..." + display_path[-57:]
                
                action = QAction(display_path, self)
                action.setToolTip(path)
                action.triggered.connect(lambda checked, p=path: self._select_dataset_path(p))
                menu.addAction(action)
            
            menu.addSeparator()
            
            # Добавляем действие для очистки истории
            clear_action = QAction("Очистить историю", self)
            clear_action.triggered.connect(self._clear_dataset_history)
            menu.addAction(clear_action)
            
            # Показываем меню под кнопкой
            button_pos = self.btnDatasetHistory.mapToGlobal(self.btnDatasetHistory.rect().bottomLeft())
            menu.exec_(button_pos)
            
        except Exception as e:
            logger.error(f"Ошибка показа меню истории датасетов: {e}", exc_info=True)
    
    def _show_project_history_menu(self):
        """Показывает меню с историей путей к проектам"""
        try:
            from qgis.PyQt.QtWidgets import QMenu, QAction
            
            history = self.path_history_manager.get_project_paths()
            
            if not history:
                QtWidgets.QMessageBox.information(
                    self, "История путей", 
                    "История путей к проектам пуста.\nВыберите путь, и он будет сохранен в истории."
                )
                return
            
            menu = QMenu(self)
            
            # Добавляем действия для каждого пути
            for path in history:
                # Обрезаем путь, если он слишком длинный
                display_path = path
                if len(display_path) > 60:
                    display_path = "..." + display_path[-57:]
                
                action = QAction(display_path, self)
                action.setToolTip(path)
                action.triggered.connect(lambda checked, p=path: self._select_project_path(p))
                menu.addAction(action)
            
            menu.addSeparator()
            
            # Добавляем действие для очистки истории
            clear_action = QAction("Очистить историю", self)
            clear_action.triggered.connect(self._clear_project_history)
            menu.addAction(clear_action)
            
            # Показываем меню под кнопкой
            button_pos = self.btnProjectHistory.mapToGlobal(self.btnProjectHistory.rect().bottomLeft())
            menu.exec_(button_pos)
            
        except Exception as e:
            logger.error(f"Ошибка показа меню истории проектов: {e}", exc_info=True)
    
    def _select_dataset_path(self, path: str):
        """Выбирает путь к датасету из истории"""
        try:
            if hasattr(self, 'fileWidgetDataset'):
                self.fileWidgetDataset.setFilePath(path)
        except Exception as e:
            logger.error(f"Ошибка выбора пути к датасету: {e}", exc_info=True)
    
    def _select_project_path(self, path: str):
        """Выбирает путь к проекту из истории"""
        try:
            if hasattr(self, 'fileWidgetSaveDir'):
                self.fileWidgetSaveDir.setFilePath(path)
        except Exception as e:
            logger.error(f"Ошибка выбора пути к проекту: {e}", exc_info=True)
    
    def _clear_dataset_history(self):
        """Очищает историю путей к датасетам"""
        reply = QtWidgets.QMessageBox.question(
            self, "Очистить историю",
            "Вы уверены, что хотите очистить историю путей к датасетам?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.path_history_manager.clear_dataset_history()
            QtWidgets.QMessageBox.information(self, "История очищена", "История путей к датасетам очищена.")
    
    def _clear_project_history(self):
        """Очищает историю путей к проектам"""
        reply = QtWidgets.QMessageBox.question(
            self, "Очистить историю",
            "Вы уверены, что хотите очистить историю путей к проектам?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.path_history_manager.clear_project_history()
            QtWidgets.QMessageBox.information(self, "История очищена", "История путей к проектам очищена.")
    
    def _start_training(self):
        """Запускает процесс тренировки"""
        try:
            if not self.training_manager:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка", 
                    "Модули тренировки не доступны. Установите ultralytics: pip install ultralytics"
                )
                return
            
            if self.is_training:
                QtWidgets.QMessageBox.warning(self, "Предупреждение", "Тренировка уже выполняется!")
                return
            
            # Собираем параметры тренировки
            dataset_path = self.fileWidgetDataset.filePath()
            if not dataset_path:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите путь к датасету!")
                return
            
            # Определяем тип задачи
            task_type = self.comboBoxTaskType.currentText()
            task = "detect" if "Детекция" in task_type else "segment"
            
            # Определяем тип модели
            model_type_text = self.comboBoxModelType.currentText()
            model_type = self._get_model_type_from_text(model_type_text, task)
            
            # Собираем параметры
            epochs = self.spinBoxEpochs.value()
            batch_size = self.spinBoxBatchSize.value()
            image_size = self.spinBoxImageSize.value()
            learning_rate = self.doubleSpinBoxLearningRate.value()
            device = "cpu" if self.comboBoxDevice.currentText() == "CPU" else "0"
            pretrained = self.checkBoxPretrained.isChecked()
            
            # Параметры аугментации
            augmentation_params = {}
            if self.groupBoxAugmentation.isChecked():
                augmentation_params = {
                    'mosaic': self.doubleSpinBoxMosaic.value(),
                    'mixup': self.doubleSpinBoxMixup.value(),
                    'copy_paste': self.doubleSpinBoxCopyPaste.value(),
                    'fliplr': self.doubleSpinBoxFlipLR.value()
                }
            
            # Выходные параметры
            project_name = self.lineEditProjectName.text() or "yolo_training"
            save_dir = self.fileWidgetSaveDir.filePath()
            
            # Сохраняем пути в историю
            self.path_history_manager.add_dataset_path(dataset_path)
            if save_dir:
                self.path_history_manager.add_project_path(save_dir)
            
            # Запускаем тренировку
            if task == "detect":
                self.current_experiment_id = self.training_manager.start_detection_training(
                    dataset_path=dataset_path,
                    model_type=model_type,
                    epochs=epochs,
                    batch_size=batch_size,
                    image_size=image_size,
                    learning_rate=learning_rate,
                    device=device,
                    pretrained=pretrained,
                    save_dir=save_dir,
                    project_name=project_name,
                    **augmentation_params
                )
            else:
                self.current_experiment_id = self.training_manager.start_segmentation_training(
                    dataset_path=dataset_path,
                    model_type=model_type,
                    epochs=epochs,
                    batch_size=batch_size,
                    image_size=image_size,
                    learning_rate=learning_rate,
                    device=device,
                    pretrained=pretrained,
                    save_dir=save_dir,
                    project_name=project_name,
                    **augmentation_params
                )
            
            if self.current_experiment_id:
                self.is_training = True
                self.pushButtonStartTraining.setEnabled(False)
                self.pushButtonStopTraining.setEnabled(True)
                self.labelTrainingStatus.setText("Тренировка запущена...")
                self.textEditTrainingLog.append(f"Запущена тренировка: {project_name}")
            else:
                QtWidgets.QMessageBox.critical(self, "Ошибка", "Не удалось запустить тренировку!")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка запуска тренировки: {e}")
    
    def _stop_training(self):
        """Останавливает процесс тренировки"""
        try:
            if self.current_experiment_id and self.training_manager:
                success = self.training_manager.cancel_training(self.current_experiment_id)
                if success:
                    # Обновляем состояние интерфейса
                    self.is_training = False
                    self.pushButtonStartTraining.setEnabled(True)
                    self.pushButtonStopTraining.setEnabled(False)
                    self.labelTrainingStatus.setText("Тренировка остановлена")
                    self.textEditTrainingLog.append("Тренировка остановлена пользователем")
                else:
                    QtWidgets.QMessageBox.warning(self, "Предупреждение", "Не удалось остановить тренировку")
            else:
                # Если нет активного эксперимента, просто обновляем состояние кнопок
                self.is_training = False
                self.pushButtonStartTraining.setEnabled(True)
                self.pushButtonStopTraining.setEnabled(False)
                self.labelTrainingStatus.setText("Тренировка остановлена")
        except Exception as e:
            # В случае ошибки все равно обновляем состояние кнопок
            self.is_training = False
            self.pushButtonStartTraining.setEnabled(True)
            self.pushButtonStopTraining.setEnabled(False)
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка остановки тренировки: {e}")
    
    def _analyze_dataset(self):
        """Анализирует выбранный датасет"""
        try:
            dataset_path = self.fileWidgetDataset.filePath()
            if not dataset_path:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите путь к датасету!")
                return
            
            task_type = self.comboBoxTaskType.currentText()
            task = "detect" if "Детекция" in task_type else "segment"
            
            # Анализируем датасет
            analysis = self.training_manager.analyze_dataset(dataset_path, task)
            
            if 'error' in analysis:
                QtWidgets.QMessageBox.critical(self, "Ошибка анализа", analysis['error'])
                return
            
            # Отображаем результаты анализа
            self._display_dataset_analysis(analysis)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка анализа датасета: {e}")
    
    def _display_dataset_analysis(self, analysis):
        """Отображает результаты анализа датасета"""
        try:
            # Создаем диалог с результатами анализа
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Анализ датасета")
            dialog.setModal(True)
            dialog.resize(600, 400)
            
            layout = QtWidgets.QVBoxLayout(dialog)
            
            # Создаем текстовое поле для отображения результатов
            text_edit = QtWidgets.QTextEdit()
            text_edit.setReadOnly(True)
            
            # Форматируем результаты анализа
            analysis_text = self._format_analysis_results(analysis)
            text_edit.setPlainText(analysis_text)
            
            layout.addWidget(text_edit)
            
            # Кнопка закрытия
            button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
            button_box.accepted.connect(dialog.accept)
            layout.addWidget(button_box)
            
            dialog.exec_()
            
        except Exception as e:
            logger.error(f"Ошибка отображения анализа: {e}", exc_info=True)
    
    def _format_analysis_results(self, analysis):
        """Форматирует результаты анализа для отображения"""
        try:
            text = "=== АНАЛИЗ ДАТАСЕТА ===\n\n"
            
            # Информация о датасете
            if 'dataset_info' in analysis:
                info = analysis['dataset_info']
                text += f"Путь: {info.get('path', 'N/A')}\n"
                text += f"Количество классов: {info.get('nc', 'N/A')}\n"
                text += f"Классы: {list(info.get('names', {}).values())}\n\n"
            
            # Статистика по сплитам
            if 'splits' in analysis:
                text += "=== СТАТИСТИКА ПО СПЛИТАМ ===\n"
                for split_name, split_data in analysis['splits'].items():
                    if 'error' not in split_data:
                        text += f"\n{split_name.upper()}:\n"
                        text += f"  Изображений: {split_data.get('image_count', 0)}\n"
                        text += f"  Аннотаций: {split_data.get('annotation_count', 0)}\n"
                        text += f"  Объектов: {split_data.get('total_objects', 0)}\n"
                        text += f"  Объектов на изображение: {split_data.get('objects_per_image', 0):.2f}\n"
                        
                        # Распределение по классам
                        if 'class_distribution' in split_data:
                            text += f"  Распределение по классам:\n"
                            for class_id, count in split_data['class_distribution'].items():
                                text += f"    Класс {class_id}: {count} объектов\n"
            
            # Общая статистика
            if 'total_images' in analysis:
                text += f"\n=== ОБЩАЯ СТАТИСТИКА ===\n"
                text += f"Всего изображений: {analysis.get('total_images', 0)}\n"
                text += f"Всего аннотаций: {analysis.get('total_annotations', 0)}\n"
            
            return text
            
        except Exception as e:
            return f"Ошибка форматирования результатов: {e}"
    
    def _get_model_type_from_text(self, model_text, task):
        """Преобразует текст модели в тип модели"""
        model_mapping = {
            "YOLOv8n (быстрая)": "yolov8n",
            "YOLOv8s (сбалансированная)": "yolov8s", 
            "YOLOv8m (средняя)": "yolov8m",
            "YOLOv8l (большая)": "yolov8l",
            "YOLOv8x (максимальная)": "yolov8x",
            "YOLOv11n (быстрая)": "yolov11n",
            "YOLOv11s (сбалансированная)": "yolov11s",
            "YOLOv11m (средняя)": "yolov11m",
            "YOLOv11l (большая)": "yolov11l",
            "YOLOv11x (максимальная)": "yolov11x",
            "YOLOv8n-seg (быстрая)": "yolov8n-seg",
            "YOLOv8s-seg (сбалансированная)": "yolov8s-seg",
            "YOLOv8m-seg (средняя)": "yolov8m-seg",
            "YOLOv8l-seg (большая)": "yolov8l-seg",
            "YOLOv8x-seg (максимальная)": "yolov8x-seg",
            "YOLOv11n-seg (быстрая)": "yolov11n-seg",
            "YOLOv11s-seg (сбалансированная)": "yolov11s-seg",
            "YOLOv11m-seg (средняя)": "yolov11m-seg",
            "YOLOv11l-seg (большая)": "yolov11l-seg",
            "YOLOv11x-seg (максимальная)": "yolov11x-seg"
        }
        return model_mapping.get(model_text, "yolov8n")
    
    def _load_training_config(self):
        """Загружает конфигурацию тренировки"""
        try:
            configs = self.config_manager.list_configs()
            if not configs:
                QtWidgets.QMessageBox.information(self, "Информация", "Нет сохраненных конфигураций")
                return
            
            # Создаем диалог выбора конфигурации
            config_name, ok = QtWidgets.QInputDialog.getItem(
                self, "Загрузить конфигурацию", "Выберите конфигурацию:", configs, 0, False
            )
            
            if ok and config_name:
                config = self.config_manager.load_config(config_name)
                if config:
                    self._apply_training_config(config)
                    QtWidgets.QMessageBox.information(self, "Успех", f"Конфигурация '{config_name}' загружена")
                else:
                    QtWidgets.QMessageBox.critical(self, "Ошибка", "Не удалось загрузить конфигурацию")
                    
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки конфигурации: {e}")
    
    def _save_training_config(self):
        """Сохраняет текущую конфигурацию тренировки"""
        try:
            config_name, ok = QtWidgets.QInputDialog.getText(
                self, "Сохранить конфигурацию", "Введите имя конфигурации:"
            )
            
            if ok and config_name:
                config = self._get_current_training_config()
                success = self.config_manager.save_config(config, config_name)
                
                if success:
                    QtWidgets.QMessageBox.information(self, "Успех", f"Конфигурация '{config_name}' сохранена")
                else:
                    QtWidgets.QMessageBox.critical(self, "Ошибка", "Не удалось сохранить конфигурацию")
                    
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения конфигурации: {e}")
    
    def _get_current_training_config(self):
        """Получает текущую конфигурацию тренировки"""
        try:
            task_type = self.comboBoxTaskType.currentText()
            task = "detect" if "Детекция" in task_type else "segment"
            
            model_type_text = self.comboBoxModelType.currentText()
            model_type = self._get_model_type_from_text(model_type_text, task)
            
            config = {
                'task': task,
                'model_type': model_type,
                'epochs': self.spinBoxEpochs.value(),
                'batch_size': self.spinBoxBatchSize.value(),
                'image_size': self.spinBoxImageSize.value(),
                'learning_rate': self.doubleSpinBoxLearningRate.value(),
                'device': self.comboBoxDevice.currentText(),
                'pretrained': self.checkBoxPretrained.isChecked(),
                'augmentation': {
                    'mosaic': self.doubleSpinBoxMosaic.value(),
                    'mixup': self.doubleSpinBoxMixup.value(),
                    'copy_paste': self.doubleSpinBoxCopyPaste.value(),
                    'fliplr': self.doubleSpinBoxFlipLR.value()
                } if self.groupBoxAugmentation.isChecked() else {}
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Ошибка получения конфигурации: {e}", exc_info=True)
            return {}
    
    def _apply_training_config(self, config):
        """Применяет загруженную конфигурацию к интерфейсу"""
        try:
            # Применяем основные параметры
            if 'epochs' in config:
                self.spinBoxEpochs.setValue(config['epochs'])
            if 'batch_size' in config:
                self.spinBoxBatchSize.setValue(config['batch_size'])
            if 'image_size' in config:
                self.spinBoxImageSize.setValue(config['image_size'])
            if 'learning_rate' in config:
                self.doubleSpinBoxLearningRate.setValue(config['learning_rate'])
            if 'pretrained' in config:
                self.checkBoxPretrained.setChecked(config['pretrained'])
            
            # Применяем параметры аугментации
            if 'augmentation' in config:
                aug = config['augmentation']
                if 'mosaic' in aug:
                    self.doubleSpinBoxMosaic.setValue(aug['mosaic'])
                if 'mixup' in aug:
                    self.doubleSpinBoxMixup.setValue(aug['mixup'])
                if 'copy_paste' in aug:
                    self.doubleSpinBoxCopyPaste.setValue(aug['copy_paste'])
                if 'fliplr' in aug:
                    self.doubleSpinBoxFlipLR.setValue(aug['fliplr'])
                    
        except Exception as e:
            logger.error(f"Ошибка применения конфигурации: {e}", exc_info=True)
    
    # Обработчики сигналов тренировки
    def _on_training_started(self, experiment_id):
        """Обработчик начала тренировки"""
        self.current_experiment_id = experiment_id
        self.is_training = True
        self.pushButtonStartTraining.setEnabled(False)
        self.pushButtonStopTraining.setEnabled(True)
        self.labelTrainingStatus.setText("Тренировка запущена...")
        self.progressBarTraining.setValue(0)
    
    def _on_training_progress(self, epoch, metrics):
        """Обработчик прогресса тренировки"""
        try:
            # Обновляем прогресс
            total_epochs = self.spinBoxEpochs.value()
            progress = int((epoch / total_epochs) * 100)
            self.progressBarTraining.setValue(progress)
            
            # Обновляем статус
            self.labelTrainingStatus.setText(f"Эпоха {epoch}/{total_epochs}")
            
            # Добавляем метрики в лог
            metrics_text = f"Эпоха {epoch}: "
            for key, value in metrics.items():
                metrics_text += f"{key}={value:.4f} "
            self.textEditTrainingLog.append(metrics_text)
            
            # Обновляем таблицу метрик
            self._update_metrics_table(epoch, metrics)
            
        except Exception as e:
            logger.error(f"Ошибка обновления прогресса: {e}", exc_info=True)
    
    def _on_training_completed(self, experiment_id, success, message):
        """Обработчик завершения тренировки"""
        self.is_training = False
        self.pushButtonStartTraining.setEnabled(True)
        self.pushButtonStopTraining.setEnabled(False)
        
        if success:
            self.labelTrainingStatus.setText("Тренировка завершена успешно")
            self.progressBarTraining.setValue(100)
            QtWidgets.QMessageBox.information(self, "Успех", f"Тренировка завершена!\n{message}")
        else:
            self.labelTrainingStatus.setText("Тренировка завершена с ошибкой")
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка тренировки:\n{message}")
        
        self.textEditTrainingLog.append(f"Тренировка завершена: {message}")
        self._refresh_experiments_list()
        
        # Автоматически выбираем завершенный эксперимент и обновляем информацию
        if experiment_id and hasattr(self, 'listWidgetExperiments'):
            # Находим элемент в списке с этим experiment_id
            for i in range(self.listWidgetExperiments.count()):
                item = self.listWidgetExperiments.item(i)
                if item and item.data(Qt.UserRole) == experiment_id:
                    self.listWidgetExperiments.setCurrentItem(item)
                    # Вызываем обработчик выбора эксперимента
                    self._on_experiment_selected()
                    break
    
    def _on_status_message(self, text: str):
        """Выводит произвольные статусные сообщения из тренера"""
        try:
            if hasattr(self, 'labelTrainingStatus'):
                self.labelTrainingStatus.setText(text)
            if hasattr(self, 'textEditTrainingLog'):
                self.textEditTrainingLog.append(text)
        except Exception as e:
            logger.error(f"Ошибка вывода статусного сообщения: {e}", exc_info=True)
    
    def _on_validation_completed(self, experiment_id, results):
        """Обработчик завершения валидации"""
        try:
            if 'error' in results:
                QtWidgets.QMessageBox.critical(self, "Ошибка валидации", results['error'])
                return
            
            # Отображаем результаты валидации
            self._display_validation_results(results)
            
        except Exception as e:
            logger.error(f"Ошибка обработки результатов валидации: {e}", exc_info=True)
    
    def _update_metrics_table(self, epoch, metrics):
        """Обновляет таблицу метрик (добавляет новую строку)"""
        try:
            # Добавляем новую строку в таблицу
            row_count = self.tableWidgetMetrics.rowCount()
            self.tableWidgetMetrics.insertRow(row_count)
            
            # Заполняем данные
            self.tableWidgetMetrics.setItem(row_count, 0, QtWidgets.QTableWidgetItem(str(epoch)))
            self.tableWidgetMetrics.setItem(row_count, 1, QtWidgets.QTableWidgetItem(f"{metrics.get('mAP50', 0):.4f}"))
            self.tableWidgetMetrics.setItem(row_count, 2, QtWidgets.QTableWidgetItem(f"{metrics.get('mAP50-95', 0):.4f}"))
            self.tableWidgetMetrics.setItem(row_count, 3, QtWidgets.QTableWidgetItem(f"{metrics.get('precision', 0):.4f}"))
            self.tableWidgetMetrics.setItem(row_count, 4, QtWidgets.QTableWidgetItem(f"{metrics.get('recall', 0):.4f}"))
            self.tableWidgetMetrics.setItem(row_count, 5, QtWidgets.QTableWidgetItem(f"{metrics.get('f1_score', 0):.4f}"))
            self.tableWidgetMetrics.setItem(row_count, 6, QtWidgets.QTableWidgetItem(f"{metrics.get('loss', 0):.4f}"))
            
            # Прокручиваем к последней строке
            self.tableWidgetMetrics.scrollToBottom()
            
        except Exception as e:
            logger.error(f"Ошибка обновления таблицы метрик: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
    
    def _load_metrics_from_database(self, experiment_id):
        """Загружает метрики из базы данных и заполняет таблицу"""
        try:
            if not self.training_manager or not experiment_id:
                return
            
            # Очищаем таблицу
            self.tableWidgetMetrics.setRowCount(0)
            
            # Получаем метрики из базы данных
            metrics_data = self.training_manager.metrics_tracker.database.get_experiment_metrics(experiment_id)
            
            if not metrics_data:
                logger.warning(f"Метрики для эксперимента {experiment_id} не найдены в базе данных")
                return
            
            # Группируем метрики по эпохам
            epochs_data = {}
            for metric in metrics_data:
                epoch = metric['epoch']
                phase = metric['phase']
                metric_name = metric['metric_name']
                metric_value = metric['metric_value']
                
                if epoch not in epochs_data:
                    epochs_data[epoch] = {'training': {}, 'validation': {}}
                
                epochs_data[epoch][phase][metric_name] = metric_value
            
            # Заполняем таблицу метриками по эпохам
            for epoch in sorted(epochs_data.keys()):
                epoch_data = epochs_data[epoch]
                validation_metrics = epoch_data.get('validation', {})
                training_metrics = epoch_data.get('training', {})
                
                # Формируем строку метрик для таблицы
                # Извлекаем loss из training метрик (может быть под разными именами)
                loss_value = 0.0
                for loss_key in ['loss', 'train_loss', 'box_loss', 'cls_loss', 'dfl_loss']:
                    if loss_key in training_metrics:
                        # Если есть общий loss, используем его, иначе суммируем компоненты
                        if loss_key == 'loss':
                            loss_value = training_metrics[loss_key]
                            break
                        else:
                            loss_value += training_metrics.get(loss_key, 0.0)
                
                # Если не нашли loss, пробуем найти любую метрику с loss в названии
                if loss_value == 0.0:
                    for key, value in training_metrics.items():
                        if 'loss' in key.lower():
                            loss_value = value
                            break
                
                precision = validation_metrics.get('precision', 0.0)
                recall = validation_metrics.get('recall', 0.0)
                
                # Вычисляем F1-score если его нет
                f1_score = validation_metrics.get('f1_score', 0.0)
                if f1_score == 0.0 and precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                
                row_metrics = {
                    'mAP50': validation_metrics.get('mAP50', 0.0),
                    'mAP50-95': validation_metrics.get('mAP50-95', validation_metrics.get('mAP50_95', validation_metrics.get('map', 0.0))),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'loss': loss_value
                }
                
                # Добавляем строку в таблицу
                row_count = self.tableWidgetMetrics.rowCount()
                self.tableWidgetMetrics.insertRow(row_count)
                
                self.tableWidgetMetrics.setItem(row_count, 0, QtWidgets.QTableWidgetItem(str(epoch)))
                self.tableWidgetMetrics.setItem(row_count, 1, QtWidgets.QTableWidgetItem(f"{row_metrics['mAP50']:.4f}"))
                self.tableWidgetMetrics.setItem(row_count, 2, QtWidgets.QTableWidgetItem(f"{row_metrics['mAP50-95']:.4f}"))
                self.tableWidgetMetrics.setItem(row_count, 3, QtWidgets.QTableWidgetItem(f"{row_metrics['precision']:.4f}"))
                self.tableWidgetMetrics.setItem(row_count, 4, QtWidgets.QTableWidgetItem(f"{row_metrics['recall']:.4f}"))
                self.tableWidgetMetrics.setItem(row_count, 5, QtWidgets.QTableWidgetItem(f"{row_metrics['f1_score']:.4f}"))
                self.tableWidgetMetrics.setItem(row_count, 6, QtWidgets.QTableWidgetItem(f"{row_metrics['loss']:.4f}"))
            
            # Прокручиваем к последней строке
            if self.tableWidgetMetrics.rowCount() > 0:
                self.tableWidgetMetrics.scrollToBottom()
            
        except Exception as e:
            logger.error(f"Ошибка загрузки метрик из базы данных: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
    
    def _refresh_experiments_list(self):
        """Обновляет список экспериментов"""
        try:
            if not self.training_manager:
                return
            
            # Сохраняем текущий выбранный эксперимент
            current_item = self.listWidgetExperiments.currentItem()
            current_experiment_id = current_item.data(Qt.UserRole) if current_item else None
            
            experiments = self.training_manager.get_all_experiments()
            self.listWidgetExperiments.clear()
            
            selected_index = None
            for i, exp in enumerate(experiments):
                item_text = f"{exp.get('name', 'Unknown')} - {exp.get('status', 'Unknown')}"
                item = QtWidgets.QListWidgetItem(item_text)
                exp_id = exp.get('id', '')
                item.setData(Qt.UserRole, exp_id)
                self.listWidgetExperiments.addItem(item)
                
                # Восстанавливаем выбор, если это был выбранный эксперимент
                if current_experiment_id and exp_id == current_experiment_id:
                    selected_index = i
            
            # Восстанавливаем выбор и загружаем метрики
            if selected_index is not None:
                self.listWidgetExperiments.setCurrentRow(selected_index)
                self._on_experiment_selected()
                
        except Exception as e:
            logger.error(f"Ошибка обновления списка экспериментов: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
    
    def _on_experiment_selected(self):
        """Обработчик выбора эксперимента"""
        try:
            current_item = self.listWidgetExperiments.currentItem()
            if not current_item:
                return
            
            experiment_id = current_item.data(Qt.UserRole)
            if not experiment_id:
                return
            
            # Получаем информацию об эксперименте
            summary = self.training_manager.get_experiment_summary(experiment_id)
            
            # Отображаем информацию
            info_text = self._format_experiment_info(summary)
            self.textEditExperimentInfo.setPlainText(info_text)
            
            # Загружаем и отображаем метрики из базы данных
            self._load_metrics_from_database(experiment_id)
            
        except Exception as e:
            logger.error(f"Ошибка выбора эксперимента: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
    
    def _format_experiment_info(self, summary):
        """Форматирует информацию об эксперименте"""
        try:
            if not summary:
                return "Информация недоступна"
            
            text = f"=== ИНФОРМАЦИЯ ОБ ЭКСПЕРИМЕНТЕ ===\n\n"
            text += f"ID: {summary.get('id', 'N/A')}\n"
            text += f"Название: {summary.get('name', 'N/A')}\n"
            text += f"Тип задачи: {summary.get('task', 'N/A')}\n"
            text += f"Тип модели: {summary.get('model_type', 'N/A')}\n"
            text += f"Статус: {summary.get('status', 'N/A')}\n"
            text += f"Дата создания: {summary.get('created_at', 'N/A')}\n"
            text += f"Дата завершения: {summary.get('completed_at', 'N/A')}\n\n"
            
            # Параметры
            if 'config' in summary:
                config = summary['config']
                text += "=== ПАРАМЕТРЫ ===\n"
                text += f"Эпохи: {config.get('epochs', 'N/A')}\n"
                text += f"Размер батча: {config.get('batch_size', 'N/A')}\n"
                text += f"Размер изображения: {config.get('image_size', 'N/A')}\n"
                text += f"Скорость обучения: {config.get('learning_rate', 'N/A')}\n"
                text += f"Устройство: {config.get('device', 'N/A')}\n\n"
            
            # Метрики
            if 'final_metrics' in summary:
                metrics = summary['final_metrics']
                text += "=== ФИНАЛЬНЫЕ МЕТРИКИ ===\n"
                for key, value in metrics.items():
                    text += f"{key}: {value}\n"
            
            return text
            
        except Exception as e:
            return f"Ошибка форматирования: {e}"
    
    # Заглушки для остальных методов
    def _start_validation(self):
        """Запускает валидацию модели"""
        try:
            if not self.training_manager:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Компоненты тренировки не инициализированы")
                return
            
            # Получаем пути модели и датасета из вкладки валидации
            model_path = ""
            dataset_path = ""
            try:
                if hasattr(self, 'fileWidgetValidationModel'):
                    model_path = self.fileWidgetValidationModel.filePath()
                if hasattr(self, 'fileWidgetValidationDataset'):
                    dataset_path = self.fileWidgetValidationDataset.filePath()
            except Exception:
                pass
            
            if not model_path:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Укажите путь к модели (*.pt)")
                return
            if not dataset_path:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Укажите путь к датасету")
                return
            
            # Определяем тип задачи из UI
            task_type = self.comboBoxTaskType.currentText() if hasattr(self, 'comboBoxTaskType') else "Детекция"
            task = "detect" if "Детекция" in task_type else "segment"
            
            # Запускаем валидацию (комплексную по умолчанию)
            results = self.training_manager.validate_model(
                model_path=model_path,
                dataset_path=dataset_path,
                task=task,
                experiment_id=None,
                comprehensive=True
            )
            
            if 'error' in results:
                QtWidgets.QMessageBox.critical(self, "Ошибка валидации", results['error'])
                return
            
            # Отобразить результаты
            self._display_validation_results(results)
            
            # Предложить сохранить результаты
            if hasattr(self, 'checkBoxSaveValidation') and self.checkBoxSaveValidation.isChecked():
                self._export_validation_results(results)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка запуска валидации: {e}")
    
    def _compare_models(self):
        """Сравнивает модели"""
        try:
            # Запросить у пользователя список моделей для сравнения
            models_text, ok = QtWidgets.QInputDialog.getMultiLineText(
                self,
                "Сравнение моделей",
                "Укажите пути к моделям (по одной на строке):"
            )
            if not ok or not models_text.strip():
                return
            
            model_paths = [p.strip() for p in models_text.splitlines() if p.strip()]
            if len(model_paths) < 2:
                QtWidgets.QMessageBox.warning(self, "Предупреждение", "Нужно указать минимум две модели")
                return
            
            dataset_path = ""
            try:
                if hasattr(self, 'fileWidgetValidationDataset'):
                    dataset_path = self.fileWidgetValidationDataset.filePath()
            except Exception:
                pass
            if not dataset_path:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Укажите путь к датасету")
                return
            
            task_type = self.comboBoxTaskType.currentText() if hasattr(self, 'comboBoxTaskType') else "Детекция"
            task = "detect" if "Детекция" in task_type else "segment"
            
            models = [{'path': p, 'name': os.path.splitext(os.path.basename(p))[0]} for p in model_paths]
            comparison = self.training_manager.compare_models(models=models, dataset_path=dataset_path, task=task)
            
            if 'error' in comparison:
                QtWidgets.QMessageBox.critical(self, "Ошибка сравнения", comparison['error'])
                return
            
            # Краткий вывод сравнения
            summary_lines = ["=== Сравнение моделей ==="]
            best = comparison.get('comparison_metrics', {})
            for key, data in best.items():
                summary_lines.append(f"{key}: {data.get('model', 'N/A')} ({data.get('value', 0):.4f})")
            self.textEditValidationLog.append("\n".join(summary_lines))
            QtWidgets.QMessageBox.information(self, "Сравнение завершено", "\n".join(summary_lines))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка сравнения моделей: {e}")
    
    def _export_validation_results(self, results: dict = None):
        """Экспортирует результаты валидации"""
        try:
            # Если результаты не переданы, пробуем собрать из таблицы
            data = results if isinstance(results, dict) else None
            if data is None:
                # Собираем из виджета (минимально)
                metrics = {}
                rows = self.tableWidgetValidationResults.rowCount() if hasattr(self, 'tableWidgetValidationResults') else 0
                for i in range(rows):
                    name_item = self.tableWidgetValidationResults.item(i, 0)
                    val_item = self.tableWidgetValidationResults.item(i, 1)
                    if name_item and val_item:
                        try:
                            metrics[name_item.text()] = float(val_item.text())
                        except Exception:
                            metrics[name_item.text()] = val_item.text()
                data = {'metrics': metrics, 'timestamp': datetime.now().isoformat()}
            
            # Диалог выбора пути
            default_name = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Сохранить результаты валидации", default_name, "JSON (*.json)"
            )
            if not out_path:
                return
            
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QtWidgets.QMessageBox.information(self, "Успех", f"Результаты сохранены в:\n{out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта результатов: {e}")
    
    def _delete_experiment(self):
        """Удаляет выбранный эксперимент"""
        QtWidgets.QMessageBox.information(self, "Информация", "Функция удаления экспериментов будет реализована")
    
    def _export_experiment(self):
        """Экспортирует данные эксперимента"""
        QtWidgets.QMessageBox.information(self, "Информация", "Функция экспорта эксперимента будет реализована")
    
    def _generate_plots(self):
        """Генерирует графики метрик"""
        QtWidgets.QMessageBox.information(self, "Информация", "Функция генерации графиков будет реализована")
    
    def _save_plots(self):
        """Сохраняет графики"""
        QtWidgets.QMessageBox.information(self, "Информация", "Функция сохранения графиков будет реализована")
    
    def _display_validation_results(self, results):
        """Отображает результаты валидации"""
        try:
            # Обновляем таблицу результатов валидации
            self.tableWidgetValidationResults.setRowCount(0)
            
            if 'metrics' in results:
                metrics = results['metrics']
                for i, (metric_name, value) in enumerate(metrics.items()):
                    self.tableWidgetValidationResults.insertRow(i)
                    self.tableWidgetValidationResults.setItem(i, 0, QtWidgets.QTableWidgetItem(metric_name))
                    self.tableWidgetValidationResults.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{value:.4f}"))
                    self.tableWidgetValidationResults.setItem(i, 2, QtWidgets.QTableWidgetItem(self._get_metric_description(metric_name)))
            
            # Добавляем информацию в лог
            self.textEditValidationLog.append(f"Валидация завершена: {results.get('timestamp', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Ошибка отображения результатов валидации: {e}", exc_info=True)
    
    def _get_metric_description(self, metric_name):
        """Возвращает описание метрики"""
        descriptions = {
            'mAP50': 'Mean Average Precision при IoU=0.5',
            'mAP50-95': 'Mean Average Precision при IoU=0.5-0.95',
            'precision': 'Точность детекции',
            'recall': 'Полнота детекции',
            'f1_score': 'F1-мера (гармоническое среднее точности и полноты)'
        }
        return descriptions.get(metric_name, 'Неизвестная метрика')