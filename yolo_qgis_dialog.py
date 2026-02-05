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
from pathlib import Path

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets, QtGui
from qgis.PyQt.QtCore import Qt

from qgis.core import (
    QgsMapLayerProxyModel,
    QgsProject,
    QgsCoordinateTransform,
    QgsCoordinateReferenceSystem,
    QgsRectangle,
)

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- ИМПОРТ ФУНКЦИЙ ИЗ ВНЕШНИХ МОДУЛЕЙ ---
from .grid_creator import create_grid_layer
from .intersection import perform_intersection
from .map_exporter import export_views
from .dataset_formatter import format_yolo_dataset
from .dataset_formatter_yolo import (
    save_yolo_native_dataset,
)  # <--- ДОБАВЛЕН НОВЫЙ ИМПОРТ
from .processing_utils import ProgressReporter
from .dataset_manager_dialog import DatasetManagerDialog
from .dataset_manager import DatasetManager

# --- ИМПОРТ МОДУЛЕЙ ТРЕНИРОВКИ ---
from .yolo_training_manager import YOLOTrainingManager, TrainingConfigManager
from .yolo_detection_trainer import DetectionTrainer, DetectionDatasetAnalyzer
from .yolo_segmentation_trainer import SegmentationTrainer, SegmentationDatasetAnalyzer
from .yolo_validation import AdvancedValidator, ModelComparator
from .yolo_metrics_tracker import MetricsTracker, MetricsVisualizer
from .path_history_manager import PathHistoryManager

# --- ИМПОРТ МОДУЛЯ ДЕТЕКЦИИ ---
from .yolo_detector import YOLODetector


FORM_CLASS, _ = uic.loadUiType(
    os.path.join(os.path.dirname(__file__), "yolo_qgis_dialog_base.ui")
)


class YoloQgisDialog(QtWidgets.QDialog, FORM_CLASS):
    # ... (весь код __init__, _setup_connections, _update_size_values, update_split_values остается без изменений) ...
    def __init__(self, parent=None, iface=None):
        """Конструктор.
        
        :param parent: Родительский виджет
        :param iface: QGIS interface для доступа к map canvas
        """
        super(YoloQgisDialog, self).__init__(parent)
        self.iface = iface

        self.setupUi(self)

        # Настройка таблицы метрик: добавляем отдельные столбцы для box_loss, cls_loss, dfl_loss
        try:
            if hasattr(self, "tableWidgetMetrics"):
                self.tableWidgetMetrics.setColumnCount(8)
                self.tableWidgetMetrics.setHorizontalHeaderLabels(
                    [
                        "Epoch",
                        "mAP50",
                        "mAP50-95",
                        "Precision",
                        "Recall",
                        "Box loss",
                        "Cls loss",
                        "DFL loss",
                    ]
                )
        except Exception as e:
            logger.warning(f"Не удалось инициализировать таблицу метрик: {e}")

        # Инициализация компонентов тренировки
        self.training_manager = None
        self.config_manager = None
        self.current_experiment_id = None
        self.is_training = False

        # Инициализация детектора
        self.detector = YOLODetector()

        # Инициализация менеджера истории путей
        self.path_history = PathHistoryManager()

        self._setup_connections()
        self._initialize_training_components()
        self._load_path_history()

    def _setup_connections(self):
        """Настройка всех сигналов и слотов."""
        self.buttonBox.accepted.connect(self.run_dataset_creation)
        self.buttonBox.rejected.connect(self.reject)

        self.mMapLayerComboBoxRaster.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.mMapLayerComboBoxObjects.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.mMapLayerComboBoxObjects.layerChanged.connect(
            self.mFieldComboBoxObjects.setLayer
        )
        self.mFieldComboBoxObjects.setLayer(
            self.mMapLayerComboBoxObjects.currentLayer()
        )

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

        # Подключение сигналов для сохранения истории путей
        self._setup_path_history_connections()

        # Подключение кнопок управления датасетами
        self.manageDatasetsButton.clicked.connect(self.open_dataset_manager_dialog)

        # Подключение checkbox для обновления датасета
        self.checkBox_UpdateDataset.toggled.connect(self.toggle_update_mode)

        # Подключение сигналов тренировки
        self._setup_training_connections()

        # Подключение сигналов детекции
        self._setup_detection_connections()

    def _update_size_values(self):
        """Автоматически пересчитывает размеры в метрах или пикселях."""
        sender = self.sender()
        if not sender:
            return
        is_pixel_source = sender in (
            self.lineEdit_WidthPixel,
            self.lineEdit_HeigthPixel,
            self.comboBox_Dpi,
        )
        is_meter_source = sender in (
            self.lineEdit_WidthMeter,
            self.lineEdit_HeigthMeter,
        )
        try:
            dpi = int(self.comboBox_Dpi.currentText())
            if is_pixel_source:
                w_px, h_px = int(self.lineEdit_WidthPixel.text()), int(
                    self.lineEdit_HeigthPixel.text()
                )
                w_m, h_m = (w_px / dpi) * 39.37, (h_px / dpi) * 39.37
                self.lineEdit_WidthMeter.blockSignals(True)
                self.lineEdit_HeigthMeter.blockSignals(True)
                self.lineEdit_WidthMeter.setText(f"{w_m:.4f}")
                self.lineEdit_HeigthMeter.setText(f"{h_m:.4f}")
                self.lineEdit_WidthMeter.blockSignals(False)
                self.lineEdit_HeigthMeter.blockSignals(False)
            elif is_meter_source:
                w_m, h_m = float(self.lineEdit_WidthMeter.text()), float(
                    self.lineEdit_HeigthMeter.text()
                )
                w_px, h_px = round((w_m / 39.37) * dpi), round((h_m / 39.37) * dpi)
                self.lineEdit_WidthPixel.blockSignals(True)
                self.lineEdit_HeigthPixel.blockSignals(True)
                self.lineEdit_WidthPixel.setText(str(w_px))
                self.lineEdit_HeigthPixel.setText(str(h_px))
                self.lineEdit_WidthPixel.blockSignals(False)
                self.lineEdit_HeigthPixel.blockSignals(False)
        except (ValueError, ZeroDivisionError):
            pass

    def update_split_values(self):
        """Перераспределяет значения Train/Val/Test, чтобы сумма была 100."""
        sender = self.sender()
        if not sender:
            return
        self.spinBox_Train.blockSignals(True)
        self.spinBox_Val.blockSignals(True)
        self.spinBox_Test.blockSignals(True)
        try:
            train, val, test = (
                self.spinBox_Train.value(),
                self.spinBox_Val.value(),
                self.spinBox_Test.value(),
            )
            remainder = 100 - sender.value()
            if sender == self.spinBox_Train:
                other_sum = val + test
                if other_sum == 0:
                    new_val, new_test = remainder, 0
                else:
                    ratio = val / other_sum
                    new_val = round(remainder * ratio)
                    new_test = remainder - new_val
                self.spinBox_Val.setValue(new_val)
                self.spinBox_Test.setValue(new_test)
            elif sender == self.spinBox_Val:
                other_sum = train + test
                if other_sum == 0:
                    new_train, new_test = remainder, 0
                else:
                    ratio = train / other_sum
                    new_train = round(remainder * ratio)
                    new_test = remainder - new_train
                self.spinBox_Train.setValue(new_train)
                self.spinBox_Test.setValue(new_test)
            elif sender == self.spinBox_Test:
                other_sum = train + val
                if other_sum == 0:
                    new_train, new_val = remainder, 0
                else:
                    ratio = train / other_sum
                    new_train = round(remainder * ratio)
                    new_val = remainder - new_train
                self.spinBox_Train.setValue(new_train)
                self.spinBox_Val.setValue(new_val)
        finally:
            self.spinBox_Train.blockSignals(False)
            self.spinBox_Val.blockSignals(False)
            self.spinBox_Test.blockSignals(False)

    def toggle_update_mode(self, checked):
        """Переключает режим обновления датасета"""
        try:
            if checked:
                # В режиме обновления меняем подсказку для основного поля
                self.mQgsFileWidget.setToolTip(
                    "Dataset directory - will update existing dataset if found, otherwise create new one"
                )
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
                # Сохраняем путь в историю
                if output_dir:
                    self.path_history.add_dataset_creation_path(output_dir)
            except Exception as e:
                logger.error(
                    f"Ошибка получения пути из QgsFileWidget: {e}", exc_info=True
                )
                output_dir = ""

            objects_layer = self.mMapLayerComboBoxObjects.currentLayer()
            classes_field = self.mFieldComboBoxObjects.currentField()
            is_update_mode = self.checkBox_UpdateDataset.isChecked()

            if not all([output_dir, objects_layer, classes_field]):
                raise ValueError("Необходимо заполнить все поля в группе 'Setup'.")

            # В режиме обновления проверяем, есть ли существующий датасет в указанной директории
            if is_update_mode:
                from .dataset_utils import DatasetUtils

                is_valid_dataset, _ = DatasetUtils.validate_dataset_path(output_dir)
                if not is_valid_dataset:
                    # Если датасета нет, переключаемся в режим создания нового
                    is_update_mode = False
                    logger.info(
                        "В указанной директории не найден существующий датасет. Создается новый датасет."
                    )
                else:
                    logger.info(
                        f"Найден существующий датасет в директории: {output_dir}"
                    )

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

            if any(
                x <= 0 for x in [img_width_m, img_height_m, img_width_px, img_height_px]
            ):
                raise ValueError("Все размеры должны быть больше нуля.")
        except (ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(
                self, "Ошибка ввода", f"Проверьте корректность входных данных:\n{e}"
            )
            return

        # --- ШАГ 1: Создание сетки (0% -> 15%) ---
        logger.info("1. Создание сетки...")
        self.progressBar.setValue(5)
        grid_layer, error_msg = create_grid_layer(
            source_layer=objects_layer,
            h_spacing=img_width_m,
            v_spacing=img_height_m,
            h_overlay=h_overlay_m,
            v_overlay=v_overlay_m,
        )
        if error_msg:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg)
            self.progressBar.setValue(0)
            return
        grid_layer.setName(f"Сетка для '{objects_layer.name()}'")
        QgsProject.instance().addMapLayer(grid_layer)
        self.progressBar.setValue(15)
        logger.info("Сетка создана.")

        # --- ШАГ 2: Выполнение пересечения (15% -> 25%) ---
        logger.info("2. Пересечение объектов с сеткой...")
        self.progressBar.setValue(20)
        intersected_layer, error_msg = perform_intersection(
            input_layer=objects_layer, overlay_layer=grid_layer
        )
        if error_msg:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg)
            self.progressBar.setValue(0)
            return
        intersected_layer.setName(f"Пересечение для '{objects_layer.name()}'")
        QgsProject.instance().addMapLayer(intersected_layer)
        self.progressBar.setValue(25)
        logger.info("Слой пересечения создан.")

        # --- ШАГ 3: Экспорт изображений (25% -> 75%) ---
        logger.info("3. Экспорт изображений тайлов...")

        # В режиме обновления экспортируем изображения в существующий датасет
        if is_update_mode:
            # Создаем временную структуру в существующем датасете
            temp_images_dir = os.path.join(output_dir, "temp_new_data", "images")
            os.makedirs(temp_images_dir, exist_ok=True)
            images_output_dir = temp_images_dir
        else:
            images_output_dir = os.path.join(output_dir, "images")

        progress_reporter_export = ProgressReporter(
            self.progressBar, start_percentage=25, end_percentage=75
        )
        success, error_msg = export_views(
            grid_layer=grid_layer,
            output_dir=images_output_dir,
            image_format=img_format,
            width_px=img_width_px,
            height_px=img_height_px,
            dpi=img_dpi,
            progress_reporter=progress_reporter_export,
        )
        if not success:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg)
            self.progressBar.setValue(0)
            return
        self.progressBar.setValue(75)
        logger.info("Экспорт изображений завершен.")

        # --- ШАГ 4: Формирование датасета (75% -> 100%) ---
        splits = {
            "train": self.spinBox_Train.value(),
            "val": self.spinBox_Val.value(),
            "test": self.spinBox_Test.value(),
        }

        metadata = {
            "task": self.comboBox_TaskDataset.currentText(),
            "name": self.lineEdit_NameDataset.text(),
            "desc": self.lineEdit_DescriptionDataset.text(),
            "url": self.lineEdit_UrlDataset.text(),
        }

        delete_void = self.voidImages.isChecked()
        progress_reporter_format = ProgressReporter(
            self.progressBar, start_percentage=75, end_percentage=100
        )

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
                progress_reporter=progress_reporter_format,
            )
        elif save_in_yolo_format:
            logger.info("4. Формирование датасета в нативном формате YOLO...")
            success, error_msg = save_yolo_native_dataset(
                intersected_layer=intersected_layer,
                grid_layer=grid_layer,
                class_field=classes_field,
                output_dir=output_dir,
                image_format=img_format,
                splits=splits,
                metadata=metadata,
                delete_void=delete_void,
                progress_reporter=progress_reporter_format,
            )
        else:
            logger.info("4. Формирование файла аннотаций data.ndjson...")
            success, error_msg = format_yolo_dataset(
                intersected_layer=intersected_layer,
                grid_layer=grid_layer,
                class_field=classes_field,
                output_dir=output_dir,
                image_format=img_format,
                image_width=img_width_px,
                image_height=img_height_px,
                splits=splits,
                metadata=metadata,
                delete_void=delete_void,
                progress_reporter=progress_reporter_format,
            )
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        if not success:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg)
            self.progressBar.setValue(0)
            return
        logger.info("Формирование датасета завершено.")

        self.progressBar.setValue(100)
        logger.info("--- Процесс успешно завершен ---")
        if is_update_mode:
            QtWidgets.QMessageBox.information(
                self, "Готово", f"Датасет успешно обновлен!\nОбновлен в: {output_dir}"
            )
        else:
            QtWidgets.QMessageBox.information(
                self, "Готово", f"Датасет успешно создан!\nСохранено в: {output_dir}"
            )
        # self.accept()  # Убрано, чтобы окно не закрывалось после выполнения

    def update_existing_dataset(
        self,
        existing_dataset_path,
        intersected_layer,
        grid_layer,
        class_field,
        image_format,
        splits,
        metadata,
        delete_void,
        progress_reporter,
    ):
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
            all_classes = [
                str(val)
                for val in unique_values
                if val is not None and str(val).strip()
            ]

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
                metadata=metadata,
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
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка открытия диалога управления: {e}"
            )

    def _setup_path_history_connections(self):
        """Настраивает соединения для сохранения истории путей"""
        try:
            # Подключаем сигналы fileChanged для автоматического сохранения
            if hasattr(self, "mQgsFileWidget"):
                self.mQgsFileWidget.fileChanged.connect(
                    lambda: self._save_path_to_history(
                        "dataset_creation", self.mQgsFileWidget.filePath()
                    )
                )
            if hasattr(self, "fileWidgetDataset"):
                self.fileWidgetDataset.fileChanged.connect(
                    lambda: self._save_path_to_history(
                        "dataset", self.fileWidgetDataset.filePath()
                    )
                )
            if hasattr(self, "fileWidgetSaveDir"):
                self.fileWidgetSaveDir.fileChanged.connect(
                    lambda: self._save_path_to_history(
                        "save_dir", self.fileWidgetSaveDir.filePath()
                    )
                )
            if hasattr(self, "fileWidgetValidationModel"):
                self.fileWidgetValidationModel.fileChanged.connect(
                    lambda: self._save_path_to_history(
                        "model", self.fileWidgetValidationModel.filePath()
                    )
                )
            if hasattr(self, "fileWidgetValidationDataset"):
                self.fileWidgetValidationDataset.fileChanged.connect(
                    lambda: self._save_path_to_history(
                        "dataset", self.fileWidgetValidationDataset.filePath()
                    )
                )
        except Exception as e:
            logger.error(
                f"Ошибка настройки соединений истории путей: {e}", exc_info=True
            )

    def _load_path_history(self):
        """Загружает последние использованные пути"""
        try:
            # Загружаем последний путь для создания датасета
            last_path = self.path_history.get_last_dataset_creation_path()
            if last_path and hasattr(self, "mQgsFileWidget"):
                try:
                    self.mQgsFileWidget.setFilePath(last_path)
                except Exception:
                    pass

            # Загружаем последний путь к датасету для обучения
            last_dataset = self.path_history.get_last_dataset_path()
            if last_dataset and hasattr(self, "fileWidgetDataset"):
                try:
                    self.fileWidgetDataset.setFilePath(last_dataset)
                except Exception:
                    pass

            # Загружаем последний путь к директории сохранения
            last_save_dir = self.path_history.get_last_save_dir_path()
            if last_save_dir and hasattr(self, "fileWidgetSaveDir"):
                try:
                    self.fileWidgetSaveDir.setFilePath(last_save_dir)
                except Exception:
                    pass

            # Загружаем последний путь к модели
            last_model = self.path_history.get_last_model_path()
            if last_model and hasattr(self, "fileWidgetValidationModel"):
                try:
                    self.fileWidgetValidationModel.setFilePath(last_model)
                except Exception:
                    pass

            # Загружаем последний путь к датасету для валидации
            if last_dataset and hasattr(self, "fileWidgetValidationDataset"):
                try:
                    self.fileWidgetValidationDataset.setFilePath(last_dataset)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Ошибка загрузки истории путей: {e}", exc_info=True)

    def _save_path_to_history(self, path_type: str, path: str):
        """
        Сохраняет путь в историю

        :param path_type: Тип пути ('dataset', 'model', 'save_dir', 'dataset_creation')
        :param path: Путь для сохранения
        """
        try:
            if not path:
                return

            if path_type == "dataset":
                self.path_history.add_dataset_path(path)
            elif path_type == "model":
                self.path_history.add_model_path(path)
            elif path_type == "save_dir":
                self.path_history.add_save_dir_path(path)
            elif path_type == "dataset_creation":
                self.path_history.add_dataset_creation_path(path)
        except Exception as e:
            logger.error(f"Ошибка сохранения пути в историю: {e}", exc_info=True)

    def _initialize_training_components(self):
        """Инициализирует компоненты системы тренировки"""
        try:
            # Создаем менеджер тренировки
            self.training_manager = YOLOTrainingManager()

            # Создаем менеджер конфигураций
            self.config_manager = TrainingConfigManager()

            # Подключаем сигналы менеджера тренировки
            self.training_manager.training_started.connect(self._on_training_started)
            self.training_manager.training_progress.connect(self._on_training_progress)
            self.training_manager.training_completed.connect(
                self._on_training_completed
            )
            self.training_manager.validation_completed.connect(
                self._on_validation_completed
            )
            # Подключаем статусные сообщения тренировки
            self.training_manager.status_message.connect(self._on_status_message)

            # Обновляем список экспериментов
            self._refresh_experiments_list()

        except Exception as e:
            logger.error(
                f"Ошибка инициализации компонентов тренировки: {e}", exc_info=True
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Предупреждение",
                f"Некоторые функции тренировки могут быть недоступны: {e}",
            )

    def _setup_training_connections(self):
        """Настраивает соединения для интерфейса тренировки"""
        try:
            # Безопасное подключение кнопок тренировки
            if hasattr(self, "pushButtonStartTraining"):
                self.pushButtonStartTraining.clicked.connect(self._start_training)
            if hasattr(self, "pushButtonStopTraining"):
                self.pushButtonStopTraining.clicked.connect(self._stop_training)

            # Безопасное подключение кнопок анализа датасета
            if hasattr(self, "pushButtonAnalyzeDataset"):
                self.pushButtonAnalyzeDataset.clicked.connect(self._analyze_dataset)

            # Безопасное подключение кнопок конфигурации
            if hasattr(self, "pushButtonLoadConfig"):
                self.pushButtonLoadConfig.clicked.connect(self._load_training_config)
            if hasattr(self, "pushButtonSaveConfig"):
                self.pushButtonSaveConfig.clicked.connect(self._save_training_config)

            # Безопасное подключение кнопок валидации
            if hasattr(self, "pushButtonStartValidation"):
                self.pushButtonStartValidation.clicked.connect(self._start_validation)
            if hasattr(self, "pushButtonCompareModels"):
                self.pushButtonCompareModels.clicked.connect(self._compare_models)
            if hasattr(self, "pushButtonExportResults"):
                self.pushButtonExportResults.clicked.connect(
                    self._export_validation_results
                )

            # Безопасное подключение кнопок метрик
            if hasattr(self, "pushButtonRefreshExperiments"):
                self.pushButtonRefreshExperiments.clicked.connect(
                    self._refresh_experiments_list
                )
            if hasattr(self, "pushButtonDeleteExperiment"):
                self.pushButtonDeleteExperiment.clicked.connect(self._delete_experiment)
            if hasattr(self, "pushButtonExportExperiment"):
                self.pushButtonExportExperiment.clicked.connect(self._export_experiment)
            if hasattr(self, "pushButtonGeneratePlots"):
                # Кнопка больше не нужна, скрываем её из интерфейса
                self.pushButtonGeneratePlots.hide()
            if hasattr(self, "pushButtonSavePlots"):
                self.pushButtonSavePlots.clicked.connect(self._save_plots)

            # Безопасное подключение списка экспериментов
            if hasattr(self, "listWidgetExperiments"):
                self.listWidgetExperiments.itemSelectionChanged.connect(
                    self._on_experiment_selected
                )

            # Безопасное подключение изменения типа задачи
            if hasattr(self, "comboBoxTaskType"):
                self.comboBoxTaskType.currentTextChanged.connect(
                    self._on_task_type_changed
                )

            # Логика взаимного исключения для resume / pretrained
            if hasattr(self, "checkBoxResume") and hasattr(self, "checkBoxPretrained"):

                def _on_resume_state_changed(state):
                    # Если включили resume, отключаем и блокируем "предобученные веса"
                    if bool(state):
                        self.checkBoxPretrained.setChecked(False)
                        self.checkBoxPretrained.setEnabled(False)
                    else:
                        # При выключении resume снова разрешаем управлять флажком предобученных весов
                        self.checkBoxPretrained.setEnabled(True)

                self.checkBoxResume.stateChanged.connect(_on_resume_state_changed)

        except Exception as e:
            logger.error(f"Ошибка настройки соединений тренировки: {e}", exc_info=True)

    def _setup_detection_connections(self):
        """Настраивает соединения для интерфейса детекции"""
        try:
            # Устанавливаем фильтр для комбобокса растровых слоев
            if hasattr(self, "mMapLayerComboBoxDetectionRaster"):
                self.mMapLayerComboBoxDetectionRaster.setFilters(QgsMapLayerProxyModel.RasterLayer)
                # Подключаем сигнал изменения слоя для обновления экстента
                self.mMapLayerComboBoxDetectionRaster.layerChanged.connect(
                    self._on_detection_raster_changed
                )

            # Настраиваем виджет экстента
            if hasattr(self, "mExtentGroupBoxDetection"):
                # Устанавливаем map canvas для доступа к "Map canvas extent"
                if self.iface and hasattr(self.iface, "mapCanvas"):
                    map_canvas = self.iface.mapCanvas()
                    logger.info(f"Map canvas: {map_canvas}")
                    if map_canvas:
                        logger.info(f"Setting map canvas")
                        logger.info(f"Map canvas extent: {map_canvas.extent().asWktCoordinates()}")
                        self.mExtentGroupBoxDetection.setMapCanvas(map_canvas)
                        # Получаем CRS из map settings, так как QgsMapCanvas не имеет метода crs()
                        try:
                            map_crs = map_canvas.mapSettings().destinationCrs()
                            if map_crs.isValid():
                                self.mExtentGroupBoxDetection.setCurrentExtent(map_canvas.extent(), map_crs)
                            else:
                                # Если CRS не валиден, используем CRS проекта
                                project = QgsProject.instance()
                                project_crs = project.crs()
                                if project_crs.isValid():
                                    self.mExtentGroupBoxDetection.setCurrentExtent(map_canvas.extent(), project_crs)
                        except AttributeError:
                            # Fallback: используем CRS проекта если mapSettings недоступен
                            project = QgsProject.instance()
                            project_crs = project.crs()
                            if project_crs.isValid():
                                self.mExtentGroupBoxDetection.setCurrentExtent(map_canvas.extent(), project_crs)
                
                # Устанавливаем текущий слой для доступа к "Current layer extent"
                if self.iface and hasattr(self.iface, "activeLayer"):
                    current_layer = self.iface.activeLayer()
                    if current_layer and current_layer.isValid():
                        # Устанавливаем текущий слой для виджета extent
                        # QgsExtentGroupBox автоматически использует его для "Current layer extent"
                        try:
                            # Метод setCurrentLayer может быть доступен в некоторых версиях QGIS
                            if hasattr(self.mExtentGroupBoxDetection, "setCurrentLayer"):
                                self.mExtentGroupBoxDetection.setCurrentLayer(current_layer)
                        except AttributeError:
                            # Если метод недоступен, виджет будет использовать iface.activeLayer() автоматически
                            pass
                
                # Подключаем сигнал изменения активного слоя для обновления extent
                if self.iface:
                    try:
                        # Подключаемся к сигналу изменения активного слоя через QgsProject
                        QgsProject.instance().layersAdded.connect(self._update_extent_for_current_layer)
                        # Также можно использовать iface.currentLayerChanged если доступен
                        if hasattr(self.iface, "currentLayerChanged"):
                            self.iface.currentLayerChanged.connect(self._update_extent_for_current_layer)
                    except Exception as e:
                        logger.debug(f"Не удалось подключить сигналы для обновления extent: {e}")
                
                # Устанавливаем CRS проекта как выходной CRS для виджета
                # Это гарантирует, что все координаты extent будут в CRS проекта
                project = QgsProject.instance()
                project_crs = project.crs()
                if project_crs.isValid():
                    # Устанавливаем output CRS в CRS проекта
                    # Все extent будут возвращаться в CRS проекта
                    try:
                        if hasattr(self.mExtentGroupBoxDetection, "setOutputCrs"):
                            self.mExtentGroupBoxDetection.setOutputCrs(project_crs)
                    except Exception as e:
                        logger.debug(f"Не удалось установить output CRS: {e}")
                    
                    self.mExtentGroupBoxDetection.setCurrentExtent(
                        project_crs.bounds(), project_crs
                    )

            # Безопасное подключение кнопки детекции
            if hasattr(self, "pushButtonStartDetection"):
                self.pushButtonStartDetection.clicked.connect(self._start_detection)

        except Exception as e:
            logger.error(f"Ошибка настройки соединений детекции: {e}", exc_info=True)

    def _on_detection_raster_changed(self, layer):
        """Обработчик изменения растрового слоя для детекции"""
        try:
            if hasattr(self, "mExtentGroupBoxDetection") and layer:
                # Обновляем экстент виджета на основе выбранного растра
                if layer.isValid():
                    extent = layer.extent()
                    layer_crs = layer.crs()
                    project = QgsProject.instance()
                    project_crs = project.crs()
                    
                    if extent and not extent.isEmpty() and layer_crs.isValid() and project_crs.isValid():
                        # Преобразуем extent из CRS слоя в CRS проекта
                        if layer_crs != project_crs:
                            try:
                                transform = QgsCoordinateTransform(
                                    layer_crs, project_crs, QgsProject.instance()
                                )
                                extent = transform.transformBoundingBox(extent)
                            except Exception as e:
                                logger.warning(
                                    f"Ошибка преобразования extent растра в CRS проекта: {e}. "
                                    f"Используется extent в CRS слоя."
                                )
                                # Используем extent в CRS слоя, если преобразование не удалось
                                self.mExtentGroupBoxDetection.setOriginalExtent(extent, layer_crs)
                                self.mExtentGroupBoxDetection.setCurrentExtent(extent, layer_crs)
                                return
                        
                        # Устанавливаем extent в CRS проекта
                        self.mExtentGroupBoxDetection.setOriginalExtent(extent, project_crs)
                        self.mExtentGroupBoxDetection.setCurrentExtent(extent, project_crs)
        except Exception as e:
            logger.warning(f"Ошибка обновления экстента: {e}", exc_info=True)

    def _update_extent_for_current_layer(self, layer=None):
        """Обновляет extent виджета при изменении текущего активного слоя"""
        try:
            if not hasattr(self, "mExtentGroupBoxDetection"):
                return
            
            # Получаем текущий активный слой
            if layer is None and self.iface:
                layer = self.iface.activeLayer()
            
            if layer and layer.isValid():
                extent = layer.extent()
                layer_crs = layer.crs()
                project = QgsProject.instance()
                project_crs = project.crs()
                
                if extent and not extent.isEmpty() and layer_crs.isValid() and project_crs.isValid():
                    # Преобразуем extent из CRS слоя в CRS проекта
                    if layer_crs != project_crs:
                        try:
                            transform = QgsCoordinateTransform(
                                layer_crs, project_crs, QgsProject.instance()
                            )
                            extent = transform.transformBoundingBox(extent)
                        except Exception as e:
                            logger.debug(
                                f"Ошибка преобразования extent текущего слоя в CRS проекта: {e}"
                            )
                            # Если преобразование не удалось, используем extent в CRS слоя
                            extent_crs = layer_crs
                        else:
                            extent_crs = project_crs
                    else:
                        extent_crs = project_crs
                    
                    # Обновляем extent для опции "Current layer extent"
                    # QgsExtentGroupBox автоматически использует это при выборе "Calculate from layer"
                    try:
                        if hasattr(self.mExtentGroupBoxDetection, "setCurrentLayer"):
                            self.mExtentGroupBoxDetection.setCurrentLayer(layer)
                    except Exception:
                        # Если метод недоступен, виджет будет использовать iface.activeLayer() автоматически
                        pass
        except Exception as e:
            logger.debug(f"Ошибка обновления extent для текущего слоя: {e}")

    def _on_task_type_changed(self, task_type):
        """Обработчик изменения типа задачи"""
        try:
            if not hasattr(self, "comboBoxModelType"):
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
                    "YOLOv11x (максимальная)",
                    "YOLOv26n (быстрая)",
                    "YOLOv26s (сбалансированная)",
                    "YOLOv26m (средняя)",
                    "YOLOv26l (большая)",
                    "YOLOv26x (максимальная)",
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
                    "YOLOv11x-seg (максимальная)",
                    "YOLOv26n-seg (быстрая)",
                    "YOLOv26s-seg (сбалансированная)",
                    "YOLOv26m-seg (средняя)",
                    "YOLOv26l-seg (большая)",
                    "YOLOv26x-seg (максимальная)",
                ]
                self.comboBoxModelType.addItems(models)
        except Exception as e:
            logger.error(f"Ошибка обновления типа задачи: {e}", exc_info=True)

    def _start_training(self):
        """Запускает процесс тренировки"""
        try:
            if self.is_training:
                QtWidgets.QMessageBox.warning(
                    self, "Предупреждение", "Тренировка уже выполняется!"
                )
                return

            # Собираем параметры тренировки
            dataset_path = self.fileWidgetDataset.filePath()
            if not dataset_path:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка", "Выберите путь к датасету!"
                )
                return

            # Сохраняем путь в историю
            self.path_history.add_dataset_path(dataset_path)

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
                    "mosaic": self.doubleSpinBoxMosaic.value(),
                    "mixup": self.doubleSpinBoxMixup.value(),
                    "copy_paste": self.doubleSpinBoxCopyPaste.value(),
                    "fliplr": self.doubleSpinBoxFlipLR.value(),
                    "scale": self.doubleSpinBoxScale.value(),
                    "translate": self.doubleSpinBoxTranslate.value(),
                    "cutmix": self.doubleSpinBoxCutmix.value(),
                }

            # Выходные параметры
            project_name = self.lineEditProjectName.text() or "yolo_training"
            save_dir = self.fileWidgetSaveDir.filePath()

            # Режим перезаписи результатов (exist_ok)
            exist_ok = (
                self.checkBoxExistOk.isChecked()
                if hasattr(self, "checkBoxExistOk")
                else False
            )

            # Режим продолжения обучения с чекпоинта (resume)
            resume = (
                self.checkBoxResume.isChecked()
                if hasattr(self, "checkBoxResume")
                else False
            )

            # Сохраняем путь к директории сохранения в историю
            if save_dir:
                self.path_history.add_save_dir_path(save_dir)

            # Запускаем тренировку
            if task == "detect":
                self.current_experiment_id = (
                    self.training_manager.start_detection_training(
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
                        exist_ok=exist_ok,
                        resume=resume,
                        **augmentation_params,
                    )
                )
            else:
                self.current_experiment_id = (
                    self.training_manager.start_segmentation_training(
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
                        exist_ok=exist_ok,
                        resume=resume,
                        **augmentation_params,
                    )
                )

            if self.current_experiment_id:
                self.is_training = True
                self.pushButtonStartTraining.setEnabled(False)
                self.pushButtonStopTraining.setEnabled(True)
                self.labelTrainingStatus.setText("Тренировка запущена...")
                self.textEditTrainingLog.append(f"Запущена тренировка: {project_name}")
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Ошибка", "Не удалось запустить тренировку!"
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка запуска тренировки: {e}"
            )

    def _stop_training(self):
        """Останавливает процесс тренировки"""
        try:
            if self.current_experiment_id and self.training_manager:
                success = self.training_manager.cancel_training(
                    self.current_experiment_id
                )
                if success:
                    self.is_training = False
                    self.pushButtonStartTraining.setEnabled(True)
                    self.pushButtonStopTraining.setEnabled(False)
                    self.labelTrainingStatus.setText("Тренировка остановлена")
                    self.textEditTrainingLog.append(
                        "Тренировка остановлена пользователем"
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Предупреждение", "Не удалось остановить тренировку"
                    )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка остановки тренировки: {e}"
            )

    def _analyze_dataset(self):
        """Анализирует выбранный датасет"""
        try:
            dataset_path = self.fileWidgetDataset.filePath()
            if not dataset_path:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка", "Выберите путь к датасету!"
                )
                return

            task_type = self.comboBoxTaskType.currentText()
            task = "detect" if "Детекция" in task_type else "segment"

            # Анализируем датасет
            analysis = self.training_manager.analyze_dataset(dataset_path, task)

            if "error" in analysis:
                QtWidgets.QMessageBox.critical(
                    self, "Ошибка анализа", analysis["error"]
                )
                return

            # Отображаем результаты анализа
            self._display_dataset_analysis(analysis)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка анализа датасета: {e}"
            )

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
            if "dataset_info" in analysis:
                info = analysis["dataset_info"]
                text += f"Путь: {info.get('path', 'N/A')}\n"
                text += f"Количество классов: {info.get('nc', 'N/A')}\n"
                text += f"Классы: {list(info.get('names', {}).values())}\n\n"

            # Статистика по сплитам
            if "splits" in analysis:
                text += "=== СТАТИСТИКА ПО СПЛИТАМ ===\n"
                for split_name, split_data in analysis["splits"].items():
                    if "error" not in split_data:
                        text += f"\n{split_name.upper()}:\n"
                        text += f"  Изображений: {split_data.get('image_count', 0)}\n"
                        text += (
                            f"  Аннотаций: {split_data.get('annotation_count', 0)}\n"
                        )
                        text += f"  Объектов: {split_data.get('total_objects', 0)}\n"
                        text += f"  Объектов на изображение: {split_data.get('objects_per_image', 0):.2f}\n"

                        # Распределение по классам
                        if "class_distribution" in split_data:
                            text += f"  Распределение по классам:\n"
                            for class_id, count in split_data[
                                "class_distribution"
                            ].items():
                                text += f"    Класс {class_id}: {count} объектов\n"

            # Общая статистика
            if "total_images" in analysis:
                text += f"\n=== ОБЩАЯ СТАТИСТИКА ===\n"
                text += f"Всего изображений: {analysis.get('total_images', 0)}\n"
                text += f"Всего аннотаций: {analysis.get('total_annotations', 0)}\n"

            return text

        except Exception as e:
            return f"Ошибка форматирования результатов: {e}"

    def _get_model_type_from_text(self, model_text, task):
        """Преобразует текст модели в тип модели

        ВАЖНО: для линеек YOLO11 и YOLO26 официальные имена моделей в ultralytics
        используют формат без буквы 'v' (например, 'yolo11n.pt', 'yolo26n.pt'),
        поэтому здесь мы возвращаем строки без 'v', чтобы загрузка весов проходила
        корректно.
        """
        model_mapping = {
            # Детекция YOLOv8
            "YOLOv8n (быстрая)": "yolov8n",
            "YOLOv8s (сбалансированная)": "yolov8s",
            "YOLOv8m (средняя)": "yolov8m",
            "YOLOv8l (большая)": "yolov8l",
            "YOLOv8x (максимальная)": "yolov8x",
            # Детекция YOLO11
            "YOLOv11n (быстрая)": "yolo11n",
            "YOLOv11s (сбалансированная)": "yolo11s",
            "YOLOv11m (средняя)": "yolo11m",
            "YOLOv11l (большая)": "yolo11l",
            "YOLOv11x (максимальная)": "yolo11x",
            # Детекция YOLO26
            "YOLOv26n (быстрая)": "yolo26n",
            "YOLOv26s (сбалансированная)": "yolo26s",
            "YOLOv26m (средняя)": "yolo26m",
            "YOLOv26l (большая)": "yolo26l",
            "YOLOv26x (максимальная)": "yolo26x",
            # Сегментация YOLOv8
            "YOLOv8n-seg (быстрая)": "yolov8n-seg",
            "YOLOv8s-seg (сбалансированная)": "yolov8s-seg",
            "YOLOv8m-seg (средняя)": "yolov8m-seg",
            "YOLOv8l-seg (большая)": "yolov8l-seg",
            "YOLOv8x-seg (максимальная)": "yolov8x-seg",
            # Сегментация YOLO11
            "YOLOv11n-seg (быстрая)": "yolo11n-seg",
            "YOLOv11s-seg (сбалансированная)": "yolo11s-seg",
            "YOLOv11m-seg (средняя)": "yolo11m-seg",
            "YOLOv11l-seg (большая)": "yolo11l-seg",
            "YOLOv11x-seg (максимальная)": "yolo11x-seg",
            # Сегментация YOLO26
            "YOLOv26n-seg (быстрая)": "yolo26n-seg",
            "YOLOv26s-seg (сбалансированная)": "yolo26s-seg",
            "YOLOv26m-seg (средняя)": "yolo26m-seg",
            "YOLOv26l-seg (большая)": "yolo26l-seg",
            "YOLOv26x-seg (максимальная)": "yolo26x-seg",
        }
        return model_mapping.get(model_text, "yolov8n")

    def _load_training_config(self):
        """Загружает конфигурацию тренировки"""
        try:
            configs = self.config_manager.list_configs()
            if not configs:
                QtWidgets.QMessageBox.information(
                    self, "Информация", "Нет сохраненных конфигураций"
                )
                return

            # Создаем диалог выбора конфигурации
            config_name, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Загрузить конфигурацию",
                "Выберите конфигурацию:",
                configs,
                0,
                False,
            )

            if ok and config_name:
                config = self.config_manager.load_config(config_name)
                if config:
                    self._apply_training_config(config)
                    QtWidgets.QMessageBox.information(
                        self, "Успех", f"Конфигурация '{config_name}' загружена"
                    )
                else:
                    QtWidgets.QMessageBox.critical(
                        self, "Ошибка", "Не удалось загрузить конфигурацию"
                    )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка загрузки конфигурации: {e}"
            )

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
                    QtWidgets.QMessageBox.information(
                        self, "Успех", f"Конфигурация '{config_name}' сохранена"
                    )
                else:
                    QtWidgets.QMessageBox.critical(
                        self, "Ошибка", "Не удалось сохранить конфигурацию"
                    )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка сохранения конфигурации: {e}"
            )

    def _get_current_training_config(self):
        """Получает текущую конфигурацию тренировки"""
        try:
            task_type = self.comboBoxTaskType.currentText()
            task = "detect" if "Детекция" in task_type else "segment"

            model_type_text = self.comboBoxModelType.currentText()
            model_type = self._get_model_type_from_text(model_type_text, task)

            config = {
                "task": task,
                "model_type": model_type,
                "epochs": self.spinBoxEpochs.value(),
                "batch_size": self.spinBoxBatchSize.value(),
                "image_size": self.spinBoxImageSize.value(),
                "learning_rate": self.doubleSpinBoxLearningRate.value(),
                "device": self.comboBoxDevice.currentText(),
                "pretrained": self.checkBoxPretrained.isChecked(),
                "exist_ok": (
                    self.checkBoxExistOk.isChecked()
                    if hasattr(self, "checkBoxExistOk")
                    else False
                ),
                "resume": (
                    self.checkBoxResume.isChecked()
                    if hasattr(self, "checkBoxResume")
                    else False
                ),
                "augmentation": (
                    {
                        "mosaic": self.doubleSpinBoxMosaic.value(),
                        "mixup": self.doubleSpinBoxMixup.value(),
                        "copy_paste": self.doubleSpinBoxCopyPaste.value(),
                        "fliplr": self.doubleSpinBoxFlipLR.value(),
                        "scale": self.doubleSpinBoxScale.value(),
                        "translate": self.doubleSpinBoxTranslate.value(),
                        "cutmix": self.doubleSpinBoxCutmix.value(),
                    }
                    if self.groupBoxAugmentation.isChecked()
                    else {}
                ),
            }

            return config

        except Exception as e:
            logger.error(f"Ошибка получения конфигурации: {e}", exc_info=True)
            return {}

    def _apply_training_config(self, config):
        """Применяет загруженную конфигурацию к интерфейсу"""
        try:
            # Применяем основные параметры
            if "epochs" in config:
                self.spinBoxEpochs.setValue(config["epochs"])
            if "batch_size" in config:
                self.spinBoxBatchSize.setValue(config["batch_size"])
            if "image_size" in config:
                self.spinBoxImageSize.setValue(config["image_size"])
            if "learning_rate" in config:
                self.doubleSpinBoxLearningRate.setValue(config["learning_rate"])
            if "pretrained" in config:
                self.checkBoxPretrained.setChecked(config["pretrained"])

            # Применяем параметры аугментации
            if "augmentation" in config:
                aug = config["augmentation"]
                if "mosaic" in aug:
                    self.doubleSpinBoxMosaic.setValue(aug["mosaic"])
                if "mixup" in aug:
                    self.doubleSpinBoxMixup.setValue(aug["mixup"])
                if "copy_paste" in aug:
                    self.doubleSpinBoxCopyPaste.setValue(aug["copy_paste"])
                if "fliplr" in aug:
                    self.doubleSpinBoxFlipLR.setValue(aug["fliplr"])
                if "scale" in aug:
                    self.doubleSpinBoxScale.setValue(aug["scale"])
                if "translate" in aug:
                    self.doubleSpinBoxTranslate.setValue(aug["translate"])
                if "cutmix" in aug:
                    self.doubleSpinBoxCutmix.setValue(aug["cutmix"])

            # Применяем режим exist_ok (если чекбокс есть в UI)
            if hasattr(self, "checkBoxExistOk") and "exist_ok" in config:
                self.checkBoxExistOk.setChecked(bool(config["exist_ok"]))

            # Применяем режим resume (если чекбокс есть в UI)
            if hasattr(self, "checkBoxResume") and "resume" in config:
                self.checkBoxResume.setChecked(bool(config["resume"]))

                # Синхронизируем состояние с флажком предобученных весов:
                # при активном resume флажок pretrained должен быть выключен и заблокирован
                if hasattr(self, "checkBoxPretrained"):
                    if config["resume"]:
                        self.checkBoxPretrained.setChecked(False)
                        self.checkBoxPretrained.setEnabled(False)
                    else:
                        self.checkBoxPretrained.setEnabled(True)

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
            QtWidgets.QMessageBox.information(
                self, "Успех", f"Тренировка завершена!\n{message}"
            )
        else:
            self.labelTrainingStatus.setText("Тренировка завершена с ошибкой")
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка тренировки:\n{message}"
            )

        self.textEditTrainingLog.append(f"Тренировка завершена: {message}")
        self._refresh_experiments_list()

        # Автоматически выбираем завершенный эксперимент и обновляем информацию
        if experiment_id and hasattr(self, "listWidgetExperiments"):
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
            if hasattr(self, "labelTrainingStatus"):
                self.labelTrainingStatus.setText(text)
            if hasattr(self, "textEditTrainingLog"):
                self.textEditTrainingLog.append(text)
        except Exception as e:
            logger.error(f"Ошибка вывода статусного сообщения: {e}", exc_info=True)

    def _on_validation_completed(self, experiment_id, results):
        """Обработчик завершения валидации"""
        try:
            if "error" in results:
                QtWidgets.QMessageBox.critical(
                    self, "Ошибка валидации", results["error"]
                )
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
            self.tableWidgetMetrics.setItem(
                row_count, 0, QtWidgets.QTableWidgetItem(str(epoch))
            )
            self.tableWidgetMetrics.setItem(
                row_count,
                1,
                QtWidgets.QTableWidgetItem(f"{metrics.get('mAP50', 0):.4f}"),
            )
            self.tableWidgetMetrics.setItem(
                row_count,
                2,
                QtWidgets.QTableWidgetItem(f"{metrics.get('mAP50-95', 0):.4f}"),
            )
            self.tableWidgetMetrics.setItem(
                row_count,
                3,
                QtWidgets.QTableWidgetItem(f"{metrics.get('precision', 0):.4f}"),
            )
            self.tableWidgetMetrics.setItem(
                row_count,
                4,
                QtWidgets.QTableWidgetItem(f"{metrics.get('recall', 0):.4f}"),
            )
            self.tableWidgetMetrics.setItem(
                row_count,
                5,
                QtWidgets.QTableWidgetItem(f"{metrics.get('box_loss', 0):.4f}"),
            )
            self.tableWidgetMetrics.setItem(
                row_count,
                6,
                QtWidgets.QTableWidgetItem(f"{metrics.get('cls_loss', 0):.4f}"),
            )
            self.tableWidgetMetrics.setItem(
                row_count,
                7,
                QtWidgets.QTableWidgetItem(f"{metrics.get('dfl_loss', 0):.4f}"),
            )

            # Прокручиваем к последней строке
            self.tableWidgetMetrics.scrollToBottom()

        except Exception as e:
            logger.error(f"Ошибка обновления таблицы метрик: {e}", exc_info=True)
            import traceback

            traceback.print_exc()

    def _normalize_metric_name(self, metric_name):
        """Нормализует имя метрики, удаляя префиксы и суффиксы"""
        if not metric_name:
            return metric_name

        # Удаляем префиксы metrics/ и val/
        normalized = metric_name
        if normalized.startswith("metrics/"):
            normalized = normalized[8:]  # Удаляем 'metrics/'
        elif normalized.startswith("val/"):
            normalized = normalized[4:]  # Удаляем 'val/'

        # Удаляем суффикс (B) или (B-1) и т.д.
        if normalized.endswith("(B)"):
            normalized = normalized[:-3]
        elif "(" in normalized:
            # Удаляем скобки с содержимым в конце
            idx = normalized.rfind("(")
            if idx > 0:
                normalized = normalized[:idx]

        return normalized.strip()

    def _load_metrics_from_database(self, experiment_id):
        """Загружает метрики из базы данных и заполняет таблицу"""
        try:
            if not self.training_manager or not experiment_id:
                return

            # Очищаем таблицу
            self.tableWidgetMetrics.setRowCount(0)

            # Получаем метрики из базы данных
            metrics_data = (
                self.training_manager.metrics_tracker.database.get_experiment_metrics(
                    experiment_id
                )
            )

            if not metrics_data:
                logger.warning(
                    f"Метрики для эксперимента {experiment_id} не найдены в базе данных"
                )
                return

            # Группируем метрики по эпохам с нормализацией имен
            epochs_data = {}
            for metric in metrics_data:
                epoch = metric["epoch"]
                phase = metric["phase"]
                metric_name = metric["metric_name"]
                metric_value = metric["metric_value"]

                # Нормализуем имя метрики
                normalized_name = self._normalize_metric_name(metric_name)

                # Определяем правильную фазу для loss метрик с префиксом val/
                if metric_name.startswith("val/"):
                    phase = "validation"
                elif metric_name.startswith("metrics/"):
                    phase = "validation"

                if epoch not in epochs_data:
                    epochs_data[epoch] = {"training": {}, "validation": {}}

                epochs_data[epoch][phase][normalized_name] = metric_value
                # Также сохраняем оригинальное имя для обратной совместимости
                epochs_data[epoch][phase][metric_name] = metric_value

            # Заполняем таблицу метриками по эпохам
            for epoch in sorted(epochs_data.keys()):
                epoch_data = epochs_data[epoch]
                validation_metrics = epoch_data.get("validation", {})
                training_metrics = epoch_data.get("training", {})

                # Формируем строку метрик для таблицы
                # Извлекаем метрики валидации (с нормализованными именами)
                map50 = validation_metrics.get(
                    "mAP50", validation_metrics.get("map50", 0.0)
                )
                map50_95 = validation_metrics.get(
                    "mAP50-95",
                    validation_metrics.get(
                        "mAP50_95",
                        validation_metrics.get(
                            "map", validation_metrics.get("map50-95", 0.0)
                        ),
                    ),
                )
                precision = validation_metrics.get("precision", 0.0)
                recall = validation_metrics.get("recall", 0.0)

                # Извлекаем loss-компоненты
                box_loss = validation_metrics.get(
                    "box_loss", training_metrics.get("box_loss", 0.0)
                )
                cls_loss = validation_metrics.get(
                    "cls_loss", training_metrics.get("cls_loss", 0.0)
                )
                dfl_loss = validation_metrics.get(
                    "dfl_loss", training_metrics.get("dfl_loss", 0.0)
                )

                # Извлекаем суммарный loss из validation метрик (val/box_loss, val/cls_loss, val/dfl_loss)
                loss_value = 0.0
                loss_keys = ["box_loss", "cls_loss", "dfl_loss"]
                for loss_key in loss_keys:
                    if loss_key in validation_metrics:
                        loss_value += validation_metrics.get(loss_key, 0.0)

                # Если не нашли в validation, пробуем training метрики
                if loss_value == 0.0:
                    for loss_key in [
                        "loss",
                        "train_loss",
                        "box_loss",
                        "cls_loss",
                        "dfl_loss",
                    ]:
                        if loss_key in training_metrics:
                            if loss_key == "loss":
                                loss_value = training_metrics[loss_key]
                                break
                            else:
                                loss_value += training_metrics.get(loss_key, 0.0)

                # Если все еще не нашли loss, пробуем найти любую метрику с loss в названии
                if loss_value == 0.0:
                    for key, value in validation_metrics.items():
                        if "loss" in key.lower():
                            loss_value = value
                            break
                    if loss_value == 0.0:
                        for key, value in training_metrics.items():
                            if "loss" in key.lower():
                                loss_value = value
                                break
                row_metrics = {
                    "mAP50": map50,
                    "mAP50-95": map50_95,
                    "precision": precision,
                    "recall": recall,
                    "box_loss": box_loss,
                    "cls_loss": cls_loss,
                    "dfl_loss": dfl_loss,
                    "loss": loss_value,
                }

                # Добавляем строку в таблицу
                row_count = self.tableWidgetMetrics.rowCount()
                self.tableWidgetMetrics.insertRow(row_count)

                self.tableWidgetMetrics.setItem(
                    row_count, 0, QtWidgets.QTableWidgetItem(str(epoch))
                )
                self.tableWidgetMetrics.setItem(
                    row_count,
                    1,
                    QtWidgets.QTableWidgetItem(f"{row_metrics['mAP50']:.4f}"),
                )
                self.tableWidgetMetrics.setItem(
                    row_count,
                    2,
                    QtWidgets.QTableWidgetItem(f"{row_metrics['mAP50-95']:.4f}"),
                )
                self.tableWidgetMetrics.setItem(
                    row_count,
                    3,
                    QtWidgets.QTableWidgetItem(f"{row_metrics['precision']:.4f}"),
                )
                self.tableWidgetMetrics.setItem(
                    row_count,
                    4,
                    QtWidgets.QTableWidgetItem(f"{row_metrics['recall']:.4f}"),
                )
                self.tableWidgetMetrics.setItem(
                    row_count,
                    5,
                    QtWidgets.QTableWidgetItem(f"{row_metrics['box_loss']:.4f}"),
                )
                self.tableWidgetMetrics.setItem(
                    row_count,
                    6,
                    QtWidgets.QTableWidgetItem(f"{row_metrics['cls_loss']:.4f}"),
                )
                self.tableWidgetMetrics.setItem(
                    row_count,
                    7,
                    QtWidgets.QTableWidgetItem(f"{row_metrics['dfl_loss']:.4f}"),
                )

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
            current_experiment_id = (
                current_item.data(Qt.UserRole) if current_item else None
            )

            experiments = self.training_manager.get_all_experiments()
            self.listWidgetExperiments.clear()

            selected_index = None
            for i, exp in enumerate(experiments):
                item_text = (
                    f"{exp.get('name', 'Unknown')} - {exp.get('status', 'Unknown')}"
                )
                item = QtWidgets.QListWidgetItem(item_text)
                exp_id = exp.get("id", "")
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

            # Автоматически генерируем и отображаем графики для выбранного эксперимента
            self._generate_plots(silent=True)

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
            if "config" in summary:
                config = summary["config"]
                text += "=== ПАРАМЕТРЫ ===\n"
                text += f"Эпохи: {config.get('epochs', 'N/A')}\n"
                text += f"Размер батча: {config.get('batch_size', 'N/A')}\n"
                text += f"Размер изображения: {config.get('image_size', 'N/A')}\n"
                text += f"Скорость обучения: {config.get('learning_rate', 'N/A')}\n"
                text += f"Устройство: {config.get('device', 'N/A')}\n\n"

            # Метрики
            if "final_metrics" in summary:
                metrics = summary["final_metrics"]
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
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка", "Компоненты тренировки не инициализированы"
                )
                return

            # Получаем пути модели и датасета из вкладки валидации
            model_path = ""
            dataset_path = ""
            try:
                if hasattr(self, "fileWidgetValidationModel"):
                    model_path = self.fileWidgetValidationModel.filePath()
                    if model_path:
                        self.path_history.add_model_path(model_path)
                if hasattr(self, "fileWidgetValidationDataset"):
                    dataset_path = self.fileWidgetValidationDataset.filePath()
                    if dataset_path:
                        self.path_history.add_dataset_path(dataset_path)
            except Exception:
                pass

            if not model_path:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка", "Укажите путь к модели (*.pt)"
                )
                return
            if not dataset_path:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Укажите путь к датасету")
                return

            # Определяем тип задачи из UI
            task_type = (
                self.comboBoxTaskType.currentText()
                if hasattr(self, "comboBoxTaskType")
                else "Детекция"
            )
            task = "detect" if "Детекция" in task_type else "segment"

            # Запускаем валидацию (комплексную по умолчанию)
            results = self.training_manager.validate_model(
                model_path=model_path,
                dataset_path=dataset_path,
                task=task,
                experiment_id=None,
                comprehensive=True,
            )

            if "error" in results:
                QtWidgets.QMessageBox.critical(
                    self, "Ошибка валидации", results["error"]
                )
                return

            # Отобразить результаты
            self._display_validation_results(results)

            # Предложить сохранить результаты
            if (
                hasattr(self, "checkBoxSaveValidation")
                and self.checkBoxSaveValidation.isChecked()
            ):
                self._export_validation_results(results)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка запуска валидации: {e}"
            )

    def _compare_models(self):
        """Сравнивает модели"""
        try:
            # Запросить у пользователя список моделей для сравнения
            models_text, ok = QtWidgets.QInputDialog.getMultiLineText(
                self,
                "Сравнение моделей",
                "Укажите пути к моделям (по одной на строке):",
            )
            if not ok or not models_text.strip():
                return

            model_paths = [p.strip() for p in models_text.splitlines() if p.strip()]
            if len(model_paths) < 2:
                QtWidgets.QMessageBox.warning(
                    self, "Предупреждение", "Нужно указать минимум две модели"
                )
                return

            dataset_path = ""
            try:
                if hasattr(self, "fileWidgetValidationDataset"):
                    dataset_path = self.fileWidgetValidationDataset.filePath()
            except Exception:
                pass
            if not dataset_path:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Укажите путь к датасету")
                return

            task_type = (
                self.comboBoxTaskType.currentText()
                if hasattr(self, "comboBoxTaskType")
                else "Детекция"
            )
            task = "detect" if "Детекция" in task_type else "segment"

            models = [
                {"path": p, "name": os.path.splitext(os.path.basename(p))[0]}
                for p in model_paths
            ]
            comparison = self.training_manager.compare_models(
                models=models, dataset_path=dataset_path, task=task
            )

            if "error" in comparison:
                QtWidgets.QMessageBox.critical(
                    self, "Ошибка сравнения", comparison["error"]
                )
                return

            # Краткий вывод сравнения
            summary_lines = ["=== Сравнение моделей ==="]
            best = comparison.get("comparison_metrics", {})
            for key, data in best.items():
                summary_lines.append(
                    f"{key}: {data.get('model', 'N/A')} ({data.get('value', 0):.4f})"
                )
            self.textEditValidationLog.append("\n".join(summary_lines))
            
            # Предлагаем сохранить результаты сравнения
            reply = QtWidgets.QMessageBox.question(
                self,
                "Сравнение завершено",
                "\n".join(summary_lines) + "\n\nСохранить результаты сравнения?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self._export_comparison_results(comparison)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка сравнения моделей: {e}"
            )

    def _export_comparison_results(self, comparison: dict):
        """Экспортирует результаты сравнения моделей"""
        try:
            # Диалог выбора пути
            default_name = (
                f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Сохранить результаты сравнения", default_name, "JSON (*.json)"
            )
            if not out_path:
                return

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            QtWidgets.QMessageBox.information(
                self, "Успех", f"Результаты сравнения сохранены в:\n{out_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка экспорта результатов сравнения: {e}"
            )

    def _export_validation_results(self, results: dict = None):
        """Экспортирует результаты валидации"""
        try:
            # Если результаты не переданы, пробуем собрать из таблицы
            data = results if isinstance(results, dict) else None
            if data is None:
                # Собираем из виджета (минимально)
                metrics = {}
                rows = (
                    self.tableWidgetValidationResults.rowCount()
                    if hasattr(self, "tableWidgetValidationResults")
                    else 0
                )
                for i in range(rows):
                    name_item = self.tableWidgetValidationResults.item(i, 0)
                    val_item = self.tableWidgetValidationResults.item(i, 1)
                    if name_item and val_item:
                        try:
                            metrics[name_item.text()] = float(val_item.text())
                        except Exception:
                            metrics[name_item.text()] = val_item.text()
                data = {"metrics": metrics, "timestamp": datetime.now().isoformat()}

            # Диалог выбора пути
            default_name = (
                f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Сохранить результаты валидации", default_name, "JSON (*.json)"
            )
            if not out_path:
                return

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QtWidgets.QMessageBox.information(
                self, "Успех", f"Результаты сохранены в:\n{out_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка экспорта результатов: {e}"
            )

    def _delete_experiment(self):
        """Удаляет выбранный эксперимент"""
        QtWidgets.QMessageBox.information(
            self, "Информация", "Функция удаления экспериментов будет реализована"
        )

    def _export_experiment(self):
        """Экспортирует данные эксперимента"""
        QtWidgets.QMessageBox.information(
            self, "Информация", "Функция экспорта эксперимента будет реализована"
        )

    def _generate_plots(self, silent: bool = False):
        """Генерирует и отображает графики метрик для выбранного эксперимента

        :param silent: Если True, не показывать информационные диалоги
        """
        try:
            # Проверяем, что есть менеджер тренировки и список экспериментов
            if not self.training_manager or not hasattr(self, "listWidgetExperiments"):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Компоненты тренировки не инициализированы или список экспериментов недоступен.",
                )
                return

            current_item = self.listWidgetExperiments.currentItem()
            if not current_item:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Выберите эксперимент в списке \"Эксперименты\" для построения графиков.",
                )
                return

            experiment_id = current_item.data(Qt.UserRole)
            if not experiment_id:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Не удалось определить ID эксперимента.",
                )
                return

            # Получаем метрики из базы данных
            metrics_data = (
                self.training_manager.metrics_tracker.database.get_experiment_metrics(
                    experiment_id
                )
            )
            if not metrics_data:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Информация",
                    "Для выбранного эксперимента нет сохранённых метрик.",
                )
                return

            # Определяем каталог для графиков (plots/ рядом с БД плагина)
            plugin_dir = os.path.dirname(__file__)
            plots_dir = os.path.join(plugin_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            output_path = os.path.join(plots_dir, f"{experiment_id}_metrics.png")

            # Создаём графики через MetricsVisualizer
            MetricsVisualizer.plot_training_curves(
                metrics_data=metrics_data, output_path=output_path
            )

            if not os.path.exists(output_path):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Не удалось создать файл с графиками метрик.",
                )
                return

            # Загружаем и отображаем картинку в labelMetricsPlot
            if hasattr(self, "labelMetricsPlot"):
                pixmap = QtGui.QPixmap(output_path)
                if pixmap.isNull():
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Ошибка",
                        "Не удалось загрузить изображение графиков.",
                    )
                    return

                self.labelMetricsPlot.setPixmap(
                    pixmap.scaled(
                        self.labelMetricsPlot.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
                self.labelMetricsPlot.setAlignment(Qt.AlignCenter)

                # Сохраняем путь к последнему сгенерированному графику для функции _save_plots
                self._last_metrics_plot_path = output_path

            if not silent:
                QtWidgets.QMessageBox.information(
                    self,
                    "Успех",
                    "Графики метрик успешно сгенерированы.",
                )

        except Exception as e:
            logger.error(f"Ошибка генерации графиков: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка генерации графиков: {e}"
            )

    def _save_plots(self):
        """Сохраняет текущие графики метрик в выбранный файл"""
        try:
            plot_path = getattr(self, "_last_metrics_plot_path", None)
            if not plot_path or not os.path.exists(plot_path):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Информация",
                    "Сначала сгенерируйте графики метрик перед сохранением.",
                )
                return

            default_name = os.path.basename(plot_path)
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Сохранить графики метрик",
                default_name,
                "PNG (*.png);;Все файлы (*.*)",
            )
            if not out_path:
                return

            import shutil

            shutil.copyfile(plot_path, out_path)
            QtWidgets.QMessageBox.information(
                self, "Успех", f"Графики метрик сохранены в:\n{out_path}"
            )

        except Exception as e:
            logger.error(f"Ошибка сохранения графиков: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка сохранения графиков: {e}"
            )

    def _display_validation_results(self, results):
        """Отображает результаты валидации"""
        try:
            # Обновляем таблицу результатов валидации
            self.tableWidgetValidationResults.setRowCount(0)

            if "metrics" in results:
                metrics = results["metrics"]
                row = 0
                for metric_name, value in metrics.items():
                    # Пропускаем словари (например, class_metrics, size_metrics из comprehensive validation)
                    if isinstance(value, dict):
                        continue
                    # Проверяем, что значение можно преобразовать в число
                    try:
                        numeric_value = float(value)
                        self.tableWidgetValidationResults.insertRow(row)
                        # Используем красивое имя для отображения
                        display_name = self._get_display_name_for_metric(metric_name)
                        self.tableWidgetValidationResults.setItem(
                            row, 0, QtWidgets.QTableWidgetItem(display_name)
                        )
                        self.tableWidgetValidationResults.setItem(
                            row, 1, QtWidgets.QTableWidgetItem(f"{numeric_value:.4f}")
                        )
                        self.tableWidgetValidationResults.setItem(
                            row,
                            2,
                            QtWidgets.QTableWidgetItem(
                                self._get_metric_description(metric_name)
                            ),
                        )
                        row += 1
                    except (ValueError, TypeError):
                        # Если значение не является числом, пропускаем его
                        continue

            # Добавляем информацию в лог
            self.textEditValidationLog.append(
                f"Валидация завершена: {results.get('timestamp', 'N/A')}"
            )

        except Exception as e:
            logger.error(
                f"Ошибка отображения результатов валидации: {e}", exc_info=True
            )

    def _normalize_metric_name_for_description(self, metric_name):
        """Нормализует имя метрики для поиска описания
        
        :param metric_name: Оригинальное имя метрики
        :return: Нормализованное имя метрики (для поиска в словаре описаний)
        """
        if not metric_name:
            return metric_name
        
        # Удаляем префиксы metrics/ и val/
        normalized = metric_name
        if normalized.startswith("metrics/"):
            normalized = normalized[8:]  # Удаляем 'metrics/'
        elif normalized.startswith("val/"):
            normalized = normalized[4:]  # Удаляем 'val/'
        
        # Удаляем суффикс (B) или (B-1) и т.д.
        if normalized.endswith("(B)"):
            normalized = normalized[:-3]
        elif "(" in normalized:
            # Удаляем скобки с содержимым в конце
            idx = normalized.rfind("(")
            if idx > 0:
                normalized = normalized[:idx]
        
        # Нормализуем различные варианты написания (приводим к нижнему регистру для поиска)
        normalized = normalized.strip().lower()
        
        # Маппинг различных вариантов имен на стандартные ключи для поиска
        name_mapping = {
            "map50": "map50",
            "map50-95": "map50-95",
            "map": "map50-95",
            "map_50": "map50",
            "map_50_95": "map50-95",
            "precision": "precision",
            "prec": "precision",
            "recall": "recall",
            "rec": "recall",
            "f1": "f1_score",
            "f1_score": "f1_score",
            "f1score": "f1_score",
        }
        
        return name_mapping.get(normalized, normalized)
    
    def _get_display_name_for_metric(self, metric_name):
        """Возвращает красивое имя метрики для отображения в таблице"""
        # Нормализуем для поиска
        normalized = self._normalize_metric_name_for_description(metric_name)
        
        # Маппинг на стандартные имена для отображения
        display_mapping = {
            "map50": "mAP50",
            "map50-95": "mAP50-95",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score",
            "f1": "F1-Score",
            "fitness": "Fitness",
        }
        
        # Если нашли в маппинге, используем красивое имя
        display_name = display_mapping.get(normalized)
        if display_name:
            return display_name
        
        # Иначе возвращаем оригинальное имя с заглавной буквы
        return metric_name.capitalize() if metric_name else metric_name
    
    def _get_metric_description(self, metric_name):
        """Возвращает описание метрики"""
        # Нормализуем имя метрики
        normalized = self._normalize_metric_name_for_description(metric_name)
        
        descriptions = {
            "map50": "Mean Average Precision при IoU=0.5",
            "mAP50": "Mean Average Precision при IoU=0.5",
            "map50-95": "Mean Average Precision при IoU=0.5-0.95",
            "mAP50-95": "Mean Average Precision при IoU=0.5-0.95",
            "map": "Mean Average Precision при IoU=0.5-0.95",
            "precision": "Точность детекции (доля правильных детекций среди всех)",
            "recall": "Полнота детекции (доля найденных объектов среди всех)",
            "f1_score": "F1-мера (гармоническое среднее точности и полноты)",
            "f1": "F1-мера (гармоническое среднее точности и полноты)",
            # Дополнительные метрики, которые могут прийти из results_dict
            "fitness": "Fitness - комплексная метрика качества модели (взвешенная комбинация mAP50, precision, recall)",
            "speed": "Скорость обработки (мс на изображение)",
            "box_loss": "Loss для предсказания bounding box",
            "cls_loss": "Loss для классификации",
            "dfl_loss": "Distribution Focal Loss",
            "loss": "Общий loss",
        }
        
        # Пробуем найти по нормализованному имени (с учетом регистра)
        result = descriptions.get(normalized)
        if result:
            return result
        
        # Пробуем найти по оригинальному имени
        result = descriptions.get(metric_name)
        if result:
            return result
        
        # Если не нашли, возвращаем более информативное сообщение
        return f"Метрика: {metric_name}"

    def _start_detection(self):
        """Запускает процесс детекции объектов"""
        try:
            # Получаем путь к модели
            model_path = ""
            if hasattr(self, "fileWidgetDetectionModel"):
                model_path = self.fileWidgetDetectionModel.filePath()
                if model_path:
                    self.path_history.add_model_path(model_path)

            if not model_path:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка", "Выберите путь к модели (*.pt)!"
                )
                return

            # Загружаем модель
            self.labelDetectionStatus.setText("Загрузка модели...")
            self.progressBarDetection.setValue(10)
            self.textEditDetectionLog.append("Загрузка модели...")

            load_result = self.detector.load_model(model_path)
            if "error" in load_result:
                QtWidgets.QMessageBox.critical(
                    self, "Ошибка загрузки модели", load_result["error"]
                )
                self.labelDetectionStatus.setText("Ошибка загрузки модели")
                self.progressBarDetection.setValue(0)
                return

            self.textEditDetectionLog.append(
                f"Модель загружена: {load_result.get('num_classes', 0)} классов"
            )

            # Получаем источник данных
            raster_layer = None
            image_path = None

            if hasattr(self, "mMapLayerComboBoxDetectionRaster"):
                raster_layer = self.mMapLayerComboBoxDetectionRaster.currentLayer()

            if hasattr(self, "fileWidgetDetectionImage"):
                image_path = self.fileWidgetDetectionImage.filePath()

            if not raster_layer and not image_path:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Выберите растровый слой или изображение для детекции!",
                )
                return

            # Получаем параметры детекции
            conf_threshold = 0.25
            iou_threshold = 0.45
            image_size = None
            device = "cpu"

            if hasattr(self, "doubleSpinBoxDetectionConf"):
                conf_threshold = self.doubleSpinBoxDetectionConf.value()
            if hasattr(self, "doubleSpinBoxDetectionIoU"):
                iou_threshold = self.doubleSpinBoxDetectionIoU.value()
            if hasattr(self, "spinBoxDetectionImageSize"):
                image_size = (
                    self.spinBoxDetectionImageSize.value()
                    if self.spinBoxDetectionImageSize.value() > 0
                    else None
                )
            if hasattr(self, "comboBoxDetectionDevice"):
                device_text = self.comboBoxDetectionDevice.currentText()
                device = "0" if "GPU" in device_text else "cpu"

            # Выполняем детекцию
            self.labelDetectionStatus.setText("Выполнение детекции...")
            self.progressBarDetection.setValue(30)

            def progress_callback(message):
                """Callback для обновления прогресса"""
                if hasattr(self, "labelDetectionStatus"):
                    self.labelDetectionStatus.setText(message)
                if hasattr(self, "textEditDetectionLog"):
                    self.textEditDetectionLog.append(message)

            if raster_layer:
                # Получаем выбранный экстент (если указан)
                detection_extent = None
                if hasattr(self, "mExtentGroupBoxDetection"):
                    extent_groupbox = self.mExtentGroupBoxDetection
                    try:
                        detection_extent = extent_groupbox.outputExtent()
                        extent_crs = extent_groupbox.outputCrs()
                        project = QgsProject.instance()
                        project_crs = project.crs()
                        
                        if detection_extent and not detection_extent.isEmpty():
                            # Проверяем и преобразуем extent в CRS проекта, если необходимо
                            # (outputCrs должен быть установлен в CRS проекта, но проверяем для надежности)
                            if extent_crs.isValid() and project_crs.isValid():
                                if extent_crs != project_crs:
                                    # Если CRS отличается, преобразуем в CRS проекта
                                    try:
                                        transform = QgsCoordinateTransform(
                                            extent_crs, project_crs, QgsProject.instance()
                                        )
                                        detection_extent = transform.transformBoundingBox(detection_extent)
                                        self.textEditDetectionLog.append(
                                            f"Экстент преобразован из CRS {extent_crs.authid()} "
                                            f"в CRS проекта {project_crs.authid()}"
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Ошибка преобразования CRS экстента: {e}. "
                                            f"Используется экстент без преобразования."
                                        )
                                else:
                                    # Extent уже в CRS проекта
                                    self.textEditDetectionLog.append(
                                        f"Экстент в CRS проекта {project_crs.authid()}"
                                    )
                            else:
                                # Если CRS не определен, логируем предупреждение
                                logger.warning(
                                    f"CRS extent или проекта не определен. "
                                    f"Extent CRS: {extent_crs.authid() if extent_crs.isValid() else 'не определен'}, "
                                    f"Project CRS: {project_crs.authid() if project_crs.isValid() else 'не определен'}"
                                )
                            
                            self.textEditDetectionLog.append(
                                f"Используется ограниченный экстент (CRS проекта): "
                                f"X: {detection_extent.xMinimum():.2f} - {detection_extent.xMaximum():.2f}, "
                                f"Y: {detection_extent.yMinimum():.2f} - {detection_extent.yMaximum():.2f}"
                            )
                    except Exception as e:
                        # Если не удалось получить extent из виджета, просто не используем ограничение
                        logger.debug(f"Не удалось получить extent из виджета: {e}")
                        detection_extent = None

                # Детекция на растровом слое
                self.textEditDetectionLog.append(
                    f"Детекция на растровом слое: {raster_layer.name()}"
                )
                results = self.detector.detect_on_raster(
                    raster_layer=raster_layer,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    image_size=image_size,
                    device=device,
                    progress_callback=progress_callback,
                    extent=detection_extent,
                )
                crs = raster_layer.crs()
            else:
                # Детекция на изображении
                self.textEditDetectionLog.append(f"Детекция на изображении: {image_path}")
                results = self.detector.detect_on_image(
                    image_path=image_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    image_size=image_size,
                    device=device,
                )
                crs = None

            self.progressBarDetection.setValue(70)

            # Проверяем результаты
            if "error" in results:
                QtWidgets.QMessageBox.critical(
                    self, "Ошибка детекции", results["error"]
                )
                self.labelDetectionStatus.setText("Ошибка детекции")
                self.progressBarDetection.setValue(0)
                return

            # Создаем векторный слой с результатами
            self.labelDetectionStatus.setText("Создание векторного слоя...")
            self.textEditDetectionLog.append("Создание векторного слоя...")

            detections = results.get("detections", [])
            if not detections:
                QtWidgets.QMessageBox.information(
                    self, "Информация", "Объекты не обнаружены"
                )
                self.labelDetectionStatus.setText("Объекты не обнаружены")
                self.progressBarDetection.setValue(0)
                return

            # Используем CRS из результатов или растра
            result_crs = results.get("crs") or crs

            layer_name = f"Детекция_{raster_layer.name() if raster_layer else Path(image_path).stem}"
            vector_layer = self.detector.create_vector_layer(
                detections=detections,
                layer_name=layer_name,
                crs=result_crs,
            )

            # Добавляем слой в проект
            QgsProject.instance().addMapLayer(vector_layer)

            self.progressBarDetection.setValue(100)
            self.labelDetectionStatus.setText(
                f"Детекция завершена: найдено {len(detections)} объектов"
            )
            self.textEditDetectionLog.append(
                f"Детекция завершена успешно. Найдено объектов: {len(detections)}"
            )

            QtWidgets.QMessageBox.information(
                self,
                "Успех",
                f"Детекция завершена!\nНайдено объектов: {len(detections)}\nСлой добавлен в проект: {layer_name}",
            )

        except Exception as e:
            logger.error(f"Ошибка детекции: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка выполнения детекции: {e}"
            )
            if hasattr(self, "labelDetectionStatus"):
                self.labelDetectionStatus.setText("Ошибка детекции")
            if hasattr(self, "progressBarDetection"):
                self.progressBarDetection.setValue(0)
