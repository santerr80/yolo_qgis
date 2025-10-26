# -*- coding: utf-8 -*-
"""
/***************************************************************************
 YoloQgisDialog
 This file was automatically generated with qtdesigner.py
 ***************************************************************************/
"""

import os

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets

from qgis.core import QgsMapLayerProxyModel, QgsProject

# --- ИМПОРТ ФУНКЦИЙ ИЗ ВНЕШНИХ МОДУЛЕЙ ---
from .grid_creator import create_grid_layer
from .intersection import perform_intersection
from .map_exporter import export_views
from .dataset_formatter import format_yolo_dataset 
from .dataset_formatter_yolo import save_yolo_native_dataset # <--- ДОБАВЛЕН НОВЫЙ ИМПОРТ
from .processing_utils import ProgressReporter
from .dataset_manager_dialog import DatasetManagerDialog
from .dataset_manager import DatasetManager

# --- ИМПОРТ МОДУЛЕЙ ТРЕНИРОВКИ ---
from .yolo_training_manager import YOLOTrainingManager, TrainingConfigManager
from .yolo_detection_trainer import DetectionTrainer, DetectionDatasetAnalyzer
from .yolo_segmentation_trainer import SegmentationTrainer, SegmentationDatasetAnalyzer
from .yolo_validation import AdvancedValidator, ModelComparator
from .yolo_metrics_tracker import MetricsTracker, MetricsVisualizer



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
        
        self._setup_connections()
        self._initialize_training_components()

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
            print(f"Предупреждение: не удалось инициализировать QgsFileWidget: {e}")
        
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
            print(f"Предупреждение: не удалось обновить подсказку QgsFileWidget: {e}")

    def run_dataset_creation(self):
        """Основная функция, запускающая весь процесс."""
        print("--- Запуск процесса ---")
        self.progressBar.setValue(0)

        # --- Сбор и проверка данных ---
        try:
            # Безопасное получение пути из QgsFileWidget
            try:
                output_dir = self.mQgsFileWidget.filePath()
            except Exception as e:
                print(f"Ошибка получения пути из QgsFileWidget: {e}")
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
                    print("В указанной директории не найден существующий датасет. Создается новый датасет.")
                else:
                    print(f"Найден существующий датасет в директории: {output_dir}")
            
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
        print("\n1. Создание сетки...")
        self.progressBar.setValue(5)
        grid_layer, error_msg = create_grid_layer(
            source_layer=objects_layer, h_spacing=img_width_m, v_spacing=img_height_m,
            h_overlay=h_overlay_m, v_overlay=v_overlay_m)
        if error_msg:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg); self.progressBar.setValue(0); return
        grid_layer.setName(f"Сетка для '{objects_layer.name()}'")
        QgsProject.instance().addMapLayer(grid_layer)
        self.progressBar.setValue(15)
        print("Сетка создана.")

        # --- ШАГ 2: Выполнение пересечения (15% -> 25%) ---
        print("\n2. Пересечение объектов с сеткой...")
        self.progressBar.setValue(20)
        intersected_layer, error_msg = perform_intersection(
            input_layer=objects_layer, overlay_layer=grid_layer)
        if error_msg:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg); self.progressBar.setValue(0); return
        intersected_layer.setName(f"Пересечение для '{objects_layer.name()}'")
        QgsProject.instance().addMapLayer(intersected_layer)
        self.progressBar.setValue(25)
        print("Слой пересечения создан.")

        # --- ШАГ 3: Экспорт изображений (25% -> 75%) ---
        print("\n3. Экспорт изображений тайлов...")
        
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
        print("Экспорт изображений завершен.")

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
            print("\n4. Обновление существующего датасета...")
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
            print("\n4. Формирование датасета в нативном формате YOLO...")
            success, error_msg = save_yolo_native_dataset(
                intersected_layer=intersected_layer, grid_layer=grid_layer,
                class_field=classes_field, output_dir=output_dir, image_format=img_format,
                splits=splits, metadata=metadata, delete_void=delete_void,
                progress_reporter=progress_reporter_format)
        else:
            print("\n4. Формирование файла аннотаций data.ndjson...")
            success, error_msg = format_yolo_dataset(
                intersected_layer=intersected_layer, grid_layer=grid_layer,
                class_field=classes_field, output_dir=output_dir, image_format=img_format,
                image_width=img_width_px, image_height=img_height_px,
                splits=splits, metadata=metadata, delete_void=delete_void,
                progress_reporter=progress_reporter_format)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            
        if not success:
            QtWidgets.QMessageBox.critical(self, "Ошибка", error_msg); self.progressBar.setValue(0); return
        print("Формирование датасета завершено.")
        
        self.progressBar.setValue(100)
        print("\n--- Процесс успешно завершен ---")
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
                print(f"Предупреждение: {backup_message}")
            else:
                print(f"Резервная копия: {backup_message}")
            
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
                print(f"Добавлены новые классы: {', '.join(new_classes)}")
            
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
            
            # Обновляем список экспериментов
            self._refresh_experiments_list()
            
        except Exception as e:
            print(f"Ошибка инициализации компонентов тренировки: {e}")
            QtWidgets.QMessageBox.warning(self, "Предупреждение", 
                                        f"Некоторые функции тренировки могут быть недоступны: {e}")
    
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
            print(f"Ошибка настройки соединений тренировки: {e}")
    
    def _on_task_type_changed(self, task_type):
        """Обработчик изменения типа задачи"""
        try:
            if not hasattr(self, 'comboBoxModelType'):
                print("Предупреждение: comboBoxModelType не найден")
                return
                
            if "Детекция" in task_type:
                # Обновляем список моделей для детекции
                self.comboBoxModelType.clear()
                models = [
                    "YOLOv8n (быстрая)",
                    "YOLOv8s (сбалансированная)", 
                    "YOLOv8m (средняя)",
                    "YOLOv8l (большая)",
                    "YOLOv8x (максимальная)"
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
                    "YOLOv8x-seg (максимальная)"
                ]
                self.comboBoxModelType.addItems(models)
        except Exception as e:
            print(f"Ошибка обновления типа задачи: {e}")
    
    def _start_training(self):
        """Запускает процесс тренировки"""
        try:
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
                    self.labelTrainingStatus.setText("Тренировка остановлена")
                    self.textEditTrainingLog.append("Тренировка остановлена пользователем")
                else:
                    QtWidgets.QMessageBox.warning(self, "Предупреждение", "Не удалось остановить тренировку")
        except Exception as e:
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
            print(f"Ошибка отображения анализа: {e}")
    
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
            "YOLOv8n-seg (быстрая)": "yolov8n-seg",
            "YOLOv8s-seg (сбалансированная)": "yolov8s-seg",
            "YOLOv8m-seg (средняя)": "yolov8m-seg",
            "YOLOv8l-seg (большая)": "yolov8l-seg",
            "YOLOv8x-seg (максимальная)": "yolov8x-seg"
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
            print(f"Ошибка получения конфигурации: {e}")
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
            print(f"Ошибка применения конфигурации: {e}")
    
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
            print(f"Ошибка обновления прогресса: {e}")
    
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
    
    def _on_validation_completed(self, experiment_id, results):
        """Обработчик завершения валидации"""
        try:
            if 'error' in results:
                QtWidgets.QMessageBox.critical(self, "Ошибка валидации", results['error'])
                return
            
            # Отображаем результаты валидации
            self._display_validation_results(results)
            
        except Exception as e:
            print(f"Ошибка обработки результатов валидации: {e}")
    
    def _update_metrics_table(self, epoch, metrics):
        """Обновляет таблицу метрик"""
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
            self.tableWidgetMetrics.setItem(row_count, 5, QtWidgets.QTableWidgetItem(f"{metrics.get('loss', 0):.4f}"))
            
            # Прокручиваем к последней строке
            self.tableWidgetMetrics.scrollToBottom()
            
        except Exception as e:
            print(f"Ошибка обновления таблицы метрик: {e}")
    
    def _refresh_experiments_list(self):
        """Обновляет список экспериментов"""
        try:
            if not self.training_manager:
                return
            
            experiments = self.training_manager.get_all_experiments()
            self.listWidgetExperiments.clear()
            
            for exp in experiments:
                item_text = f"{exp.get('name', 'Unknown')} - {exp.get('status', 'Unknown')}"
                item = QtWidgets.QListWidgetItem(item_text)
                item.setData(QtWidgets.Qt.UserRole, exp.get('id', ''))
                self.listWidgetExperiments.addItem(item)
                
        except Exception as e:
            print(f"Ошибка обновления списка экспериментов: {e}")
    
    def _on_experiment_selected(self):
        """Обработчик выбора эксперимента"""
        try:
            current_item = self.listWidgetExperiments.currentItem()
            if not current_item:
                return
            
            experiment_id = current_item.data(QtWidgets.Qt.UserRole)
            if not experiment_id:
                return
            
            # Получаем информацию об эксперименте
            summary = self.training_manager.get_experiment_summary(experiment_id)
            
            # Отображаем информацию
            info_text = self._format_experiment_info(summary)
            self.textEditExperimentInfo.setPlainText(info_text)
            
        except Exception as e:
            print(f"Ошибка выбора эксперимента: {e}")
    
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
        QtWidgets.QMessageBox.information(self, "Информация", "Функция валидации будет реализована")
    
    def _compare_models(self):
        """Сравнивает модели"""
        QtWidgets.QMessageBox.information(self, "Информация", "Функция сравнения моделей будет реализована")
    
    def _export_validation_results(self):
        """Экспортирует результаты валидации"""
        QtWidgets.QMessageBox.information(self, "Информация", "Функция экспорта результатов будет реализована")
    
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
            print(f"Ошибка отображения результатов валидации: {e}")
    
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