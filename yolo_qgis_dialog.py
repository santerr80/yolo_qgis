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


FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'yolo_qgis_dialog_base.ui'))


class YoloQgisDialog(QtWidgets.QDialog, FORM_CLASS):
    # ... (весь код __init__, _setup_connections, _update_size_values, update_split_values остается без изменений) ...
    def __init__(self, parent=None):
        """Конструктор."""
        super(YoloQgisDialog, self).__init__(parent)
        self.setupUi(self)
        self._setup_connections()

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

    def run_dataset_creation(self):
        """Основная функция, запускающая весь процесс."""
        print("--- Запуск процесса ---")
        self.progressBar.setValue(0)

        # --- Сбор и проверка данных ---
        try:
            output_dir = self.mQgsFileWidget.filePath()
            objects_layer = self.mMapLayerComboBoxObjects.currentLayer()
            classes_field = self.mFieldComboBoxObjects.currentField()
            if not all([output_dir, objects_layer, classes_field]):
                raise ValueError("Необходимо заполнить все поля в группе 'Setup'.")
            
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
        
        # --- ИЗМЕНЕНИЕ: ВЫБОР ФОРМАТА ДЛЯ СОХРАНЕНИЯ ---
        if save_in_yolo_format:
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
        QtWidgets.QMessageBox.information(self, "Готово", f"Датасет успешно создан!\nСохранено в: {output_dir}")
        self.accept()