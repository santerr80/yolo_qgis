# -*- coding: utf-8 -*-
"""
/***************************************************************************
 YoloQgisDialog
 ... (you header here) ...
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
from .processing_utils import ProgressReporter # <--- ДОБАВЛЕН ИМПОРТ


FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'yolo_qgis_dialog_base.ui'))


class YoloQgisDialog(QtWidgets.QDialog, FORM_CLASS):
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
        # ... (код этой функции остается без изменений)
        sender = self.sender()
        if not sender: return
        is_pixel_source = sender in (self.lineEdit_WidthPixel, self.lineEdit_HeigthPixel, self.comboBox_Dpi)
        is_meter_source = sender in (self.lineEdit_WidthMeter, self.lineEdit_HeigthMeter)
        try:
            dpi = int(self.comboBox_Dpi.currentText())
            if is_pixel_source:
                w_px, h_px = int(self.lineEdit_WidthPixel.text()), int(self.lineEdit_HeigthPixel.text())
                w_m, h_m = (w_px / dpi) * 0.0254, (h_px / dpi) * 0.0254
                self.lineEdit_WidthMeter.blockSignals(True); self.lineEdit_HeigthMeter.blockSignals(True)
                self.lineEdit_WidthMeter.setText(f"{w_m:.4f}"); self.lineEdit_HeigthMeter.setText(f"{h_m:.4f}")
                self.lineEdit_WidthMeter.blockSignals(False); self.lineEdit_HeigthMeter.blockSignals(False)
            elif is_meter_source:
                w_m, h_m = float(self.lineEdit_WidthMeter.text()), float(self.lineEdit_HeigthMeter.text())
                w_px, h_px = round((w_m / 0.0254) * dpi), round((h_m / 0.0254) * dpi)
                self.lineEdit_WidthPixel.blockSignals(True); self.lineEdit_HeigthPixel.blockSignals(True)
                self.lineEdit_WidthPixel.setText(str(w_px)); self.lineEdit_HeigthPixel.setText(str(h_px))
                self.lineEdit_WidthPixel.blockSignals(False); self.lineEdit_HeigthPixel.blockSignals(False)
        except (ValueError, ZeroDivisionError):
            pass

    def update_split_values(self):
        """Перераспределяет значения Train/Val/Test, чтобы сумма была 100."""
        # ... (код этой функции остается без изменений)
        sender = self.sender()
        if not sender: return
        self.spinBox_Train.blockSignals(True); self.spinBox_Val.blockSignals(True); self.spinBox_Test.blockSignals(True)
        try:
            train, val, test = self.spinBox_Train.value(), self.spinBox_Val.value(), self.spinBox_Test.value()
            remainder = 100 - sender.value()
            if sender == self.spinBox_Train:
                other_sum = val + test
                if other_sum == 0: new_val, new_test = remainder, 0
                else: new_val, new_test = round(remainder * (val / other_sum)), remainder - round(remainder * (val / other_sum))
                self.spinBox_Val.setValue(new_val); self.spinBox_Test.setValue(new_test)
            elif sender == self.spinBox_Val:
                other_sum = train + test
                if other_sum == 0: new_train, new_test = remainder, 0
                else: new_train, new_test = round(remainder * (train / other_sum)), remainder - round(remainder * (train / other_sum))
                self.spinBox_Train.setValue(new_train); self.spinBox_Test.setValue(new_test)
            elif sender == self.spinBox_Test:
                other_sum = train + val
                if other_sum == 0: new_train, new_val = remainder, 0
                else: new_train, new_val = round(remainder * (train / other_sum)), remainder - round(remainder * (train / other_sum))
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
            if not output_dir or not objects_layer:
                raise ValueError("Необходимо указать выходную папку и слой объектов.")
            
            img_width_m = float(self.lineEdit_WidthMeter.text())
            img_height_m = float(self.lineEdit_HeigthMeter.text())
            v_overlay_m = float(self.lineEdit_VerticalOverlay.text() or 0.0)
            h_overlay_m = float(self.lineEdit_HorizontalOverlay.text() or 0.0)
            
            img_width_px = int(self.lineEdit_WidthPixel.text())
            img_height_px = int(self.lineEdit_HeigthPixel.text())
            img_dpi = int(self.comboBox_Dpi.currentText())
            img_format = self.comboBoxFileFormat.currentText()
            
            if img_width_m <= 0 or img_height_m <= 0 or img_width_px <= 0 or img_height_px <= 0:
                raise ValueError("Все размеры должны быть больше нуля.")
        except (ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка ввода", f"Проверьте корректность входных данных:\n{e}")
            return
        
        # --- ШАГ 1: Создание сетки (0% -> 20%) ---
        print("\n1. Создание сетки...")
        self.progressBar.setValue(5)
        grid_layer, error_msg = create_grid_layer(
            source_layer=objects_layer, h_spacing=img_width_m, v_spacing=img_height_m,
            h_overlay=h_overlay_m, v_overlay=v_overlay_m)
        
        if error_msg:
            QtWidgets.QMessageBox.critical(self, "Ошибка создания сетки", error_msg)
            self.progressBar.setValue(0); return
        grid_layer.setName(f"Сетка для '{objects_layer.name()}'")
        QgsProject.instance().addMapLayer(grid_layer)
        self.progressBar.setValue(20)
        print("Сетка успешно создана.")

        # --- ШАГ 2: Выполнение пересечения (20% -> 30%) ---
        print("\n2. Выполнение пересечения объектов с сеткой...")
        self.progressBar.setValue(25)
        intersected_layer, error_msg = perform_intersection(
            input_layer=objects_layer, overlay_layer=grid_layer)

        if error_msg:
            QtWidgets.QMessageBox.critical(self, "Ошибка пересечения", error_msg)
            self.progressBar.setValue(0); return
        intersected_layer.setName(f"Пересечение для '{objects_layer.name()}'")
        QgsProject.instance().addMapLayer(intersected_layer)
        self.progressBar.setValue(30)
        print("Слой пересечения успешно создан.")

        # --- ШАГ 3: Экспорт изображений (30% -> 100%) ---
        print("\n3. Экспорт видов карты...")
        images_output_dir = os.path.join(output_dir, 'images')
        
        progress_reporter = ProgressReporter(self.progressBar, start_percentage=30, end_percentage=100)
        
        success, error_msg = export_views(
            grid_layer=grid_layer,
            output_dir=images_output_dir,
            image_format=img_format,
            width_px=img_width_px,
            height_px=img_height_px,
            dpi=img_dpi,
            progress_reporter=progress_reporter
        )

        if not success:
            QtWidgets.QMessageBox.critical(self, "Ошибка экспорта", error_msg)
            self.progressBar.setValue(0); return
        
        self.progressBar.setValue(100)
        print("\n--- Процесс успешно завершен ---")
        QtWidgets.QMessageBox.information(self, "Готово", 
            f"Все операции успешно завершены!\n"
            f"Изображения сохранены в:\n{images_output_dir}")
        self.accept()