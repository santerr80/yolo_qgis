# -*- coding: utf-8 -*-
"""
Диалог для управления датасетами - просмотр, резервное копирование, очистка
"""

import os
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QHeaderView

from .dataset_utils import DatasetUtils


class DatasetScanThread(QThread):
    """Поток для сканирования датасетов"""
    
    datasets_found = pyqtSignal(list)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
    
    def run(self):
        """Выполняет сканирование датасетов"""
        try:
            datasets = DatasetUtils.list_available_datasets(self.directory)
            self.datasets_found.emit(datasets)
        except Exception as e:
            self.datasets_found.emit([])


class DatasetManagerDialog(QtWidgets.QDialog):
    """Диалог для управления датасетами"""
    
    def __init__(self, parent=None):
        super(DatasetManagerDialog, self).__init__(parent)
        self.setupUi()
        self.setup_connections()
        self.scan_thread = None
        
    def setupUi(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle("Управление датасетами")
        self.setModal(True)
        self.resize(800, 600)
        
        # Основной layout
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Группа поиска датасетов
        search_group = QtWidgets.QGroupBox("Поиск датасетов")
        search_layout = QtWidgets.QHBoxLayout(search_group)
        
        search_layout.addWidget(QtWidgets.QLabel("Директория поиска:"))
        self.search_path_edit = QtWidgets.QLineEdit()
        self.search_path_edit.setPlaceholderText("Выберите директорию для поиска датасетов")
        search_layout.addWidget(self.search_path_edit)
        
        self.browse_search_button = QtWidgets.QPushButton("Обзор...")
        search_layout.addWidget(self.browse_search_button)
        
        self.scan_button = QtWidgets.QPushButton("Сканировать")
        search_layout.addWidget(self.scan_button)
        
        main_layout.addWidget(search_group)
        
        # Таблица датасетов
        self.datasets_table = QtWidgets.QTableWidget()
        self.datasets_table.setColumnCount(8)
        self.datasets_table.setHorizontalHeaderLabels([
            "Название", "Тип", "Изображения", "Классы", "Размер (MB)", 
            "Создан", "Обновлен", "Путь"
        ])
        
        # Настройка таблицы
        header = self.datasets_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        
        self.datasets_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.datasets_table.setAlternatingRowColors(True)
        
        main_layout.addWidget(self.datasets_table)
        
        # Группа действий
        actions_group = QtWidgets.QGroupBox("Действия")
        actions_layout = QtWidgets.QHBoxLayout(actions_group)
        
        self.info_button = QtWidgets.QPushButton("Информация")
        self.backup_button = QtWidgets.QPushButton("Создать резервную копию")
        self.cleanup_button = QtWidgets.QPushButton("Очистить")
        self.export_info_button = QtWidgets.QPushButton("Экспорт информации")
        self.open_button = QtWidgets.QPushButton("Открыть")
        
        actions_layout.addWidget(self.info_button)
        actions_layout.addWidget(self.backup_button)
        actions_layout.addWidget(self.cleanup_button)
        actions_layout.addWidget(self.export_info_button)
        actions_layout.addWidget(self.open_button)
        
        main_layout.addWidget(actions_group)
        
        # Прогресс бар
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Кнопки диалога
        button_layout = QtWidgets.QHBoxLayout()
        self.close_button = QtWidgets.QPushButton("Закрыть")
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        main_layout.addLayout(button_layout)
        
    def setup_connections(self):
        """Настройка соединений сигналов и слотов"""
        self.browse_search_button.clicked.connect(self.browse_search_directory)
        self.scan_button.clicked.connect(self.scan_datasets)
        self.info_button.clicked.connect(self.show_dataset_info)
        self.backup_button.clicked.connect(self.create_backup)
        self.cleanup_button.clicked.connect(self.cleanup_dataset)
        self.export_info_button.clicked.connect(self.export_dataset_info)
        self.open_button.clicked.connect(self.open_dataset)
        self.close_button.clicked.connect(self.accept)
        
        # Двойной клик по таблице
        self.datasets_table.itemDoubleClicked.connect(self.show_dataset_info)
        
    def browse_search_directory(self):
        """Открывает диалог выбора директории для поиска"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Выберите директорию для поиска датасетов",
            "",
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.search_path_edit.setText(directory)
    
    def scan_datasets(self):
        """Запускает сканирование датасетов"""
        search_path = self.search_path_edit.text().strip()
        if not search_path:
            QMessageBox.warning(self, "Ошибка", "Укажите директорию для поиска")
            return
        
        if not os.path.exists(search_path):
            QMessageBox.warning(self, "Ошибка", "Указанная директория не существует")
            return
        
        # Запускаем сканирование в отдельном потоке
        self.scan_thread = DatasetScanThread(search_path)
        self.scan_thread.datasets_found.connect(self.on_datasets_found)
        self.scan_thread.progress_updated.connect(self.progress_bar.setValue)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Неопределенный прогресс
        self.scan_button.setEnabled(False)
        
        self.scan_thread.start()
    
    def on_datasets_found(self, datasets):
        """Обрабатывает найденные датасеты"""
        self.progress_bar.setVisible(False)
        self.scan_button.setEnabled(True)
        
        self.datasets_table.setRowCount(len(datasets))
        
        for row, dataset in enumerate(datasets):
            # Название
            name = os.path.basename(dataset['path'])
            self.datasets_table.setItem(row, 0, QTableWidgetItem(name))
            
            # Тип
            dataset_type = dataset.get('type', 'Неизвестно')
            self.datasets_table.setItem(row, 1, QTableWidgetItem(dataset_type))
            
            # Количество изображений
            images_count = dataset.get('images_count', 0)
            self.datasets_table.setItem(row, 2, QTableWidgetItem(str(images_count)))
            
            # Количество классов
            classes_count = len(dataset.get('class_names', {}))
            self.datasets_table.setItem(row, 3, QTableWidgetItem(str(classes_count)))
            
            # Размер
            size_mb = dataset.get('size_mb', 0)
            self.datasets_table.setItem(row, 4, QTableWidgetItem(f"{size_mb:.1f}"))
            
            # Дата создания
            created_at = dataset.get('created_at', '')
            if created_at:
                try:
                    # Парсим ISO дату и форматируем
                    from datetime import datetime
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    created_at = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            self.datasets_table.setItem(row, 5, QTableWidgetItem(created_at))
            
            # Дата обновления
            updated_at = dataset.get('updated_at', '')
            if updated_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    updated_at = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            self.datasets_table.setItem(row, 6, QTableWidgetItem(updated_at))
            
            # Путь
            self.datasets_table.setItem(row, 7, QTableWidgetItem(dataset['path']))
        
        if datasets:
            QMessageBox.information(self, "Результат", f"Найдено датасетов: {len(datasets)}")
        else:
            QMessageBox.information(self, "Результат", "Датасеты не найдены")
    
    def get_selected_dataset(self):
        """Возвращает выбранный датасет"""
        current_row = self.datasets_table.currentRow()
        print(f"Выбранная строка: {current_row}")
        if current_row >= 0:
            path_item = self.datasets_table.item(current_row, 7)
            if path_item:
                path_text = path_item.text()
                print(f"Путь из таблицы: {path_text}")
                return path_text
            else:
                print("Элемент пути не найден в таблице")
        else:
            print("Не выбрана строка в таблице")
        return None
    
    def show_dataset_info(self):
        """Показывает подробную информацию о датасете"""
        dataset_path = self.get_selected_dataset()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Выберите датасет в таблице")
            return
        
        try:
            info = DatasetUtils.get_dataset_info(dataset_path)
            
            # Создаем диалог с информацией
            info_dialog = QtWidgets.QDialog(self)
            info_dialog.setWindowTitle(f"Информация о датасете: {os.path.basename(dataset_path)}")
            info_dialog.setModal(True)
            info_dialog.resize(500, 400)
            
            layout = QtWidgets.QVBoxLayout(info_dialog)
            
            # Текстовое поле с информацией
            text_edit = QtWidgets.QTextEdit()
            text_edit.setReadOnly(True)
            
            info_text = f"Путь: {info['path']}\n"
            info_text += f"Тип: {info.get('type', 'Неизвестно')}\n"
            info_text += f"Количество изображений: {info.get('images_count', 0)}\n"
            info_text += f"Размер: {info.get('size_mb', 0):.1f} MB\n"
            
            if info.get('splits'):
                info_text += f"Разбивка:\n"
                for split, count in info['splits'].items():
                    info_text += f"  {split}: {count} изображений\n"
            
            if info.get('class_names'):
                info_text += f"Классы:\n"
                for class_id, class_name in info['class_names'].items():
                    info_text += f"  {class_id}: {class_name}\n"
            
            if info.get('created_at'):
                info_text += f"Создан: {info['created_at']}\n"
            
            if info.get('updated_at'):
                info_text += f"Обновлен: {info['updated_at']}\n"
            
            if info.get('metadata'):
                info_text += f"Метаданные:\n"
                for key, value in info['metadata'].items():
                    info_text += f"  {key}: {value}\n"
            
            text_edit.setText(info_text)
            layout.addWidget(text_edit)
            
            # Кнопка закрытия
            close_button = QtWidgets.QPushButton("Закрыть")
            close_button.clicked.connect(info_dialog.accept)
            layout.addWidget(close_button)
            
            info_dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка получения информации: {e}")
    
    def create_backup(self):
        """Создает резервную копию выбранного датасета"""
        dataset_path = self.get_selected_dataset()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Выберите датасет в таблице")
            return
        
        try:
            backup_dir = QFileDialog.getExistingDirectory(
                self,
                "Выберите директорию для резервной копии",
                os.path.dirname(dataset_path),
                QFileDialog.ShowDirsOnly
            )
            
            if backup_dir:
                success, message = DatasetUtils.create_dataset_backup(dataset_path, backup_dir)
                if success:
                    QMessageBox.information(self, "Успех", message)
                else:
                    QMessageBox.warning(self, "Ошибка", message)
                    
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка создания резервной копии: {e}")
    
    def cleanup_dataset(self):
        """Очищает выбранный датасет"""
        dataset_path = self.get_selected_dataset()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Выберите датасет в таблице")
            return
        
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            f"Очистить датасет от временных файлов?\n{dataset_path}",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                success, message = DatasetUtils.cleanup_dataset(dataset_path)
                if success:
                    QMessageBox.information(self, "Успех", message)
                else:
                    QMessageBox.warning(self, "Ошибка", message)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка очистки: {e}")
    
    def export_dataset_info(self):
        """Экспортирует информацию о датасете"""
        dataset_path = self.get_selected_dataset()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Выберите датасет в таблице")
            return
        
        try:
            output_file, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить информацию о датасете",
                f"{os.path.basename(dataset_path)}_info.json",
                "JSON файлы (*.json)"
            )
            
            if output_file:
                success, message = DatasetUtils.export_dataset_info(dataset_path, output_file)
                if success:
                    QMessageBox.information(self, "Успех", message)
                else:
                    QMessageBox.warning(self, "Ошибка", message)
                    
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта: {e}")
    
    def open_dataset(self):
        """Открывает выбранный датасет в проводнике"""
        dataset_path = self.get_selected_dataset()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Выберите датасет в таблице")
            return
        
        try:
            import subprocess
            import platform
            import os
            
            print(f"Попытка открыть директорию: {dataset_path}")
            
            # Проверяем, что путь существует
            if not os.path.exists(dataset_path):
                QMessageBox.warning(self, "Ошибка", f"Директория не существует: {dataset_path}")
                return
            
            # Нормализуем путь для Windows
            normalized_path = os.path.normpath(dataset_path)
            print(f"Нормализованный путь: {normalized_path}")
            
            if platform.system() == "Windows":
                # Сначала пробуем os.startfile (более надежный способ)
                try:
                    os.startfile(normalized_path)
                    print("Директория открыта через os.startfile")
                except Exception as e_startfile:
                    print(f"os.startfile не сработал: {e_startfile}")
                    # Если не сработало, пробуем через subprocess
                    try:
                        command = f'explorer "{normalized_path}"'
                        print(f"Выполняем команду: {command}")
                        result = subprocess.run(command, shell=True, capture_output=True, text=True)
                        if result.returncode != 0:
                            print(f"Ошибка выполнения команды: {result.stderr}")
                            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть директорию. Код ошибки: {result.returncode}")
                    except Exception as e_subprocess:
                        print(f"subprocess также не сработал: {e_subprocess}")
                        QMessageBox.warning(self, "Ошибка", f"Не удалось открыть директорию: {e_subprocess}")
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", normalized_path])
            else:  # Linux
                subprocess.run(["xdg-open", normalized_path])
                
        except Exception as e:
            print(f"Исключение при открытии директории: {e}")
            # Предлагаем скопировать путь в буфер обмена
            reply = QMessageBox.question(
                self, 
                "Ошибка открытия", 
                f"Не удалось открыть директорию: {e}\n\nСкопировать путь в буфер обмена?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    from qgis.PyQt.QtWidgets import QApplication
                    clipboard = QApplication.clipboard()
                    clipboard.setText(dataset_path)
                    QMessageBox.information(self, "Успех", "Путь скопирован в буфер обмена")
                except Exception as e_clipboard:
                    QMessageBox.warning(self, "Ошибка", f"Не удалось скопировать в буфер обмена: {e_clipboard}")
