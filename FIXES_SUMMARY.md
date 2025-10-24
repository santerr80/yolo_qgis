# Исправления для функциональности обновления датасетов

## Проблема
При запуске диалога обновления датасета возникала ошибка:
```
'QComboBox' object has no attribute 'setFilters'
```

## Причина
В файле `dataset_update_dialog.py` использовались обычные `QComboBox` вместо специализированных QGIS виджетов:
- `QComboBox` вместо `QgsMapLayerComboBox` для выбора слоев
- `QComboBox` вместо `QgsFieldComboBox` для выбора полей

## Исправления

### 1. Обновлены импорты
```python
# Добавлен импорт QGIS виджетов
from qgis.gui import QgsMapLayerComboBox, QgsFieldComboBox
```

### 2. Заменены виджеты
```python
# Было:
self.objects_layer_combo = QtWidgets.QComboBox()
self.class_field_combo = QtWidgets.QComboBox()

# Стало:
self.objects_layer_combo = QgsMapLayerComboBox()
self.class_field_combo = QgsFieldComboBox()
```

### 3. Обновлены методы работы с виджетами
```python
# Обновление списка слоев
def update_layers_list(self):
    # QgsMapLayerComboBox автоматически обновляется
    pass

# Обновление полей классов
def update_class_fields(self):
    layer = self.objects_layer_combo.currentLayer()
    self.class_field_combo.setLayer(layer)
```

### 4. Исправлены соединения сигналов
```python
# Было:
self.objects_layer_combo.currentTextChanged.connect(self.update_class_fields)

# Стало:
self.objects_layer_combo.layerChanged.connect(self.update_class_fields)
```

### 5. Обновлены методы получения данных
```python
# Было:
layer = self.objects_layer_combo.currentData()
class_field = self.class_field_combo.currentText()

# Стало:
layer = self.objects_layer_combo.currentLayer()
class_field = self.class_field_combo.currentField()
```

## Результат
Теперь диалог обновления датасета должен работать корректно:
- Автоматическое заполнение списка слоев
- Автоматическое обновление полей при смене слоя
- Корректная работа с QGIS API

## Файлы, которые были изменены
- `dataset_update_dialog.py` - основные исправления
- `yolo_qgis_dialog_base.ui` - обновлен интерфейс (пользователем)
- `yolo_qgis_dialog.py` - интеграция новых диалогов

## Тестирование
Для проверки исправлений:
1. Запустите QGIS
2. Откройте плагин YOLO QGIS
3. На вкладке "Dataset" нажмите "Update dataset"
4. Диалог должен открыться без ошибок
5. Проверьте работу выбора слоев и полей
