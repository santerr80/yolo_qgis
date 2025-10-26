# Руководство по интеграции системы обучения YOLO

## Обзор реализованной системы

Разработана комплексная система для обучения и валидации YOLO моделей детекции и сегментации объектов, интегрированная с существующим QGIS плагином.

## Созданные модули

### 1. Основные модули обучения
- **`yolo_trainer.py`** - Базовый класс для обучения YOLO моделей
- **`yolo_detection_trainer.py`** - Специализированный тренажер для детекции объектов
- **`yolo_segmentation_trainer.py`** - Специализированный тренажер для сегментации объектов

### 2. Модули валидации и анализа
- **`yolo_validation.py`** - Расширенная система валидации и сравнения моделей
- **`yolo_metrics_tracker.py`** - Отслеживание метрик и логирование

### 3. Управляющие модули
- **`yolo_training_manager.py`** - Главный менеджер системы
- **`yolo_training_example.py`** - Примеры использования

### 4. Документация
- **`requirements_training.txt`** - Зависимости
- **`README_training.md`** - Подробная документация
- **`INTEGRATION_GUIDE.md`** - Данное руководство

## Ключевые возможности

### 🎯 Обучение моделей
- **Детекция**: YOLOv8n/s/m/l/x для детекции объектов
- **Сегментация**: YOLOv8n/s/m/l/x-seg для сегментации объектов
- **Гибкие параметры**: эпохи, размер батча, скорость обучения, аугментации
- **Поддержка GPU/CPU**: автоматическое определение устройства

### 📊 Валидация и анализ
- **Простая валидация**: базовые метрики (mAP, precision, recall)
- **Комплексная валидация**: анализ по порогам уверенности и IoU
- **Сравнение моделей**: автоматическое сравнение нескольких моделей
- **Анализ датасетов**: детальный анализ структуры и качества

### 📈 Отслеживание метрик
- **Логирование в реальном времени**: JSON, CSV, SQLite
- **Визуализация**: автоматическое создание графиков обучения
- **Экспорт данных**: различные форматы экспорта
- **История экспериментов**: полная история всех экспериментов

## Интеграция с существующим плагином

### 1. Добавление в основной диалог

В файл `yolo_qgis_dialog.py` можно добавить новые вкладки для обучения:

```python
# Добавить импорт
from .yolo_training_manager import YOLOTrainingManager

class YoloQgisDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(YoloQgisDialog, self).__init__(parent)
        self.setupUi(self)
        self._setup_connections()
        
        # Инициализация менеджера обучения
        self.training_manager = YOLOTrainingManager()
        self._setup_training_connections()
    
    def _setup_training_connections(self):
        """Настройка соединений для обучения"""
        self.training_manager.training_started.connect(self.on_training_started)
        self.training_manager.training_progress.connect(self.on_training_progress)
        self.training_manager.training_completed.connect(self.on_training_completed)
        self.training_manager.validation_completed.connect(self.on_validation_completed)
    
    def start_detection_training(self):
        """Запуск обучения детекции"""
        dataset_path = self.mQgsFileWidget.filePath()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Выберите датасет")
            return
        
        experiment_id = self.training_manager.start_detection_training(
            dataset_path=dataset_path,
            model_type='yolov8n',
            epochs=100,
            batch_size=16,
            device='cpu'
        )
        
        if experiment_id:
            QMessageBox.information(self, "Успех", f"Обучение запущено: {experiment_id}")
    
    def start_segmentation_training(self):
        """Запуск обучения сегментации"""
        dataset_path = self.mQgsFileWidget.filePath()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Выберите датасет")
            return
        
        experiment_id = self.training_manager.start_segmentation_training(
            dataset_path=dataset_path,
            model_type='yolov8n-seg',
            epochs=100,
            batch_size=16,
            device='cpu'
        )
        
        if experiment_id:
            QMessageBox.information(self, "Успех", f"Обучение запущено: {experiment_id}")
    
    def on_training_started(self, experiment_id):
        """Обработчик начала обучения"""
        self.statusBar().showMessage(f"Обучение начато: {experiment_id}")
    
    def on_training_progress(self, epoch, metrics):
        """Обработчик прогресса обучения"""
        self.progressBar.setValue(int((epoch / 100) * 100))  # Предполагаем 100 эпох
        self.statusBar().showMessage(f"Эпоха {epoch}: mAP50={metrics.get('mAP50', 0):.3f}")
    
    def on_training_completed(self, experiment_id, success, message):
        """Обработчик завершения обучения"""
        if success:
            QMessageBox.information(self, "Успех", f"Обучение завершено: {message}")
        else:
            QMessageBox.critical(self, "Ошибка", f"Ошибка обучения: {message}")
    
    def on_validation_completed(self, experiment_id, results):
        """Обработчик завершения валидации"""
        if 'error' not in results:
            QMessageBox.information(self, "Валидация", 
                f"mAP50: {results.get('performance_metrics', {}).get('overall_mAP50', 0):.3f}")
```

### 2. Обновление UI файла

В файл `yolo_qgis_dialog_base.ui` можно добавить новые вкладки:

```xml
<!-- Добавить вкладку "Обучение" -->
<widget class="QTabWidget" name="tabWidget">
    <property name="currentIndex">
        <number>0</number>
    </property>
    <widget class="QWidget" name="tabDataset">
        <!-- Существующая вкладка датасета -->
    </widget>
    <widget class="QWidget" name="tabTraining">
        <attribute name="title">
            <string>Обучение</string>
        </attribute>
        <layout class="QVBoxLayout" name="verticalLayout_training">
            <item>
                <widget class="QGroupBox" name="groupBoxTraining">
                    <property name="title">
                        <string>Параметры обучения</string>
                    </property>
                    <layout class="QGridLayout" name="gridLayout_training">
                        <!-- Поля для параметров обучения -->
                    </layout>
                </widget>
            </item>
            <item>
                <widget class="QGroupBox" name="groupBoxValidation">
                    <property name="title">
                        <string>Валидация</string>
                    </property>
                    <layout class="QGridLayout" name="gridLayout_validation">
                        <!-- Поля для валидации -->
                    </layout>
                </widget>
            </item>
            <item>
                <layout class="QHBoxLayout" name="horizontalLayout_training_buttons">
                    <item>
                        <widget class="QPushButton" name="pushButtonStartDetection">
                            <property name="text">
                                <string>Обучение детекции</string>
                            </property>
                        </widget>
                    </item>
                    <item>
                        <widget class="QPushButton" name="pushButtonStartSegmentation">
                            <property name="text">
                                <string>Обучение сегментации</string>
                            </property>
                        </widget>
                    </item>
                    <item>
                        <widget class="QPushButton" name="pushButtonValidate">
                            <property name="text">
                                <string>Валидация модели</string>
                            </property>
                        </widget>
                    </item>
                </layout>
            </item>
        </layout>
    </widget>
</widget>
```

### 3. Добавление зависимостей

Обновить файл `requirements.txt` или создать отдельный файл для обучения:

```bash
# Установка зависимостей для обучения
pip install -r requirements_training.txt
```

## Примеры использования

### Базовое обучение детекции

```python
from yolo_training_manager import YOLOTrainingManager

# Инициализация
manager = YOLOTrainingManager()

# Запуск обучения
experiment_id = manager.start_detection_training(
    dataset_path="path/to/dataset",
    model_type='yolov8n',
    epochs=100,
    batch_size=16,
    device='cpu'
)

# Валидация после обучения
results = manager.validate_model(
    model_path="runs/detect/train/weights/best.pt",
    dataset_path="path/to/dataset",
    task='detect'
)
```

### Обучение сегментации

```python
# Запуск обучения сегментации
experiment_id = manager.start_segmentation_training(
    dataset_path="path/to/dataset",
    model_type='yolov8n-seg',
    epochs=100,
    batch_size=16,
    device='cpu',
    copy_paste=0.3  # Важно для сегментации
)
```

### Сравнение моделей

```python
# Сравнение нескольких моделей
models = [
    {'name': 'Model_1', 'path': 'path/to/model1.pt'},
    {'name': 'Model_2', 'path': 'path/to/model2.pt'}
]

comparison = manager.compare_models(
    models=models,
    dataset_path="path/to/dataset",
    task='detect'
)
```

## Структура файлов проекта

```
yolo_qgis/
├── yolo_qgis.py                    # Основной файл плагина
├── yolo_qgis_dialog.py            # Диалог плагина
├── yolo_qgis_dialog_base.ui       # UI файл
├── dataset_formatter_yolo.py      # Существующий форматтер
├── dataset_manager.py             # Существующий менеджер датасетов
├── 
├── # Новые модули обучения
├── yolo_trainer.py                # Базовый тренажер
├── yolo_detection_trainer.py      # Тренажер детекции
├── yolo_segmentation_trainer.py   # Тренажер сегментации
├── yolo_validation.py             # Система валидации
├── yolo_metrics_tracker.py        # Отслеживание метрик
├── yolo_training_manager.py       # Главный менеджер
├── yolo_training_example.py       # Примеры использования
├── 
├── # Документация
├── requirements_training.txt      # Зависимости
├── README_training.md             # Документация
└── INTEGRATION_GUIDE.md           # Данное руководство
```

## Настройка окружения

### 1. Установка зависимостей

```bash
# Основные зависимости (совместимые с numpy v1)
pip install ultralytics torch torchvision

# Дополнительные зависимости для анализа (совместимые с numpy v1)
pip install 'numpy>=1.21.0,<2.0.0' 'pandas>=1.3.0,<2.0.0'
pip install 'matplotlib>=3.4.0,<4.0.0' 'seaborn>=0.11.0,<1.0.0'

# Зависимости для работы с файлами
pip install PyYAML 'opencv-python>=4.5.0,<5.0.0' Pillow
```

**ВАЖНО**: Все зависимости ограничены версиями, совместимыми с numpy v1.x для стабильной работы в QGIS.

### 2. Проверка установки

```python
# Проверка доступности библиотек
try:
    from ultralytics import YOLO
    print("✓ ultralytics установлен")
except ImportError:
    print("✗ ultralytics не установлен")

try:
    import torch
    print(f"✓ PyTorch установлен: {torch.__version__}")
except ImportError:
    print("✗ PyTorch не установлен")

# Проверка совместимости numpy
try:
    import numpy as np
    version_parts = np.__version__.split('.')
    major_version = int(version_parts[0])
    if major_version >= 2:
        print(f"⚠ numpy {np.__version__} может быть несовместима с QGIS (требуется v1.x)")
    else:
        print(f"✓ numpy {np.__version__} совместима с QGIS")
except ImportError:
    print("✗ numpy не установлен")
```

### 3. Проверка совместимости

Для полной проверки совместимости используйте утилиту:

```bash
python compatibility_checker.py
```

## Мониторинг и отладка

### 1. Логи обучения

Система автоматически создает логи в директории `logs/`:
- `yolo_training.log` - основной лог
- `metrics.json` - метрики в JSON
- `metrics.csv` - метрики в CSV
- `yolo_metrics.db` - SQLite база данных

### 2. Графики и визуализация

Автоматически создаются графики в директории `plots/`:
- Кривые обучения для каждой метрики
- Сводные графики
- Графики сравнения моделей

### 3. Отладка проблем

```python
# Проверка доступности GPU
import torch
print(f"CUDA доступен: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Количество GPU: {torch.cuda.device_count()}")
    print(f"Текущий GPU: {torch.cuda.current_device()}")

# Проверка структуры датасета
from yolo_training_manager import YOLOTrainingManager
manager = YOLOTrainingManager()
analysis = manager.analyze_dataset("path/to/dataset", "detect")
print(f"Анализ датасета: {analysis}")
```

## Рекомендации по использованию

### 1. Для начинающих
- Начните с моделей `yolov8n` (самые быстрые)
- Используйте предустановленные конфигурации
- Начните с небольшого количества эпох (50-100)

### 2. Для продвинутых пользователей
- Экспериментируйте с различными размерами моделей
- Настройте параметры аугментации
- Используйте комплексную валидацию
- Сравнивайте несколько моделей

### 3. Для продакшена
- Используйте GPU для обучения
- Сохраняйте все эксперименты
- Экспортируйте модели в ONNX для развертывания
- Настройте автоматическое логирование

## Поддержка и развитие

### 1. Добавление новых функций
- Новые типы аугментаций
- Дополнительные метрики
- Поддержка других архитектур YOLO
- Интеграция с внешними сервисами

### 2. Оптимизация производительности
- Параллельное обучение
- Оптимизация памяти
- Ускорение валидации
- Кэширование результатов

### 3. Улучшение пользовательского интерфейса
- Визуализация в реальном времени
- Интерактивные графики
- Управление экспериментами
- Автоматические отчеты

## Заключение

Разработанная система предоставляет полный цикл обучения и валидации YOLO моделей для детекции и сегментации объектов. Система интегрируется с существующим QGIS плагином и предоставляет:

- ✅ Простой интерфейс для обучения
- ✅ Комплексную систему валидации
- ✅ Детальное отслеживание метрик
- ✅ Автоматическую визуализацию
- ✅ Гибкие конфигурации
- ✅ Поддержку различных задач

Система готова к использованию и может быть легко расширена для решения специфических задач пользователей.

