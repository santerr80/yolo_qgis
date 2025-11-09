# Подробная структура программы YOLO QGIS со всеми зависимостями

## 📋 Содержание
1. [Общая структура проекта](#общая-структура-проекта)
2. [Внешние зависимости](#внешние-зависимости)
3. [Внутренние зависимости модулей](#внутренние-зависимости-модулей)
4. [Дерево зависимостей](#дерево-зависимостей)
5. [Импорты и связи между модулями](#импорты-и-связи-между-модулями)
6. [Архитектурная схема](#архитектурная-схема)

---

## 📁 Общая структура проекта

```
yolo_qgis/
│
├── 📄 __init__.py                          # Точка входа плагина
├── 📄 yolo_qgis.py                         # Главный класс плагина (YoloQgis)
├── 📄 metadata.txt                         # Метаданные плагина для QGIS
├── 📄 requirements.txt                     # Внешние зависимости Python
├── 📄 icon.png                             # Иконка плагина
│
├── 🎨 Интерфейс пользователя
│   ├── yolo_qgis_dialog.py                # Главный диалог плагина
│   ├── yolo_qgis_dialog_base.ui           # UI форма (Qt Designer)
│   ├── yolo_training_dialog_base.ui       # UI форма обучения
│   ├── dataset_manager_dialog.py          # Диалог управления датасетами
│   ├── resources.py                        # Скомпилированные ресурсы Qt
│   └── resources.qrc                       # Ресурсы Qt (иконки, изображения)
│
├── 🔧 Утилиты и вспомогательные модули
│   ├── stderr_fix.py                       # Исправление stderr для QGIS
│   ├── fix_pillow.py                       # Исправление проблем Pillow
│   ├── compatibility_checker.py           # Проверка совместимости библиотек
│   ├── processing_utils.py                 # Утилиты обработки данных
│   └── dataset_utils.py                    # Утилиты для работы с датасетами
│
├── 📊 Работа с данными
│   ├── dataset_formatter.py                # Форматирование датасета YOLO
│   ├── dataset_formatter_yolo.py          # Нативное форматирование YOLO
│   ├── dataset_manager.py                  # Менеджер датасетов
│   └── dataset_manager_dialog.py           # Диалог управления датасетами
│
├── 🗺️ Геопространственные операции
│   ├── grid_creator.py                     # Создание сетки для разбиения
│   ├── intersection.py                    # Операции пересечения геометрий
│   └── map_exporter.py                     # Экспорт карт в изображения
│
├── 🎓 Система обучения YOLO
│   ├── yolo_trainer.py                     # Базовый класс обучения (YOLOTrainer)
│   ├── yolo_detection_trainer.py           # Обучение детекции (DetectionTrainer)
│   ├── yolo_segmentation_trainer.py       # Обучение сегментации (SegmentationTrainer)
│   ├── yolo_training_manager.py            # Главный менеджер обучения
│   └── yolo_training_example.py            # Примеры использования
│
├── ✅ Валидация и анализ
│   ├── yolo_validation.py                  # Расширенная валидация моделей
│   └── yolo_metrics_tracker.py             # Отслеживание метрик обучения
│
├── 🧪 Тесты
│   └── test/
│       ├── __init__.py
│       ├── qgis_interface.py              # Мок интерфейса QGIS
│       ├── test_init.py                    # Тесты инициализации
│       ├── test_qgis_environment.py        # Тесты окружения QGIS
│       ├── test_resources.py               # Тесты ресурсов
│       ├── test_translations.py            # Тесты переводов
│       ├── test_yolo_qgis_dialog.py        # Тесты диалога
│       └── utilities.py                    # Утилиты для тестов
│
├── 🌍 Интернационализация
│   └── i18n/
│       └── af.ts                           # Файлы переводов (африкаанс)
│
├── 📖 Документация
│   ├── README.txt                          # Основное описание
│   ├── README.html                         # HTML версия README
│   ├── README_training.md                  # Документация по обучению
│   ├── INTEGRATION_GUIDE.md                # Руководство по интеграции
│   ├── TRAINING_SYSTEM_SUMMARY.md          # Краткое описание системы
│   ├── TRAINING_UI_GUIDE.md                # Руководство по UI обучения
│   ├── NUMPY_V1_COMPATIBILITY.md           # Совместимость с NumPy
│   ├── PLUGIN_STRUCTURE.md                 # Структура плагина
│   └── PROGRAM_STRUCTURE.md               # Данный файл
│
├── 📚 Документация (Sphinx)
│   └── help/
│       ├── Makefile
│       ├── make.bat
│       └── source/
│           ├── conf.py
│           └── index.rst
│
├── 🔨 Сборочные файлы
│   ├── Makefile                            # Makefile для сборки
│   ├── pb_tool.cfg                         # Конфигурация pb_tool
│   ├── pylintrc                            # Конфигурация pylint
│   └── scripts/
│       ├── compile-strings.sh              # Компиляция строк
│       ├── run-env-linux.sh                # Запуск окружения Linux
│       └── update-strings.sh               # Обновление строк
│
└── 🔧 Дополнительные утилиты
    └── plugin_upload.py                    # Загрузка плагина
```

---

## 🔌 Внешние зависимости

### Основные зависимости для YOLO

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `ultralytics` | >=8.0.0 | Основная библиотека YOLO для обучения и инференса |
| `torch` | >=1.9.0 | PyTorch - фреймворк глубокого обучения |
| `torchvision` | >=0.10.0 | Библиотека для работы с изображениями в PyTorch |

### Обработка данных (совместимость с numpy v1)

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `numpy` | >=1.21.0,<2.0.0 | Математические операции и массивы |
| `pandas` | * | Обработка табличных данных |
| `opencv-python` | >=4.5.0,<5.0.0 | Обработка изображений и компьютерное зрение |
| `Pillow` | >=8.3.0,<9.5.0 | Работа с изображениями |

### Визуализация и графики (совместимость с numpy v1)

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `matplotlib` | >=3.4.0,<4.0.0 | Создание графиков и визуализаций |
| `seaborn` | >=0.11.0,<1.0.0 | Статистическая визуализация |
| `plotly` | >=5.0.0,<6.0.0 | Интерактивные графики |

### Работа с файлами и данными

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `PyYAML` | >=5.4.0 | Парсинг YAML конфигураций |
| `json5` | >=0.9.0 | Расширенный JSON парсер |
| `h5py` | >=3.1.0 | Работа с HDF5 файлами |

### База данных

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `sqlite3` | * | Встроенная в Python, для хранения метрик |

### Логирование и мониторинг

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `tensorboard` | >=2.7.0 | Визуализация метрик обучения |
| `wandb` | >=0.12.0 | Weights & Biases для отслеживания экспериментов (опционально) |

### Дополнительные утилиты

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `tqdm` | >=4.62.0 | Прогресс-бары для длительных операций |
| `psutil` | >=5.8.0 | Мониторинг системных ресурсов |
| `GPUtil` | >=1.4.0 | Мониторинг GPU |

### Дополнительные зависимости для расширенной функциональности

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `scikit-learn` | >=1.0.0,<2.0.0 | Дополнительные метрики и инструменты ML |
| `scipy` | >=1.7.0,<2.0.0 | Статистические вычисления |
| `albumentations` | >=1.1.0,<2.0.0 | Дополнительные аугментации изображений |

### Экспорт моделей

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `onnx` | >=1.10.0,<2.0.0 | Экспорт моделей в формат ONNX |
| `onnxruntime` | >=1.9.0,<2.0.0 | Запуск ONNX моделей |

### QGIS зависимости (встроенные в QGIS)

| Пакет | Назначение |
|-------|------------|
| `qgis.PyQt.QtCore` | Основные классы Qt (QObject, QThread, pyqtSignal) |
| `qgis.PyQt.QtGui` | GUI компоненты Qt (QIcon, QPixmap) |
| `qgis.PyQt.QtWidgets` | Виджеты Qt (QDialog, QWidget, QMessageBox) |
| `qgis.PyQt.uic` | Загрузка UI файлов |
| `qgis.core` | Основной API QGIS (QgsInterface, QgsMapLayer, QgsProject) |
| `qgis.gui` | GUI компоненты QGIS |

### Системные зависимости

| Компонент | Требование |
|-----------|------------|
| Python | >=3.8 |
| QGIS | >=3.0 |
| RAM | >=4 GB (рекомендуется 8+ GB) |
| GPU | Опционально, но рекомендуется для обучения |
| Дисковое пространство | >=2 GB для логов и моделей |

---

## 🔗 Внутренние зависимости модулей

### Иерархия модулей по уровням

#### Уровень 1: Точка входа
```
__init__.py
    └──► yolo_qgis.py
```

#### Уровень 2: Главный класс плагина
```
yolo_qgis.py
    ├──► stderr_fix.py
    ├──► resources.py
    └──► yolo_qgis_dialog.py
```

#### Уровень 3: Интерфейс пользователя
```
yolo_qgis_dialog.py
    ├──► stderr_fix.py
    ├──► grid_creator.py
    ├──► intersection.py
    ├──► map_exporter.py
    ├──► dataset_formatter.py
    ├──► dataset_formatter_yolo.py
    ├──► processing_utils.py
    ├──► dataset_manager_dialog.py
    └──► yolo_training_manager.py
```

#### Уровень 4: Работа с данными
```
dataset_manager_dialog.py
    └──► dataset_manager.py

dataset_formatter.py
    └──► dataset_utils.py

dataset_formatter_yolo.py
    └──► dataset_utils.py
```

#### Уровень 5: Система обучения
```
yolo_training_manager.py
    ├──► stderr_fix.py
    ├──► yolo_trainer.py
    ├──► yolo_detection_trainer.py
    ├──► yolo_segmentation_trainer.py
    ├──► yolo_validation.py
    └──► yolo_metrics_tracker.py
```

#### Уровень 6: Базовые классы обучения
```
yolo_detection_trainer.py
    └──► yolo_trainer.py

yolo_segmentation_trainer.py
    └──► yolo_trainer.py
```

---

## 🌳 Дерево зависимостей

### Полное дерево зависимостей

```
yolo_qgis (плагин)
│
├─── Точка входа
│   └─── __init__.py
│       ├─── stderr_fix.py
│       └─── yolo_qgis.py
│           ├─── stderr_fix.py
│           ├─── resources.py
│           └─── yolo_qgis_dialog.py
│
├─── Интерфейс пользователя
│   └─── yolo_qgis_dialog.py
│       ├─── stderr_fix.py
│       ├─── grid_creator.py
│       │   └─── qgis.core
│       ├─── intersection.py
│       │   └─── qgis.core
│       ├─── map_exporter.py
│       │   └─── qgis.core
│       ├─── dataset_formatter.py
│       │   ├─── dataset_utils.py
│       │   └─── PyYAML
│       ├─── dataset_formatter_yolo.py
│       │   ├─── dataset_utils.py
│       │   └─── PyYAML
│       ├─── processing_utils.py
│       ├─── dataset_manager_dialog.py
│       │   └─── dataset_manager.py
│       └─── yolo_training_manager.py
│
├─── Система обучения
│   └─── yolo_training_manager.py
│       ├─── stderr_fix.py
│       ├─── yolo_trainer.py
│       │   ├─── stderr_fix.py
│       │   ├─── ultralytics
│       │   ├─── torch
│       │   ├─── torchvision
│       │   ├─── numpy
│       │   ├─── opencv-python
│       │   ├─── Pillow
│       │   └─── PyYAML
│       ├─── yolo_detection_trainer.py
│       │   ├─── yolo_trainer.py
│       │   └─── (все зависимости yolo_trainer.py)
│       ├─── yolo_segmentation_trainer.py
│       │   ├─── yolo_trainer.py
│       │   └─── (все зависимости yolo_trainer.py)
│       ├─── yolo_validation.py
│       │   ├─── ultralytics
│       │   ├─── numpy
│       │   ├─── pandas
│       │   ├─── matplotlib
│       │   ├─── seaborn
│       │   └─── scikit-learn
│       └─── yolo_metrics_tracker.py
│           ├─── numpy
│           ├─── pandas
│           ├─── matplotlib
│           ├─── seaborn
│           └─── sqlite3
│
└─── Утилиты
    ├─── stderr_fix.py
    ├─── fix_pillow.py
    │   └─── Pillow
    ├─── compatibility_checker.py
    │   ├─── numpy
    │   └─── ultralytics
    └─── dataset_utils.py
        ├─── numpy
        ├─── opencv-python
        └─── Pillow
```

---

## 📦 Импорты и связи между модулями

### Детальная карта импортов

#### `__init__.py`
```python
# Внутренние импорты
from . import stderr_fix
from .yolo_qgis import YoloQgis
```

#### `yolo_qgis.py`
```python
# Внутренние импорты
from . import stderr_fix
from .resources import *
from .yolo_qgis_dialog import YoloQgisDialog

# QGIS импорты
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
```

#### `yolo_qgis_dialog.py`
```python
# Внутренние импорты
from .grid_creator import create_grid_layer
from .intersection import perform_intersection
from .map_exporter import export_views
from .dataset_formatter import format_yolo_dataset
from .dataset_formatter_yolo import save_yolo_native_dataset
from .processing_utils import ProgressReporter
from .dataset_manager_dialog import DatasetManagerDialog
from .dataset_manager import DatasetManager
from .yolo_training_manager import YOLOTrainingManager, TrainingConfigManager
from .yolo_detection_trainer import DetectionTrainer, DetectionDatasetAnalyzer
from .yolo_segmentation_trainer import SegmentationTrainer, SegmentationDatasetAnalyzer
from .yolo_validation import AdvancedValidator, ModelComparator
from .yolo_metrics_tracker import MetricsTracker, MetricsVisualizer

# QGIS импорты
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.core import QgsMapLayerProxyModel, QgsProject
```

#### `yolo_training_manager.py`
```python
# Внутренние импорты
from . import stderr_fix
from .yolo_trainer import YOLOTrainer, TrainingProgress
from .yolo_detection_trainer import DetectionTrainer, DetectionDatasetAnalyzer
from .yolo_segmentation_trainer import SegmentationTrainer, SegmentationDatasetAnalyzer
from .yolo_validation import AdvancedValidator, ModelComparator
from .yolo_metrics_tracker import MetricsTracker, MetricsVisualizer

# QGIS импорты
from qgis.PyQt.QtCore import QObject, pyqtSignal, QThread
from qgis.PyQt.QtWidgets import QMessageBox

# Стандартные библиотеки
import os, json, uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
```

#### `yolo_trainer.py`
```python
# Внутренние импорты
from . import stderr_fix

# QGIS импорты
from qgis.PyQt.QtCore import QObject, pyqtSignal, QThread, QTimer
from qgis.PyQt.QtWidgets import QMessageBox

# Внешние зависимости
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Стандартные библиотеки
import os, sys, io, json, yaml, shutil, subprocess, threading, time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import tempfile
```

#### `yolo_detection_trainer.py`
```python
# Внутренние импорты
from .yolo_trainer import YOLOTrainer, TrainingProgress, TrainingThread

# Внешние зависимости
try:
    from ultralytics import YOLO
    import torch
    import torchvision
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Стандартные библиотеки
import os, json, yaml
from typing import Dict, List, Optional, Union
from pathlib import Path
```

#### `yolo_segmentation_trainer.py`
```python
# Внутренние импорты
from .yolo_trainer import YOLOTrainer, TrainingProgress, TrainingThread

# Внешние зависимости
try:
    from ultralytics import YOLO
    import torch
    import torchvision
    import numpy as np
    import cv2
    from PIL import Image
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Стандартные библиотеки
import os, json, yaml
from typing import Dict, List, Optional, Union
from pathlib import Path
```

#### `yolo_validation.py`
```python
# Внешние зависимости
try:
    from ultralytics import YOLO
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Стандартные библиотеки
import os, json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from collections import defaultdict
```

#### `yolo_metrics_tracker.py`
```python
# Внешние зависимости
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sqlite3
    import json
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# Стандартные библиотеки
import os
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict
```

#### `dataset_formatter.py`
```python
# Внутренние импорты
from .dataset_utils import *

# Внешние зависимости
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# QGIS импорты
from qgis.core import QgsVectorLayer, QgsFeature, QgsGeometry
```

#### `dataset_formatter_yolo.py`
```python
# Внутренние импорты
from .dataset_utils import *

# Внешние зависимости
try:
    import yaml
    import numpy as np
    import cv2
    from PIL import Image
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# QGIS импорты
from qgis.core import QgsVectorLayer, QgsFeature, QgsGeometry
```

#### `grid_creator.py`
```python
# QGIS импорты
from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsField,
    QgsProject, QgsWkbTypes, QgsCoordinateReferenceSystem
)
from qgis.PyQt.QtCore import QVariant
```

#### `intersection.py`
```python
# QGIS импорты
from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsProject, QgsSpatialIndex
)
```

#### `map_exporter.py`
```python
# QGIS импорты
from qgis.core import (
    QgsMapLayer, QgsRasterLayer, QgsProject,
    QgsRectangle, QgsCoordinateReferenceSystem
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QImage, QPainter
```

---

## 🏗️ Архитектурная схема

### Полная архитектурная схема с зависимостями

```
┌─────────────────────────────────────────────────────────────────┐
│                         QGIS Application                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  qgis.PyQt.QtCore, qgis.PyQt.QtGui, qgis.PyQt.QtWidgets  │  │
│  │  qgis.core (QgsInterface, QgsMapLayer, QgsProject)       │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         __init__.py                             │
│                    (classFactory - точка входа)                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Зависимости:                                             │  │
│  │  • stderr_fix.py                                         │  │
│  │  • yolo_qgis.py                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        yolo_qgis.py                             │
│                         class YoloQgis                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Зависимости:                                             │  │
│  │  • stderr_fix.py                                         │  │
│  │  • resources.py                                          │  │
│  │  • yolo_qgis_dialog.py                                   │  │
│  │  • qgis.PyQt.*                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    yolo_qgis_dialog.py                          │
│                   class YoloQgisDialog                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Зависимости:                                             │  │
│  │  • grid_creator.py                                       │  │
│  │  • intersection.py                                       │  │
│  │  • map_exporter.py                                       │  │
│  │  • dataset_formatter.py                                  │  │
│  │  • dataset_formatter_yolo.py                             │  │
│  │  • processing_utils.py                                   │  │
│  │  • dataset_manager_dialog.py                             │  │
│  │  • yolo_training_manager.py                              │  │
│  └───────────────────────────────────────────────────────────┘  │
└────┬──────────────┬──────────────┬──────────────┬───────────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
│ Dataset  │  │   Grid   │  │   Map    │  │   Training   │
│ Manager  │  │ Creator  │  │ Exporter │  │   Manager    │
│          │  │          │  │          │  │              │
│ Зависимости:│ Зависимости:│ Зависимости:│ Зависимости: │
│ • dataset_│ • qgis.core│ • qgis.core│ • yolo_trainer │
│   manager │ • qgis.PyQt│ • qgis.PyQt│ • yolo_detect │
│ • dataset_│            │            │ • yolo_segment │
│   utils   │            │            │ • yolo_valid  │
│ • PyYAML  │            │            │ • yolo_metrics │
└──────────┘  └──────────┘  └──────────┘  └──────┬───────┘
                                                  │
                                                  ▼
                          ┌───────────────────────────────────────┐
                          │     yolo_training_manager.py          │
                          │     class YOLOTrainingManager         │
                          │  ┌─────────────────────────────────┐  │
                          │  │ Зависимости:                    │  │
                          │  │ • yolo_trainer.py               │  │
                          │  │ • yolo_detection_trainer.py     │  │
                          │  │ • yolo_segmentation_trainer.py │  │
                          │  │ • yolo_validation.py            │  │
                          │  │ • yolo_metrics_tracker.py       │  │
                          │  └─────────────────────────────────┘  │
                          └──────┬──────────┬──────────┬──────────┘
                                 │          │          │
                    ┌────────────┘          │          └────────────┐
                    ▼                       ▼                       ▼
        ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
        │ yolo_trainer.py  │    │ yolo_validation.py│    │yolo_metrics_     │
        │ class YOLOTrainer│    │ class Advanced    │    │tracker.py        │
        │                  │    │      Validator    │    │class Metrics     │
        │ Зависимости:     │    │                  │    │     Tracker      │
        │ • ultralytics    │    │ Зависимости:     │    │                  │
        │ • torch          │    │ • ultralytics    │    │ Зависимости:     │
        │ • torchvision    │    │ • numpy          │    │ • numpy          │
        │ • numpy          │    │ • pandas         │    │ • pandas         │
        │ • opencv-python  │    │ • matplotlib     │    │ • matplotlib     │
        │ • Pillow         │    │ • seaborn        │    │ • seaborn        │
        │ • PyYAML         │    │ • scikit-learn   │    │ • sqlite3        │
        └────────┬──────────┘    └──────────────────┘    └──────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│yolo_detection│  │yolo_segment  │
│_trainer.py   │  │ation_trainer│
│              │  │.py           │
│ Наследуется  │  │              │
│ от YOLOTrain │  │ Наследуется  │
│     er       │  │ от YOLOTrain │
│              │  │     er       │
│ + все        │  │              │
│ зависимости  │  │ + все        │
│ базового     │  │ зависимости  │
│ класса       │  │ базового     │
│              │  │ класса       │
└──────────────┘  └──────────────┘
```

---

## 📊 Матрица зависимостей модулей

### Таблица зависимостей между внутренними модулями

| Модуль | Зависит от модулей | Используется модулями |
|--------|-------------------|----------------------|
| `__init__.py` | `stderr_fix.py`, `yolo_qgis.py` | QGIS (автоматически) |
| `yolo_qgis.py` | `stderr_fix.py`, `resources.py`, `yolo_qgis_dialog.py` | `__init__.py` |
| `yolo_qgis_dialog.py` | `grid_creator.py`, `intersection.py`, `map_exporter.py`, `dataset_formatter.py`, `dataset_formatter_yolo.py`, `processing_utils.py`, `dataset_manager_dialog.py`, `yolo_training_manager.py` | `yolo_qgis.py` |
| `yolo_training_manager.py` | `yolo_trainer.py`, `yolo_detection_trainer.py`, `yolo_segmentation_trainer.py`, `yolo_validation.py`, `yolo_metrics_tracker.py` | `yolo_qgis_dialog.py` |
| `yolo_trainer.py` | `stderr_fix.py` | `yolo_training_manager.py`, `yolo_detection_trainer.py`, `yolo_segmentation_trainer.py` |
| `yolo_detection_trainer.py` | `yolo_trainer.py` | `yolo_training_manager.py` |
| `yolo_segmentation_trainer.py` | `yolo_trainer.py` | `yolo_training_manager.py` |
| `yolo_validation.py` | - | `yolo_training_manager.py` |
| `yolo_metrics_tracker.py` | - | `yolo_training_manager.py` |
| `dataset_manager.py` | - | `dataset_manager_dialog.py` |
| `dataset_manager_dialog.py` | `dataset_manager.py` | `yolo_qgis_dialog.py` |
| `dataset_formatter.py` | `dataset_utils.py` | `yolo_qgis_dialog.py` |
| `dataset_formatter_yolo.py` | `dataset_utils.py` | `yolo_qgis_dialog.py` |
| `grid_creator.py` | - | `yolo_qgis_dialog.py` |
| `intersection.py` | - | `yolo_qgis_dialog.py` |
| `map_exporter.py` | - | `yolo_qgis_dialog.py` |
| `processing_utils.py` | - | `yolo_qgis_dialog.py` |
| `dataset_utils.py` | - | `dataset_formatter.py`, `dataset_formatter_yolo.py` |
| `stderr_fix.py` | - | `__init__.py`, `yolo_qgis.py`, `yolo_trainer.py`, `yolo_training_manager.py` |
| `compatibility_checker.py` | - | (опционально) |
| `fix_pillow.py` | - | (опционально) |

---

## 🔄 Потоки данных и зависимостей

### 1. Поток создания датасета

```
Raster Layer + Vector Layer
         │
         ▼
   [grid_creator.py] ──► Создание сетки
         │ Зависимости: qgis.core
         ▼
  [intersection.py] ──► Нахождение пересечений
         │ Зависимости: qgis.core
         ▼
  [map_exporter.py] ──► Экспорт изображений
         │ Зависимости: qgis.core, qgis.PyQt.QtGui
         ▼
[dataset_formatter.py] ──► Форматирование в YOLO
         │ Зависимости: dataset_utils.py, PyYAML, qgis.core
         ▼
     YOLO Dataset
```

### 2. Поток обучения модели

```
Dataset
   │
   ▼
[dataset_manager.py] ──► Загрузка/управление
   │ Зависимости: -
   │
   ▼
[yolo_training_manager.py] ──► Управление процессом
   │ Зависимости: yolo_trainer.py, yolo_detection_trainer.py,
   │              yolo_segmentation_trainer.py, yolo_validation.py,
   │              yolo_metrics_tracker.py
   │
   ├──► [yolo_detection_trainer.py] ──► Обучение детекции
   │       │ Зависимости: yolo_trainer.py, ultralytics, torch,
   │       │              torchvision, numpy, opencv-python, Pillow
   │       │
   │       └──► [yolo_trainer.py] ──► Базовое обучение
   │               │ Зависимости: ultralytics, torch, torchvision,
   │               │              numpy, opencv-python, Pillow, PyYAML
   │
   └──► [yolo_segmentation_trainer.py] ──► Обучение сегментации
           │ Зависимости: yolo_trainer.py, ultralytics, torch,
           │              torchvision, numpy, opencv-python, Pillow
           │
           └──► [yolo_trainer.py] ──► Базовое обучение
                   │ (те же зависимости)
                   │
                   ▼
[yolo_metrics_tracker.py] ──► Отслеживание метрик
         │ Зависимости: numpy, pandas, matplotlib, seaborn, sqlite3
         │
         ▼
   Trained Model + Metrics
```

### 3. Поток валидации модели

```
Trained Model + Test Dataset
         │
         ▼
[yolo_validation.py] ──► Комплексная валидация
         │ Зависимости: ultralytics, numpy, pandas, matplotlib,
         │              seaborn, scikit-learn
         │
         ├──► Анализ порогов уверенности
         ├──► Анализ порогов IoU
         ├──► Анализ по классам
         └──► Анализ ошибок
         │
         ▼
[yolo_training_manager.py] ──► Сравнение моделей
         │ Зависимости: yolo_validation.py
         │
         ▼
   Validation Results
```

---

## 📝 Резюме зависимостей

### Критические зависимости (без них плагин не работает)

1. **QGIS API** - основа плагина
   - `qgis.PyQt.QtCore`, `qgis.PyQt.QtGui`, `qgis.PyQt.QtWidgets`
   - `qgis.core`

2. **YOLO/Ultralytics** - основная функциональность
   - `ultralytics >= 8.0.0`
   - `torch >= 1.9.0`
   - `torchvision >= 0.10.0`

3. **Обработка данных**
   - `numpy >= 1.21.0, < 2.0.0`
   - `opencv-python >= 4.5.0, < 5.0.0`
   - `Pillow >= 8.3.0, < 9.5.0`

### Важные зависимости (расширяют функциональность)

1. **Визуализация**
   - `matplotlib >= 3.4.0, < 4.0.0`
   - `seaborn >= 0.11.0, < 1.0.0`
   - `plotly >= 5.0.0, < 6.0.0`

2. **Работа с данными**
   - `pandas`
   - `PyYAML >= 5.4.0`
   - `h5py >= 3.1.0`

3. **Мониторинг и логирование**
   - `tensorboard >= 2.7.0`
   - `tqdm >= 4.62.0`
   - `psutil >= 5.8.0`

### Опциональные зависимости

1. **Расширенная функциональность**
   - `scikit-learn >= 1.0.0, < 2.0.0`
   - `scipy >= 1.7.0, < 2.0.0`
   - `albumentations >= 1.1.0, < 2.0.0`

2. **Экспорт моделей**
   - `onnx >= 1.10.0, < 2.0.0`
   - `onnxruntime >= 1.9.0, < 2.0.0`

3. **Облачные сервисы**
   - `wandb >= 0.12.0` (опционально)

---

## 🎯 Рекомендации по установке зависимостей

### Порядок установки

1. **Установить базовые зависимости**
   ```bash
   pip install numpy>=1.21.0,<2.0.0
   pip install opencv-python>=4.5.0,<5.0.0
   pip install Pillow>=8.3.0,<9.5.0
   ```

2. **Установить PyTorch (с учетом версии CUDA)**
   ```bash
   pip install torch>=1.9.0 torchvision>=0.10.0
   ```

3. **Установить Ultralytics**
   ```bash
   pip install ultralytics>=8.0.0
   ```

4. **Установить зависимости для визуализации**
   ```bash
   pip install matplotlib>=3.4.0,<4.0.0
   pip install seaborn>=0.11.0,<1.0.0
   pip install plotly>=5.0.0,<6.0.0
   ```

5. **Установить остальные зависимости**
   ```bash
   pip install -r requirements.txt
   ```

### Проверка установки

Используйте `compatibility_checker.py` для проверки всех зависимостей:
```python
from compatibility_checker import check_all_dependencies
check_all_dependencies()
```

---

## 📚 Дополнительная информация

### Версионирование зависимостей

Все зависимости ограничены версиями для обеспечения совместимости с:
- **NumPy v1.x** (не v2.0.0+) - для совместимости с QGIS
- **Python 3.8+** - минимальная версия Python
- **QGIS 3.0+** - минимальная версия QGIS

### Обработка отсутствующих зависимостей

Все модули используют паттерн graceful degradation:
```python
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    # Плагин продолжит работу, но без функциональности YOLO
```

### Совместимость

- ✅ **Windows 10+**
- ✅ **Linux** (Ubuntu 18.04+, Debian 10+)
- ✅ **macOS 10.14+**
- ✅ **QGIS 3.0+**
- ✅ **Python 3.8+**

---

**Дата создания документа**: 2025-01-XX  
**Версия плагина**: 0.1  
**Автор**: santerr80

