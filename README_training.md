# Система обучения и валидации YOLO моделей для QGIS

Этот модуль предоставляет комплексную систему для обучения и валидации YOLO моделей детекции и сегментации объектов в рамках QGIS плагина.

## Возможности

### 🎯 Обучение моделей
- **Детекция объектов**: Обучение YOLOv8 моделей для детекции объектов
- **Сегментация объектов**: Обучение YOLOv8-seg моделей для сегментации объектов
- **Поддержка различных размеров моделей**: от yolov8n до yolov8x
- **Гибкие параметры обучения**: эпохи, размер батча, скорость обучения, аугментации

### 📊 Валидация и анализ
- **Простая валидация**: Базовые метрики (mAP, precision, recall)
- **Комплексная валидация**: Анализ по порогам уверенности и IoU
- **Сравнение моделей**: Автоматическое сравнение нескольких моделей
- **Анализ датасетов**: Детальный анализ структуры и качества датасетов

### 📈 Отслеживание метрик
- **Логирование в реальном времени**: JSON, CSV, SQLite
- **Визуализация**: Автоматическое создание графиков обучения
- **Экспорт данных**: Экспорт результатов в различных форматах
- **История экспериментов**: Полная история всех экспериментов

### ⚙️ Управление конфигурациями
- **Предустановленные конфигурации**: Быстрый старт с готовыми настройками
- **Сохранение/загрузка**: Управление пользовательскими конфигурациями
- **Валидация параметров**: Проверка корректности настроек

## Структура модулей

```
yolo_training/
├── yolo_trainer.py              # Базовый класс для обучения
├── yolo_detection_trainer.py    # Специализированный тренажер детекции
├── yolo_segmentation_trainer.py # Специализированный тренажер сегментации
├── yolo_validation.py           # Система валидации и сравнения
├── yolo_metrics_tracker.py      # Отслеживание метрик и логирование
├── yolo_training_manager.py     # Главный менеджер системы
├── yolo_training_example.py     # Примеры использования
├── requirements_training.txt    # Зависимости
└── README_training.md          # Документация
```

## Установка зависимостей

```bash
pip install -r requirements_training.txt
```

## Быстрый старт

### 1. Обучение модели детекции

```python
from yolo_training_manager import YOLOTrainingManager

# Инициализация менеджера
manager = YOLOTrainingManager(
    log_dir='logs/detection',
    db_path='yolo_metrics.db'
)

# Запуск обучения
experiment_id = manager.start_detection_training(
    dataset_path="path/to/your/dataset",
    model_type='yolov8n',
    epochs=100,
    batch_size=16,
    image_size=640,
    learning_rate=0.01,
    device='cpu',  # или '0' для GPU
    pretrained=True,
    project_name='my_detection_model'
)

print(f"Обучение запущено. ID эксперимента: {experiment_id}")
```

### 2. Обучение модели сегментации

```python
# Запуск обучения сегментации
experiment_id = manager.start_segmentation_training(
    dataset_path="path/to/your/dataset",
    model_type='yolov8n-seg',
    epochs=100,
    batch_size=16,
    image_size=640,
    learning_rate=0.01,
    device='cpu',
    pretrained=True,
    project_name='my_segmentation_model',
    # Специфичные для сегментации параметры
    mask_ratio=4,
    overlap_mask=True,
    copy_paste=0.3
)
```

### 3. Валидация модели

```python
# Простая валидация
results = manager.validate_model(
    model_path="path/to/trained/model.pt",
    dataset_path="path/to/dataset",
    task='detect',
    comprehensive=False
)

# Комплексная валидация
comprehensive_results = manager.validate_model(
    model_path="path/to/trained/model.pt",
    dataset_path="path/to/dataset",
    task='detect',
    comprehensive=True
)
```

### 4. Сравнение моделей

```python
# Сравнение нескольких моделей
models = [
    {'name': 'Model_1', 'path': 'path/to/model1.pt'},
    {'name': 'Model_2', 'path': 'path/to/model2.pt'},
    {'name': 'Model_3', 'path': 'path/to/model3.pt'}
]

comparison_results = manager.compare_models(
    models=models,
    dataset_path="path/to/dataset",
    task='detect'
)
```

### 5. Анализ датасета

```python
# Анализ датасета
analysis = manager.analyze_dataset(
    dataset_path="path/to/dataset",
    task='detect'
)

print(f"Общее количество изображений: {analysis['total_images']}")
print(f"Общее количество аннотаций: {analysis['total_annotations']}")
```

## Конфигурации по умолчанию

Система включает предустановленные конфигурации:

### Детекция
- **detection_fast**: Быстрое обучение (yolov8n, 50 эпох)
- **detection_accurate**: Точное обучение (yolov8l, 200 эпох)

### Сегментация
- **segmentation_fast**: Быстрая сегментация (yolov8n-seg, 50 эпох)
- **segmentation_accurate**: Точная сегментация (yolov8l-seg, 200 эпох)

## Отслеживание метрик

### Получение сводки по эксперименту

```python
summary = manager.get_experiment_summary(experiment_id)
print(f"Всего эпох: {summary['total_epochs']}")
print(f"Лучшие метрики: {summary['best_metrics']}")
print(f"Финальные метрики: {summary['final_metrics']}")
```

### Создание графиков обучения

```python
success = manager.create_training_plots(
    experiment_id=experiment_id,
    output_dir='plots/experiment_1'
)
```

### Экспорт данных

```python
# Экспорт в JSON
manager.export_experiment_data(
    experiment_id=experiment_id,
    output_path='export/experiment_1.json',
    format='json'
)

# Экспорт в CSV
manager.export_experiment_data(
    experiment_id=experiment_id,
    output_path='export/experiment_1.csv',
    format='csv'
)
```

## Управление конфигурациями

```python
from yolo_training_manager import TrainingConfigManager

config_manager = TrainingConfigManager('my_configs')

# Сохранение конфигурации
config = {
    'task': 'detect',
    'model_type': 'yolov8s',
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.005
}
config_manager.save_config(config, 'my_custom_config')

# Загрузка конфигурации
loaded_config = config_manager.load_config('my_custom_config')

# Список доступных конфигураций
configs = config_manager.list_configs()
```

## Сигналы для UI

Система предоставляет сигналы для интеграции с пользовательским интерфейсом:

```python
# Подключение сигналов
manager.training_started.connect(on_training_started)
manager.training_progress.connect(on_training_progress)
manager.training_completed.connect(on_training_completed)
manager.validation_started.connect(on_validation_started)
manager.validation_completed.connect(on_validation_completed)

def on_training_started(experiment_id):
    print(f"Обучение начато: {experiment_id}")

def on_training_progress(epoch, metrics):
    print(f"Эпоха {epoch}: {metrics}")

def on_training_completed(experiment_id, success, message):
    print(f"Обучение завершено: {experiment_id}, успех: {success}, сообщение: {message}")
```

## Структура датасета

Датасет должен иметь следующую структуру:

```
dataset/
├── dataset.yaml          # Конфигурация датасета
├── images/
│   ├── train/           # Обучающие изображения
│   ├── val/             # Валидационные изображения
│   └── test/            # Тестовые изображения (опционально)
└── labels/
    ├── train/           # Аннотации для обучения
    ├── val/             # Аннотации для валидации
    └── test/            # Аннотации для тестирования (опционально)
```

### Формат dataset.yaml

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: class1
  1: class2
  2: class3
```

## Поддерживаемые форматы моделей

- **PyTorch (.pt)**: Нативный формат YOLO
- **ONNX (.onnx)**: Для развертывания
- **TorchScript (.torchscript)**: Для оптимизации
- **TensorFlow Lite (.tflite)**: Для мобильных устройств
- **OpenVINO (.xml)**: Для Intel процессоров

## Мониторинг и логирование

### Логи в файлах
- `logs/yolo_training.log`: Основной лог
- `logs/metrics.json`: Метрики в JSON формате
- `logs/metrics.csv`: Метрики в CSV формате

### База данных
- `yolo_metrics.db`: SQLite база с полной историей экспериментов

### Графики
- Кривые обучения для каждой метрики
- Сводные графики
- Графики сравнения моделей
- Матрицы путаницы

## Примеры использования

См. файл `yolo_training_example.py` для подробных примеров использования всех возможностей системы.

## Требования к системе

- Python 3.8+
- QGIS 3.x
- PyTorch 1.9+
- Ultralytics 8.0+
- 4+ GB RAM (рекомендуется 8+ GB)
- GPU (опционально, но рекомендуется для больших моделей)

## Поддержка

Для вопросов и предложений создавайте issues в репозитории проекта.

## Лицензия

Совместимо с лицензией QGIS плагина.

