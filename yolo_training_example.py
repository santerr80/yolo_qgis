# -*- coding: utf-8 -*-
"""
Пример использования системы обучения и валидации YOLO моделей
"""

import os
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(os.path.dirname(__file__))

from yolo_training_manager import YOLOTrainingManager, TrainingConfigManager


def example_detection_training():
    """Пример обучения модели детекции"""
    print("=== Пример обучения модели детекции ===")
    
    # Инициализируем менеджер обучения
    manager = YOLOTrainingManager(
        log_dir='logs/detection_example',
        db_path='yolo_metrics.db'
    )
    
    # Путь к датасету (должен содержать dataset.yaml)
    dataset_path = "path/to/your/detection/dataset"
    
    # Проверяем наличие датасета
    if not os.path.exists(dataset_path):
        print(f"Датасет не найден: {dataset_path}")
        print("Создайте датасет с помощью yolo_qgis_dialog.py")
        return
    
    # Запускаем обучение
    experiment_id = manager.start_detection_training(
        dataset_path=dataset_path,
        model_type='yolov8n',  # Можно использовать yolov8s, yolov8m, yolov8l, yolov8x
        epochs=50,
        batch_size=16,
        image_size=640,
        learning_rate=0.01,
        device='cpu',  # Или '0' для GPU
        pretrained=True,
        project_name='detection_example',
        # Параметры аугментации
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5
    )
    
    if experiment_id:
        print(f"Обучение запущено. ID эксперимента: {experiment_id}")
        
        # Ждем завершения обучения (в реальном приложении это будет в отдельном потоке)
        print("Ожидание завершения обучения...")
        
        # После завершения обучения можно выполнить валидацию
        # model_path = f"runs/detect/detection_example/weights/best.pt"
        # validation_results = manager.validate_model(model_path, dataset_path, 'detect')
        # print(f"Результаты валидации: {validation_results}")
    else:
        print("Ошибка запуска обучения")


def example_segmentation_training():
    """Пример обучения модели сегментации"""
    print("\n=== Пример обучения модели сегментации ===")
    
    # Инициализируем менеджер обучения
    manager = YOLOTrainingManager(
        log_dir='logs/segmentation_example',
        db_path='yolo_metrics.db'
    )
    
    # Путь к датасету
    dataset_path = "path/to/your/segmentation/dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Датасет не найден: {dataset_path}")
        return
    
    # Запускаем обучение
    experiment_id = manager.start_segmentation_training(
        dataset_path=dataset_path,
        model_type='yolov8n-seg',  # Можно использовать yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
        epochs=50,
        batch_size=16,
        image_size=640,
        learning_rate=0.01,
        device='cpu',
        pretrained=True,
        project_name='segmentation_example',
        # Специфичные для сегментации параметры
        mask_ratio=4,
        overlap_mask=True,
        copy_paste=0.3,  # Важно для сегментации
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5
    )
    
    if experiment_id:
        print(f"Обучение запущено. ID эксперимента: {experiment_id}")
    else:
        print("Ошибка запуска обучения")


def example_resume_training():
    """Пример возобновления прерванного обучения"""
    print("\n=== Пример возобновления прерванного обучения ===")
    
    # Инициализируем менеджер обучения
    manager = YOLOTrainingManager(
        log_dir='logs/resume_example',
        db_path='yolo_metrics.db'
    )
    
    # Путь к датасету
    dataset_path = "path/to/your/dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Датасет не найден: {dataset_path}")
        return
    
    # Способ 1: Автоматический поиск last.pt в стандартном месте
    # Система автоматически найдет last.pt в save_dir/project_name/weights/last.pt
    print("Способ 1: Автоматический поиск last.pt")
    experiment_id = manager.start_detection_training(
        dataset_path=dataset_path,
        model_type='yolov8n',
        epochs=100,
        batch_size=16,
        image_size=640,
        learning_rate=0.01,
        device='cpu',
        pretrained=True,
        save_dir='runs',  # Директория, где сохраняются результаты
        project_name='my_training',  # Имя проекта (должно совпадать с предыдущим обучением)
        resume_training=True,  # Включаем режим возобновления
    )
    
    if experiment_id:
        print(f"Обучение возобновлено. ID эксперимента: {experiment_id}")
    else:
        print("Ошибка возобновления обучения (возможно, last.pt не найден)")
    
    # Способ 2: Указание пути к last.pt напрямую через base_weights_path
    print("\nСпособ 2: Указание пути к last.pt напрямую")
    last_pt_path = "path/to/last.pt"  # Полный путь к файлу last.pt
    
    if os.path.exists(last_pt_path):
        experiment_id = manager.start_detection_training(
            dataset_path=dataset_path,
            model_type='yolov8n',
            epochs=100,
            batch_size=16,
            image_size=640,
            learning_rate=0.01,
            device='cpu',
            pretrained=True,
            save_dir='runs',
            project_name='my_training',
            resume_training=True,  # Включаем режим возобновления
            base_weights_path=last_pt_path  # Указываем путь к чекпоинту
        )
        
        if experiment_id:
            print(f"Обучение возобновлено из {last_pt_path}. ID эксперимента: {experiment_id}")
        else:
            print("Ошибка возобновления обучения")
    else:
        print(f"Файл last.pt не найден: {last_pt_path}")
    
    # Пример использования с ultralytics напрямую (как в документации)
    print("\nСпособ 3: Использование ultralytics напрямую")
    print("""
    from ultralytics import YOLO
    
    # Load a model
    model = YOLO("path/to/last.pt")  # load a partially trained model
    
    # Resume training
    results = model.train(resume=True)
    """)
    print("Этот способ реализован автоматически при использовании resume_training=True")


def example_model_validation():
    """Пример валидации модели"""
    print("\n=== Пример валидации модели ===")
    
    manager = YOLOTrainingManager()
    
    # Пути к модели и датасету
    model_path = "path/to/your/trained/model.pt"
    dataset_path = "path/to/your/dataset"
    task = 'detect'  # или 'segment'
    
    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        print("Модель или датасет не найдены")
        return
    
    # Простая валидация
    print("Выполнение простой валидации...")
    simple_results = manager.validate_model(
        model_path=model_path,
        dataset_path=dataset_path,
        task=task,
        comprehensive=False
    )
    print(f"Простая валидация: {simple_results}")
    
    # Комплексная валидация
    print("Выполнение комплексной валидации...")
    comprehensive_results = manager.validate_model(
        model_path=model_path,
        dataset_path=dataset_path,
        task=task,
        comprehensive=True
    )
    print(f"Комплексная валидация: {comprehensive_results}")


def example_model_comparison():
    """Пример сравнения моделей"""
    print("\n=== Пример сравнения моделей ===")
    
    manager = YOLOTrainingManager()
    
    # Список моделей для сравнения
    models = [
        {'name': 'Model_1', 'path': 'path/to/model1.pt'},
        {'name': 'Model_2', 'path': 'path/to/model2.pt'},
        {'name': 'Model_3', 'path': 'path/to/model3.pt'}
    ]
    
    dataset_path = "path/to/your/dataset"
    task = 'detect'
    
    if not os.path.exists(dataset_path):
        print("Датасет не найден")
        return
    
    # Сравниваем модели
    comparison_results = manager.compare_models(
        models=models,
        dataset_path=dataset_path,
        task=task
    )
    
    print(f"Результаты сравнения: {comparison_results}")
    
    # Выводим ранжирование
    if 'ranking' in comparison_results:
        ranking = comparison_results['ranking']
        print(f"Лучшая модель: {ranking.get('best_overall', 'Не определена')}")
        
        if 'ranked_models' in ranking:
            print("Ранжирование моделей:")
            for i, model in enumerate(ranking['ranked_models'], 1):
                print(f"{i}. {model['name']}: {model['score']:.4f}")


def example_dataset_analysis():
    """Пример анализа датасета"""
    print("\n=== Пример анализа датасета ===")
    
    manager = YOLOTrainingManager()
    
    dataset_path = "path/to/your/dataset"
    task = 'detect'  # или 'segment'
    
    if not os.path.exists(dataset_path):
        print("Датасет не найден")
        return
    
    # Анализируем датасет
    analysis_results = manager.analyze_dataset(dataset_path, task)
    
    print(f"Результаты анализа датасета:")
    print(f"Общее количество изображений: {analysis_results.get('total_images', 0)}")
    print(f"Общее количество аннотаций: {analysis_results.get('total_annotations', 0)}")
    
    # Анализ по сплитам
    splits = analysis_results.get('splits', {})
    for split_name, split_data in splits.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Изображений: {split_data.get('image_count', 0)}")
        print(f"  Аннотаций: {split_data.get('annotation_count', 0)}")
        print(f"  Объектов: {split_data.get('total_objects', 0)}")
        print(f"  Объектов на изображение: {split_data.get('objects_per_image', 0):.2f}")


def example_config_management():
    """Пример управления конфигурациями"""
    print("\n=== Пример управления конфигурациями ===")
    
    config_manager = TrainingConfigManager('configs')
    
    # Получаем конфигурации по умолчанию
    default_configs = config_manager.get_default_configs()
    
    # Сохраняем конфигурации
    for name, config in default_configs.items():
        success = config_manager.save_config(config, name)
        if success:
            print(f"Конфигурация '{name}' сохранена")
    
    # Загружаем конфигурацию
    config = config_manager.load_config('detection_fast')
    if config:
        print(f"Загружена конфигурация: {config}")
    
    # Получаем список конфигураций
    configs_list = config_manager.list_configs()
    print(f"Доступные конфигурации: {configs_list}")


def example_metrics_tracking():
    """Пример отслеживания метрик"""
    print("\n=== Пример отслеживания метрик ===")
    
    manager = YOLOTrainingManager()
    
    # Получаем список всех экспериментов
    experiments = manager.get_all_experiments()
    print(f"Всего экспериментов: {len(experiments)}")
    
    for exp in experiments:
        print(f"Эксперимент: {exp['name']} ({exp['id']})")
        print(f"  Задача: {exp['task']}")
        print(f"  Модель: {exp['model_type']}")
        print(f"  Статус: {exp['status']}")
        print(f"  Создан: {exp['created_at']}")
    
    # Получаем сводку по конкретному эксперименту
    if experiments:
        exp_id = experiments[0]['id']
        summary = manager.get_experiment_summary(exp_id)
        print(f"\nСводка по эксперименту {exp_id}:")
        print(f"  Всего эпох: {summary.get('total_epochs', 0)}")
        print(f"  Лучшие метрики: {summary.get('best_metrics', {})}")
        print(f"  Финальные метрики: {summary.get('final_metrics', {})}")
        
        # Создаем графики обучения
        output_dir = f"plots/{exp_id}"
        success = manager.create_training_plots(exp_id, output_dir)
        if success:
            print(f"Графики обучения сохранены в: {output_dir}")
        
        # Экспортируем данные эксперимента
        export_path = f"export/{exp_id}_data.json"
        success = manager.export_experiment_data(exp_id, export_path, 'json')
        if success:
            print(f"Данные эксперимента экспортированы в: {export_path}")


def main():
    """Главная функция с примерами"""
    print("Примеры использования системы обучения YOLO")
    print("=" * 50)
    
    # Создаем необходимые директории
    os.makedirs('logs', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('export', exist_ok=True)
    
    # Запускаем примеры
    try:
        example_config_management()
        example_dataset_analysis()
        example_metrics_tracking()
        
        # Раскомментируйте для запуска обучения (требует реальные датасеты)
        # example_detection_training()
        # example_segmentation_training()
        # example_resume_training()  # Пример возобновления обучения
        # example_model_validation()
        # example_model_comparison()
        
    except Exception as e:
        print(f"Ошибка выполнения примеров: {e}")
    
    print("\nПримеры завершены!")


if __name__ == "__main__":
    main()

