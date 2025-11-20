# -*- coding: utf-8 -*-
"""
Утилиты для работы с устройствами (CPU/CUDA)
Проверка доступности и информации о GPU
"""

import logging
import sys
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


def _ensure_logger_handler():
    """
    Убеждается, что у логгера есть валидный обработчик.
    Исправляет проблему с None stream в QGIS окружении.
    """
    # Проверяем существующие обработчики
    has_valid_handler = False
    handlers_to_remove = []
    
    for handler in logger.handlers:
        if hasattr(handler, 'stream'):
            if handler.stream is None:
                # Помечаем для удаления обработчики с None stream
                handlers_to_remove.append(handler)
            else:
                has_valid_handler = True
    
    # Удаляем невалидные обработчики
    for handler in handlers_to_remove:
        logger.removeHandler(handler)
    
    # Если нет валидного обработчика, добавляем новый
    if not has_valid_handler:
        try:
            # Пытаемся использовать sys.stdout, если доступен
            if sys.stdout and not sys.stdout.closed:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
        except (AttributeError, OSError):
            # Если sys.stdout недоступен, используем NullHandler
            logger.addHandler(logging.NullHandler())


# Инициализируем обработчик при импорте модуля
_ensure_logger_handler()


def test_cuda_devices() -> Dict[str, Any]:
    """
    Независимый тест CUDA устройств с подробным выводом информации в лог
    
    :return: Словарь с результатами теста
    """
    result = {
        'cuda_available': False,
        'cuda_version': None,
        'device_count': 0,
        'devices': [],
        'errors': []
    }
    
    logger.info("=" * 60)
    logger.info("НАЧАЛО ТЕСТИРОВАНИЯ CUDA УСТРОЙСТВ")
    logger.info("=" * 60)
    
    # Проверяем наличие PyTorch
    try:
        import torch
        logger.info("✓ PyTorch найден")
        logger.info(f"  Версия PyTorch: {torch.__version__}")
    except ImportError:
        error_msg = "✗ PyTorch не установлен"
        logger.error(error_msg)
        result['errors'].append(error_msg)
        logger.info("=" * 60)
        logger.info("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        logger.info("=" * 60)
        return result
    
    # Проверяем доступность CUDA
    try:
        cuda_available = torch.cuda.is_available()
        result['cuda_available'] = cuda_available
        
        if cuda_available:
            logger.info("✓ CUDA доступна")
            
            # Получаем версию CUDA
            try:
                cuda_version = torch.version.cuda
                result['cuda_version'] = cuda_version
                logger.info(f"  Версия CUDA: {cuda_version}")
            except Exception as e:
                logger.warning(f"  Не удалось получить версию CUDA: {e}")
            
            # Получаем количество устройств
            device_count = torch.cuda.device_count()
            result['device_count'] = device_count
            logger.info(f"  Количество CUDA устройств: {device_count}")
            
            # Получаем информацию о каждом устройстве
            for i in range(device_count):
                device_info = {
                    'index': i,
                    'name': None,
                    'memory_total': None,
                    'memory_allocated': None,
                    'memory_free': None,
                    'compute_capability': None
                }
                
                try:
                    # Имя устройства
                    device_name = torch.cuda.get_device_name(i)
                    device_info['name'] = device_name
                    logger.info(f"\n  Устройство {i}: {device_name}")
                    
                    # Вычислительная способность
                    try:
                        major, minor = torch.cuda.get_device_capability(i)
                        compute_cap = f"{major}.{minor}"
                        device_info['compute_capability'] = compute_cap
                        logger.info(f"    Вычислительная способность: {compute_cap}")
                    except Exception as e:
                        logger.warning(f"    Не удалось получить вычислительную способность: {e}")
                    
                    # Информация о памяти
                    try:
                        # Переключаемся на устройство для получения информации о памяти
                        with torch.cuda.device(i):
                            memory_total = torch.cuda.get_device_properties(i).total_memory
                            memory_allocated = torch.cuda.memory_allocated(i)
                            memory_free = memory_total - memory_allocated
                            
                            device_info['memory_total'] = memory_total
                            device_info['memory_allocated'] = memory_allocated
                            device_info['memory_free'] = memory_free
                            
                            # Конвертируем в GB
                            memory_total_gb = memory_total / (1024 ** 3)
                            memory_allocated_gb = memory_allocated / (1024 ** 3)
                            memory_free_gb = memory_free / (1024 ** 3)
                            
                            logger.info(f"    Память:")
                            logger.info(f"      Всего: {memory_total_gb:.2f} GB ({memory_total:,} байт)")
                            logger.info(f"      Использовано: {memory_allocated_gb:.2f} GB ({memory_allocated:,} байт)")
                            logger.info(f"      Свободно: {memory_free_gb:.2f} GB ({memory_free:,} байт)")
                    except Exception as e:
                        logger.warning(f"    Не удалось получить информацию о памяти: {e}")
                    
                    # Дополнительная информация об устройстве
                    try:
                        props = torch.cuda.get_device_properties(i)
                        logger.info(f"    Дополнительная информация:")
                        logger.info(f"      Мультипроцессоры: {props.multi_processor_count}")
                        # max_threads_per_block не является атрибутом torch._C._CudaDeviceProperties
                        # Используем альтернативные доступные свойства
                        if hasattr(props, 'major') and hasattr(props, 'minor'):
                            logger.info(f"      Compute Capability: {props.major}.{props.minor}")
                    except Exception as e:
                        logger.warning(f"    Не удалось получить дополнительную информацию: {e}")
                    
                    result['devices'].append(device_info)
                    
                except Exception as e:
                    error_msg = f"Ошибка при получении информации об устройстве {i}: {e}"
                    logger.error(f"  ✗ {error_msg}")
                    result['errors'].append(error_msg)
            
            # Проверяем текущее устройство
            try:
                current_device = torch.cuda.current_device()
                logger.info(f"\n  Текущее активное устройство: {current_device}")
            except Exception as e:
                logger.warning(f"  Не удалось определить текущее устройство: {e}")
            
            # Проверяем переменную окружения CUDA_VISIBLE_DEVICES
            import os
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible:
                logger.info(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
            else:
                logger.info(f"  CUDA_VISIBLE_DEVICES: не установлена")
            
        else:
            logger.warning("✗ CUDA недоступна")
            logger.info("  Причины могут быть:")
            logger.info("    - CUDA драйверы не установлены")
            logger.info("    - PyTorch скомпилирован без поддержки CUDA")
            logger.info("    - Нет совместимых GPU устройств")
            
            # Проверяем переменную окружения
            import os
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible:
                logger.info(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
                logger.warning("  Внимание: CUDA_VISIBLE_DEVICES установлена, но CUDA недоступна")
            
    except Exception as e:
        error_msg = f"Ошибка при проверке CUDA: {e}"
        logger.error(f"✗ {error_msg}")
        result['errors'].append(error_msg)
    
    # Проверяем cuDNN
    try:
        if torch.backends.cudnn.enabled:
            logger.info("\n✓ cuDNN включен")
            try:
                cudnn_version = torch.backends.cudnn.version()
                logger.info(f"  Версия cuDNN: {cudnn_version}")
            except:
                logger.info("  Версия cuDNN: неизвестна")
        else:
            logger.warning("\n✗ cuDNN отключен")
    except Exception as e:
        logger.warning(f"\nНе удалось проверить cuDNN: {e}")
    
    # Итоговая информация
    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    logger.info("=" * 60)
    logger.info(f"CUDA доступна: {'Да' if result['cuda_available'] else 'Нет'}")
    logger.info(f"Количество устройств: {result['device_count']}")
    if result['cuda_version']:
        logger.info(f"Версия CUDA: {result['cuda_version']}")
    if result['errors']:
        logger.warning(f"Обнаружено ошибок: {len(result['errors'])}")
        for error in result['errors']:
            logger.warning(f"  - {error}")
    logger.info("=" * 60)
    logger.info("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    logger.info("=" * 60)
    
    return result


def get_recommended_device() -> str:
    """
    Определяет рекомендуемое устройство для обучения
    
    :return: Рекомендуемое устройство ('cpu' или номер GPU)
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Проверяем доступность памяти на первом устройстве
            try:
                memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                memory_free_gb = memory_free / (1024 ** 3)
                
                # Если свободно меньше 1 GB, рекомендуем CPU
                if memory_free_gb < 1.0:
                    logger.warning(f"На GPU доступно только {memory_free_gb:.2f} GB памяти, рекомендуется использовать CPU")
                    return 'cpu'
                
                return '0'
            except:
                return '0'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'


def check_device_availability(device: str) -> Tuple[bool, Optional[str]]:
    """
    Проверяет доступность указанного устройства
    
    :param device: Устройство для проверки ('cpu', '0', '1', и т.д.)
    :return: (доступно_ли, сообщение_об_ошибке)
    """
    if device == 'cpu':
        return True, None
    
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA недоступна"
        
        try:
            device_id = int(device)
            if device_id < 0 or device_id >= torch.cuda.device_count():
                return False, f"Устройство {device_id} не существует (доступно устройств: {torch.cuda.device_count()})"
            return True, None
        except ValueError:
            return False, f"Некорректный формат устройства: {device}"
    except ImportError:
        return False, "PyTorch не установлен"


# Пример использования:
# 
# if __name__ == "__main__":
#     # Настройка логирования для вывода в консоль
#     import sys
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[logging.StreamHandler(sys.stdout)]
#     )
#     
#     # Запуск теста CUDA устройств
#     result = test_cuda_devices()
#     
#     # Использование результатов
#     if result['cuda_available']:
#         print(f"Найдено {result['device_count']} CUDA устройств")
#         for device in result['devices']:
#             print(f"  - {device['name']}: {device['memory_total'] / (1024**3):.2f} GB")
#     else:
#         print("CUDA недоступна, будет использоваться CPU")
#     
#     # Получение рекомендуемого устройства
#     recommended = get_recommended_device()
#     print(f"Рекомендуемое устройство: {recommended}")

