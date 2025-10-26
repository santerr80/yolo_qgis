# -*- coding: utf-8 -*-
"""
Утилита для проверки совместимости зависимостей с numpy v1
"""

import sys
import subprocess
from typing import Dict, List, Tuple


class CompatibilityChecker:
    """Класс для проверки совместимости зависимостей"""
    
    def __init__(self):
        self.required_packages = {
            'numpy': '>=1.21.0,<2.0.0',
            'pandas': '>=1.3.0,<2.0.0',
            'matplotlib': '>=3.4.0,<4.0.0',
            'opencv-python': '>=4.5.0,<5.0.0',
            'ultralytics': '>=8.0.0',
            'torch': '>=1.9.0',
            'torchvision': '>=0.10.0',
            'scikit-learn': '>=1.0.0,<2.0.0',
            'scipy': '>=1.7.0,<2.0.0'
        }
        
        self.compatibility_issues = []
        self.warnings = []
    
    def check_numpy_compatibility(self) -> Tuple[bool, str]:
        """
        Проверяет совместимость numpy с другими пакетами
        
        :return: (совместимость, сообщение)
        """
        try:
            import numpy as np
            
            if not hasattr(np, '__version__'):
                return False, "Не удалось определить версию numpy"
            
            version_parts = np.__version__.split('.')
            major_version = int(version_parts[0])
            minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0
            
            if major_version >= 2:
                return False, f"numpy версии {np.__version__} несовместима с QGIS. Требуется numpy v1.x"
            
            if major_version == 1 and minor_version < 21:
                return False, f"numpy версии {np.__version__} слишком старая. Требуется >= 1.21.0"
            
            return True, f"numpy версии {np.__version__} совместима"
            
        except ImportError:
            return False, "numpy не установлен"
        except Exception as e:
            return False, f"Ошибка проверки numpy: {e}"
    
    def check_package_versions(self) -> Dict[str, Tuple[bool, str]]:
        """
        Проверяет версии всех требуемых пакетов
        
        :return: Словарь с результатами проверки
        """
        results = {}
        
        for package, version_req in self.required_packages.items():
            try:
                # Получаем установленную версию
                result = subprocess.run(
                    [sys.executable, '-c', f'import {package}; print({package}.__version__)'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    installed_version = result.stdout.strip()
                    is_compatible, message = self._check_version_compatibility(
                        package, installed_version, version_req
                    )
                    results[package] = (is_compatible, message)
                else:
                    results[package] = (False, f"Пакет {package} не установлен")
                    
            except subprocess.TimeoutExpired:
                results[package] = (False, f"Таймаут при проверке {package}")
            except Exception as e:
                results[package] = (False, f"Ошибка проверки {package}: {e}")
        
        return results
    
    def _check_version_compatibility(self, package: str, installed_version: str, version_req: str) -> Tuple[bool, str]:
        """
        Проверяет совместимость версии пакета
        
        :param package: Название пакета
        :param installed_version: Установленная версия
        :param version_req: Требуемая версия
        :return: (совместимость, сообщение)
        """
        try:
            # Простая проверка для numpy v1 совместимости
            if package == 'numpy':
                version_parts = installed_version.split('.')
                major_version = int(version_parts[0])
                
                if major_version >= 2:
                    return False, f"numpy {installed_version} несовместима с QGIS (требуется v1.x)"
                elif major_version == 1:
                    minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0
                    if minor_version < 21:
                        return False, f"numpy {installed_version} слишком старая (требуется >= 1.21.0)"
                    else:
                        return True, f"numpy {installed_version} совместима"
            
            # Для других пакетов проверяем основные версии
            version_parts = installed_version.split('.')
            major_version = int(version_parts[0])
            
            if '>=1.0.0,<2.0.0' in version_req:
                if major_version >= 2:
                    return False, f"{package} {installed_version} может быть несовместима (требуется v1.x)"
                else:
                    return True, f"{package} {installed_version} совместима"
            
            return True, f"{package} {installed_version} установлена"
            
        except (ValueError, IndexError) as e:
            return False, f"Не удалось разобрать версию {package}: {installed_version}"
    
    def check_qgis_compatibility(self) -> Tuple[bool, str]:
        """
        Проверяет совместимость с QGIS
        
        :return: (совместимость, сообщение)
        """
        try:
            from qgis.core import QgsApplication
            return True, f"QGIS совместим (версия: {QgsApplication.instance().applicationVersion()})"
        except ImportError:
            return False, "QGIS не найден в окружении"
        except Exception as e:
            return False, f"Ошибка проверки QGIS: {e}"
    
    def run_full_check(self) -> Dict:
        """
        Выполняет полную проверку совместимости
        
        :return: Результаты проверки
        """
        results = {
            'numpy_compatibility': self.check_numpy_compatibility(),
            'package_versions': self.check_package_versions(),
            'qgis_compatibility': self.check_qgis_compatibility(),
            'overall_compatible': True,
            'issues': [],
            'warnings': []
        }
        
        # Проверяем общую совместимость
        if not results['numpy_compatibility'][0]:
            results['overall_compatible'] = False
            results['issues'].append(results['numpy_compatibility'][1])
        
        if not results['qgis_compatibility'][0]:
            results['overall_compatible'] = False
            results['issues'].append(results['qgis_compatibility'][1])
        
        # Проверяем совместимость пакетов
        for package, (is_compatible, message) in results['package_versions'].items():
            if not is_compatible:
                results['overall_compatible'] = False
                results['issues'].append(f"{package}: {message}")
            elif 'может быть несовместима' in message:
                results['warnings'].append(f"{package}: {message}")
        
        return results
    
    def print_report(self, results: Dict):
        """
        Выводит отчет о совместимости
        
        :param results: Результаты проверки
        """
        print("=" * 60)
        print("ОТЧЕТ О СОВМЕСТИМОСТИ ЗАВИСИМОСТЕЙ")
        print("=" * 60)
        
        # Общий статус
        status = "✓ СОВМЕСТИМО" if results['overall_compatible'] else "✗ НЕСОВМЕСТИМО"
        print(f"\nОбщий статус: {status}")
        
        # Проверка numpy
        numpy_ok, numpy_msg = results['numpy_compatibility']
        numpy_status = "✓" if numpy_ok else "✗"
        print(f"\n{numpy_status} numpy: {numpy_msg}")
        
        # Проверка QGIS
        qgis_ok, qgis_msg = results['qgis_compatibility']
        qgis_status = "✓" if qgis_ok else "✗"
        print(f"{qgis_status} QGIS: {qgis_msg}")
        
        # Проверка пакетов
        print(f"\nПроверка пакетов:")
        for package, (is_compatible, message) in results['package_versions'].items():
            status = "✓" if is_compatible else "✗"
            print(f"  {status} {package}: {message}")
        
        # Проблемы
        if results['issues']:
            print(f"\n✗ ПРОБЛЕМЫ:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        # Предупреждения
        if results['warnings']:
            print(f"\n⚠ ПРЕДУПРЕЖДЕНИЯ:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        # Рекомендации
        if not results['overall_compatible']:
            print(f"\nРЕКОМЕНДАЦИИ:")
            print("1. Установите numpy версии 1.x: pip install 'numpy>=1.21.0,<2.0.0'")
            print("2. Обновите несовместимые пакеты до версий, совместимых с numpy v1")
            print("3. Используйте виртуальное окружение для изоляции зависимостей")
        
        print("=" * 60)


def main():
    """Основная функция для запуска проверки"""
    checker = CompatibilityChecker()
    results = checker.run_full_check()
    checker.print_report(results)
    
    # Возвращаем код выхода
    return 0 if results['overall_compatible'] else 1


if __name__ == "__main__":
    sys.exit(main())
