#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для исправления проблемы с PIL ImageOps
"""

import subprocess
import sys
import os


def run_command(command):
    """Выполняет команду и возвращает результат"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def fix_pillow_issue():
    """Исправляет проблему с PIL ImageOps"""
    print("Исправление проблемы с PIL ImageOps...")

    # 1. Удаляем старую версию Pillow
    print("1. Удаление старой версии Pillow...")
    success, stdout, stderr = run_command("pip uninstall Pillow -y")
    if success:
        print("   ✓ Pillow удален")
    else:
        print(f"   ⚠ Предупреждение при удалении Pillow: {stderr}")

    # 2. Устанавливаем совместимую версию Pillow
    print("2. Установка совместимой версии Pillow (9.4.0)...")
    success, stdout, stderr = run_command("pip install Pillow==9.4.0")
    if success:
        print("   ✓ Pillow 9.4.0 установлен")
    else:
        print(f"   ✗ Ошибка установки Pillow: {stderr}")
        return False

    # 3. Проверяем импорт ImageOps
    print("3. Проверка импорта ImageOps...")
    try:
        from PIL import ImageOps

        print("   ✓ ImageOps успешно импортирован")
        return True
    except ImportError as e:
        print(f"   ✗ Ошибка импорта ImageOps: {e}")
        return False


def main():
    """Основная функция"""
    print("=" * 50)
    print("ИСПРАВЛЕНИЕ ПРОБЛЕМЫ С PIL ImageOps")
    print("=" * 50)

    # Проверяем, что мы в правильной среде
    if "QGIS" not in os.environ.get("PYTHONPATH", ""):
        print("⚠ Предупреждение: Скрипт запущен не в среде QGIS")
        print("  Убедитесь, что вы запускаете скрипт из QGIS Python консоли")

    success = fix_pillow_issue()

    if success:
        print("\n" + "=" * 50)
        print("✓ ПРОБЛЕМА ИСПРАВЛЕНА!")
        print("Теперь можно запускать тренировку YOLO моделей")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("✗ ПРОБЛЕМА НЕ ИСПРАВЛЕНА")
        print("Попробуйте выполнить команды вручную:")
        print("pip uninstall Pillow -y")
        print("pip install Pillow==9.4.0")
        print("=" * 50)


if __name__ == "__main__":
    main()
