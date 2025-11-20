# -*- coding: utf-8 -*-
"""
Исправление для проблемы NumPy stderr в среде QGIS
Обеспечивает правильную инициализацию sys.stderr перед операциями NumPy
"""

# Импортируем только стандартные библиотеки, чтобы избежать циклических импортов
import sys

# Инициализация sys.stderr для предотвращения ошибок NumPy в QGIS
if not hasattr(sys, "stderr") or sys.stderr is None:
    # Если stderr не инициализирован, создаем его
    try:
        sys.stderr = sys.__stderr__
    except AttributeError:
        # Если __stderr__ тоже отсутствует, создаем новый
        import io

        sys.stderr = io.TextIOWrapper(io.BytesIO())
