# processing_utils.py

from qgis.PyQt import QtWidgets

class ProgressReporter:
    """
    Класс-обертка для управления QProgressBar и отслеживания отмены.
    """
    def __init__(self, progress_bar: QtWidgets.QProgressBar, start_percentage: int, end_percentage: int):
        self.progress_bar = progress_bar
        self.start_percentage = start_percentage
        self.range = end_percentage - start_percentage
        self._is_canceled = False

    def set_progress(self, current_step, total_steps):
        """Обновляет значение ProgressBar на основе текущего шага."""
        if total_steps > 0:
            percentage = (current_step / total_steps) * self.range
            self.progress_bar.setValue(self.start_percentage + int(percentage))
            QtWidgets.QApplication.processEvents() # Обновляем интерфейс

    def cancel(self):
        self._is_canceled = True

    def is_canceled(self):
        return self._is_canceled