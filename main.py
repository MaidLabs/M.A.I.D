import sys
import os
import importlib
import logging

from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QIcon, QPalette, QColor, QClipboard
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QLabel,
    QDockWidget,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QStyleFactory,
    QHBoxLayout,
    QFileDialog,
    QComboBox,
    QLineEdit
)

from check_dependencies import ensure_dependencies_installed
from base_plugin import BasePlugin


class EmittingStream(QObject):
    """
    A helper class to redirect stdout / stderr to both the PyCharm console
    and the GUI log panel.
    """
    textWritten = pyqtSignal(str)

    def __init__(self, original_stream):
        super().__init__()
        self._original_stream = original_stream  # Save a reference to the original stream

    def write(self, text):
        self.textWritten.emit(str(text))
        self._original_stream.write(text)

    def flush(self):
        self._original_stream.flush()


class LogBoxHandler(logging.Handler):
    """
    A logging handler that appends log records to a QPlainTextEdit widget.
    Useful to ensure all logs, regardless of logger configuration, appear in the GUI log box.
    """
    def __init__(self, log_text_edit):
        super().__init__()
        self.log_text_edit = log_text_edit

    def emit(self, record):
        msg = self.format(record)
        self.log_text_edit.appendPlainText(msg)


def load_plugins():
    """
    Dynamically load all .py modules in the 'plugins' folder that
    define a class named 'Plugin' inheriting from BasePlugin.
    """
    plugins_dir = os.path.join(os.path.dirname(__file__), "plugins")
    plugins = []

    if not os.path.isdir(plugins_dir):
        return plugins

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            full_module_name = f"plugins.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
                if hasattr(module, "Plugin"):
                    plugin_class = getattr(module, "Plugin")
                    if issubclass(plugin_class, BasePlugin):
                        plugin_instance = plugin_class()
                        plugins.append(plugin_instance)
            except Exception as e:
                print(f"[WARN] Could not load plugin {filename}: {e}")
    return plugins


class MainWindow(QMainWindow):
    def __init__(self, logo_path=None):
        super().__init__()
        self.setWindowTitle("M.A.I.D")

        # If you have a path to a logo, set it as the window icon
        if logo_path and os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))

        self.resize(1300, 800)

        # Central widget and layout
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Create a tab widget for loaded plugins
        self.tabs = QTabWidget()
        # Move tabs to the top (North)
        self.tabs.setTabPosition(QTabWidget.North)

        # Load plugins
        self.plugins = load_plugins()

        # If no plugins found, show a label; otherwise show plugin tabs
        if not self.plugins:
            label_no_plugins = QLabel("No plugins installed.")
            label_no_plugins.setStyleSheet("font-size: 20px;")
            central_layout.addWidget(label_no_plugins)
        else:
            central_layout.addWidget(self.tabs)
            for plugin in self.plugins:
                tab_widget = plugin.create_tab()
                self.tabs.addTab(tab_widget, plugin.plugin_name)

        # -------------------- Logs Dock Widget --------------------
        self.log_dock = QDockWidget("Logs", self)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.log_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        flags = self.log_dock.windowFlags()
        flags |= Qt.WindowMinimizeButtonHint
        flags |= Qt.WindowMaximizeButtonHint
        self.log_dock.setWindowFlags(flags)

        self.log_text_edit = QPlainTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_dock.setWidget(self.log_text_edit)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

        # -------------------- "Logs" Tab with Actions --------------------
        logs_tab = QWidget()
        logs_tab_layout = QVBoxLayout(logs_tab)

        # A button to show the Logs dock if it's closed
        self.open_logs_button = QPushButton("Open Logs")
        self.open_logs_button.clicked.connect(self.show_logs_dock)
        logs_tab_layout.addWidget(self.open_logs_button)

        # A horizontal layout for additional log controls
        controls_layout = QHBoxLayout()

        # 1) Clear Logs Button
        self.clear_logs_button = QPushButton("Clear Logs")
        self.clear_logs_button.clicked.connect(self.clear_logs)
        controls_layout.addWidget(self.clear_logs_button)

        # 2) Copy Logs Button
        self.copy_logs_button = QPushButton("Copy Logs")
        self.copy_logs_button.clicked.connect(self.copy_logs_to_clipboard)
        controls_layout.addWidget(self.copy_logs_button)

        # 3) Save Logs Button
        self.save_logs_button = QPushButton("Save Logs")
        self.save_logs_button.clicked.connect(self.save_logs_to_file)
        controls_layout.addWidget(self.save_logs_button)

        # 4) Log Level Filter
        self.log_level_filter = QComboBox()
        self.log_level_filter.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_filter.currentIndexChanged.connect(self.filter_logs)
        controls_layout.addWidget(self.log_level_filter)

        # 5) Search Field
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search logs...")
        self.search_field.textChanged.connect(self.search_logs)
        controls_layout.addWidget(self.search_field)

        logs_tab_layout.addLayout(controls_layout)

        # Add this new "Logs" tab to the tab widget
        self.tabs.addTab(logs_tab, "Logs")

        # -------------------- Redirect stdout/stderr --------------------
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        self.stdout_stream = EmittingStream(self.original_stdout)
        self.stderr_stream = EmittingStream(self.original_stderr)

        self.stdout_stream.textWritten.connect(self.append_log)
        self.stderr_stream.textWritten.connect(self.append_log)

        sys.stdout = self.stdout_stream
        sys.stderr = self.stderr_stream

        # -------------------- Logging Handler --------------------
        self.log_handler = LogBoxHandler(self.log_text_edit)
        self.log_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
        self.log_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(self.log_handler)

        # Keep a list of all log lines for filtering
        self.all_logs = []

    # -------------------- Log Handling Methods --------------------
    def append_log(self, text):
        """
        Append text to the log text box and save it to self.all_logs
        so we can filter/search later.
        """
        self.all_logs.append(text)
        self.log_text_edit.appendPlainText(text)

    def clear_logs(self):
        """
        Clears the log text box and local log history.
        """
        self.all_logs.clear()
        self.log_text_edit.clear()

    def copy_logs_to_clipboard(self):
        """
        Copies the current log text to the clipboard.
        """
        clipboard = QApplication.instance().clipboard()
        clipboard.setText(self.log_text_edit.toPlainText())

    def save_logs_to_file(self):
        """
        Opens a file dialog and saves the current log content to a text file.
        """
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Logs",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.log_text_edit.toPlainText())

    def filter_logs(self):
        """
        Show only logs at/above the chosen level (DEBUG, INFO, etc.).
        """
        selected_level = self.log_level_filter.currentText()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        threshold = level_map[selected_level]

        # Clear displayed text
        self.log_text_edit.clear()

        # Display only lines at/above threshold
        for line in self.all_logs:
            # Check bracketed log level, e.g. [INFO], [DEBUG], etc.
            if line.strip().startswith("[") and "]" in line:
                bracket_content = line.split("]")[0][1:].upper()  # e.g. "DEBUG"
                line_level = level_map.get(bracket_content, logging.DEBUG)
                if line_level >= threshold:
                    self.log_text_edit.appendPlainText(line)
            else:
                # If we can't parse a bracket, just display the line
                self.log_text_edit.appendPlainText(line)

    def search_logs(self, text):
        """
        Filter displayed logs by the current level filter AND by search text.
        """
        selected_level = self.log_level_filter.currentText()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        threshold = level_map[selected_level]
        search_lower = text.lower()

        # Clear displayed text
        self.log_text_edit.clear()

        for line in self.all_logs:
            # First check log level
            if line.strip().startswith("[") and "]" in line:
                bracket_content = line.split("]")[0][1:].upper()
                line_level = level_map.get(bracket_content, logging.DEBUG)
                if line_level < threshold:
                    continue

            # Then check the search text
            if search_lower in line.lower():
                self.log_text_edit.appendPlainText(line)

    def show_logs_dock(self):
        """
        Show/re-open the logs dock if it was closed.
        """
        self.log_dock.show()


def set_dark_theme(app):
    """
    Configure the Fusion style with a custom dark palette for a modern 'dark mode'.
    """
    app.setStyle(QStyleFactory.create("Fusion"))
    dark_palette = QPalette()

    # Base colors
    dark_color = QColor(53, 53, 53)
    highlight_color = QColor(142, 45, 197)
    dark_palette.setColor(QPalette.Window, dark_color)
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, dark_color)
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, dark_color)
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, highlight_color)
    dark_palette.setColor(QPalette.Highlight, highlight_color)
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)

    # Style sheet includes spacing (margin) for tabs
    app.setStyleSheet("""
        QToolTip {
            color: #ffffff;
            background-color: #2a2a2a;
            border: 1px solid #333;
        }
        QTabWidget::pane {
            border: none;
        }
        QTabBar::tab {
            background: #3d3d3d;
            padding: 6px;
            margin-right: 8px; /* Spacing between tabs */
        }
        QTabBar::tab:selected {
            background: #5a5a5a;
            margin-right: 8px; /* Keep the same spacing for the selected tab */
        }
        QPushButton {
            background-color: #3d3d3d;
            border: 1px solid #5a5a5a;
            padding: 5px 10px;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
    """)


def main():
    needed_packages = [
        "PyQt5",
        "matplotlib",
        "torch",
        "torchvision",
        "pytorch_lightning",
        "opencv-python",
        "albumentations",
        "scikit-learn",
        "numpy",
        "Pillow"
    ]
    ensure_dependencies_installed(needed_packages)

    app = QApplication(sys.argv)

    # ---- Apply dark theme ----
    set_dark_theme(app)

    # Provide the path to your logo/icon here, e.g. "path/to/logo.png"
    window = MainWindow(logo_path="logo.png")
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
