# main.py
import sys
import os
import importlib

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QLabel

from check_dependencies import ensure_dependencies_installed
from base_plugin import BasePlugin

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("M.A.I.D")
        self.resize(1300, 800)

        # Attempt to load plugins
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.plugins = load_plugins()

        if not self.plugins:
            self.tabs.hide()
            self.label_no_plugins = QLabel("No plugins installed.")
            self.label_no_plugins.setStyleSheet("font-size: 20px;")
            self.setCentralWidget(self.label_no_plugins)
        else:
            for plugin in self.plugins:
                tab_widget = plugin.create_tab()
                self.tabs.addTab(tab_widget, plugin.plugin_name)

def main():
    # 1) Ensure dependencies
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

    # 2) Launch the PyQt app
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
