# base_plugin.py
from PyQt5.QtWidgets import QWidget

class BasePlugin:
    """
    Base class for all plugins. Each plugin must implement:
      - self.plugin_name (str): the tab's name
      - create_tab(self) -> QWidget: returns the tab widget
    """
    def __init__(self):
        self.plugin_name = "Base Plugin"

    def create_tab(self) -> QWidget:
        raise NotImplementedError("Plugins must implement create_tab()")
