# plugins/plugin_data_exploration.py

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io import BytesIO

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QScrollArea, QMessageBox, QSpinBox,
    QTextEdit, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from base_plugin import BasePlugin

def convert_to_qpixmap(numpy_image: np.ndarray) -> QPixmap:
    """
    Convert a BGR or RGB numpy image (uint8) to QPixmap safely.
    """
    array_copied = np.ascontiguousarray(numpy_image)
    height, width, channels = array_copied.shape
    bytes_per_line = channels * width
    # QImage.Format_RGB888, then .rgbSwapped() to go from BGR->RGB
    q_image = QImage(array_copied.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_image.copy())


class DataExplorationThread(QThread):
    """
    Thread to scan the given data folder, gather class counts,
    up to N sample images from each class, etc., without freezing the UI.
    """
    progress_signal = pyqtSignal(str)      # for logging
    done_signal = pyqtSignal(dict, list)   # (stats_dict, class_info_list)
    # stats_dict = {"classes": [...], "counts": [...]} for chart
    # class_info_list = [(class_name, [list_of_img_paths]), ...]

    def __init__(self, data_folder, max_images_per_class=3, parent=None):
        super().__init__(parent)
        self.data_folder = data_folder
        self.max_images = max_images_per_class

    def run(self):
        if not os.path.isdir(self.data_folder):
            self.done_signal.emit({"classes": [], "counts": []}, [])
            return

        # Gather classes
        class_list = sorted(
            d for d in os.listdir(self.data_folder)
            if os.path.isdir(os.path.join(self.data_folder, d))
        )
        if not class_list:
            self.progress_signal.emit("No subfolders found. Possibly empty dataset.")
            self.done_signal.emit({"classes": [], "counts": []}, [])
            return

        # For chart
        counts = []
        # For gallery
        class_info_list = []

        for c in class_list:
            sub_dir = os.path.join(self.data_folder, c)
            file_list = []
            for ext in ("*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg"):
                file_list.extend(
                    glob.glob(os.path.join(sub_dir, "**", ext), recursive=True)
                )
            counts.append(len(file_list))

            if file_list:
                sample_files = file_list[:self.max_images]
            else:
                sample_files = []

            class_info_list.append((c, sample_files))

            msg = f"Class '{c}': {len(file_list)} images found."
            self.progress_signal.emit(msg)

        stats_dict = {"classes": class_list, "counts": counts}
        self.done_signal.emit(stats_dict, class_info_list)


class Plugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.plugin_name = "Data Exploration"

        self.widget_main = None
        self.explore_data_dir_edit = None
        self.spin_max_images = None

        self.chart_label = None
        self.gallery_area = None

        # For logging and progress
        self.text_log = None
        self.progress_bar = None

        self.explore_thread = None

    def create_tab(self) -> QWidget:
        """
        Build and return the QWidget that implements the data exploration UI.
        """
        self.widget_main = QWidget()
        layout = QVBoxLayout(self.widget_main)

        # Folder selection row
        h_folder = QHBoxLayout()
        self.explore_data_dir_edit = QLineEdit()
        btn_browse_exp = QPushButton("Browse Data Folder...")
        btn_browse_exp.clicked.connect(self.browse_explore_data_folder)
        h_folder.addWidget(QLabel("Data Folder:"))
        h_folder.addWidget(self.explore_data_dir_edit)
        h_folder.addWidget(btn_browse_exp)
        layout.addLayout(h_folder)

        # spin box for max images
        h_spin = QHBoxLayout()
        self.spin_max_images = QSpinBox()
        self.spin_max_images.setRange(1, 100)
        self.spin_max_images.setValue(3)
        h_spin.addWidget(QLabel("Max Images/Class:"))
        h_spin.addWidget(self.spin_max_images)
        layout.addLayout(h_spin)

        # Analyze dataset button
        btn_run_explore = QPushButton("Analyze Dataset")
        btn_run_explore.clicked.connect(self.analyze_dataset)
        layout.addWidget(btn_run_explore)

        # Chart display
        self.chart_label = QLabel()
        self.chart_label.setFixedSize(600, 400)
        layout.addWidget(self.chart_label, alignment=Qt.AlignCenter)

        # Gallery scroll area
        self.gallery_area = QScrollArea()
        self.gallery_area.setWidgetResizable(True)
        layout.addWidget(self.gallery_area)

        # Logging + Progress
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        layout.addWidget(self.text_log)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        return self.widget_main

    def browse_explore_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Data Folder")
        if folder:
            self.explore_data_dir_edit.setText(folder)

    def analyze_dataset(self):
        folder = self.explore_data_dir_edit.text().strip()
        if not os.path.isdir(folder):
            QMessageBox.warning(self.widget_main, "Error", "Invalid folder.")
            return

        self.progress_bar.setRange(0, 0)  # indefinite
        self.progress_bar.setVisible(True)
        self.text_log.clear()

        max_imgs = self.spin_max_images.value()

        self.explore_thread = DataExplorationThread(folder, max_imgs)
        self.explore_thread.progress_signal.connect(self.on_thread_log)
        self.explore_thread.done_signal.connect(self.on_analysis_done)
        self.explore_thread.start()

    def on_thread_log(self, msg: str):
        self.text_log.append(msg)
        self.text_log.ensureCursorVisible()

    def on_analysis_done(self, stats_dict: dict, class_info_list: list):
        self.progress_bar.setVisible(False)
        # stats_dict = {"classes": [...], "counts": [...]}
        classes = stats_dict["classes"]
        counts = stats_dict["counts"]

        if not classes:
            # No classes => Clear chart + gallery
            self.chart_label.clear()
            self.gallery_area.takeWidget()
            QMessageBox.information(self.widget_main, "Done", "No classes found in folder.")
            return

        # Plot the distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(classes, counts, color='blue')
        ax.set_title("Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Images")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        chart_img = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(chart_img)
        self.chart_label.setPixmap(
            pixmap.scaled(self.chart_label.width(), self.chart_label.height(), Qt.KeepAspectRatio)
        )

        # Build a gallery widget
        gallery_widget = QWidget()
        g_layout = QVBoxLayout(gallery_widget)

        for (class_name, sample_files) in class_info_list:
            label_class = QLabel(f"Class: {class_name} ({len(sample_files)} sample(s) shown)")
            g_layout.addWidget(label_class)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            g_layout.addWidget(row_widget)

            if not sample_files:
                # If no images in that class
                label_none = QLabel("No images found.")
                row_layout.addWidget(label_none)
            else:
                for sf in sample_files:
                    img_ = cv2.imread(sf)
                    if img_ is not None:
                        pm_ = convert_to_qpixmap(img_)
                        label_img = QLabel()
                        label_img.setPixmap(pm_.scaled(100, 100, Qt.KeepAspectRatio))
                        row_layout.addWidget(label_img)

        self.gallery_area.setWidget(gallery_widget)

        QMessageBox.information(self.widget_main, "Analysis Complete", "Dataset analysis finished.")

