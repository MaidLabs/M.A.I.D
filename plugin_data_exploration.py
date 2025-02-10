# plugin_data_exploration.py

import os
import glob
import cv2
import hashlib
import random
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io import BytesIO
from typing import List, Dict, Any

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal
)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QScrollArea, QMessageBox, QSpinBox,
    QTextEdit, QProgressBar, QComboBox, QCheckBox, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage
from base_plugin import BasePlugin  # ensure this exists in your project


########################
# Utility Functions
########################
def md5_hash_file(filepath: str, chunk_size: int = 8192) -> str:
    """
    Return the MD5 hash of a file by reading in chunks (to avoid loading
    the entire file into memory at once).
    """
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def convert_to_qpixmap(numpy_image: np.ndarray) -> QPixmap:
    """
    Safely convert a BGR or RGB numpy image (uint8) to QPixmap.
    """
    array_copied = np.ascontiguousarray(numpy_image)
    height, width, channels = array_copied.shape
    bytes_per_line = channels * width
    q_image = QImage(array_copied.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_image.copy())


########################
# Threads
########################
class DataExplorationThread(QThread):
    """
    Thread to:
      - Scan the dataset folder.
      - Gather class info: image counts, average dims, disk usage.
      - Detect duplicates via MD5 hashes.
      - Possibly random sample up to N images per class for display.
      - Provide incremental progress signals, final stats.
      - Allows user to request_stop().
    """
    progress_signal = pyqtSignal(str, int)  # (message, progress_value)
    done_signal = pyqtSignal(dict, list)    # (stats_dict, class_info_list)

    def __init__(
        self,
        data_folder: str,
        max_images_per_class: int = 3,
        recursive: bool = True,
        file_types: str = "*.tiff,*.tif,*.png,*.jpg,*.jpeg",
        min_width: int = 0,
        min_height: int = 0,
        random_sample: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.data_folder = data_folder
        self.max_images = max_images_per_class
        self.recursive = recursive
        self.file_types = [ft.strip() for ft in file_types.split(",") if ft.strip()]
        self.min_width = min_width
        self.min_height = min_height
        self.random_sample = random_sample

        self._stop_requested = False

    def request_stop(self) -> None:
        """
        Ask this thread to stop processing as soon as possible.
        """
        self._stop_requested = True

    def run(self) -> None:
        """
        Scans the dataset folder to collect:
         - class-wise info (count, dims, disk usage, etc.)
         - duplicates
         - aggregated stats
        Emits (stats_dict, class_info_list) upon completion or early stop.
        """
        import os

        if not os.path.isdir(self.data_folder):
            self.done_signal.emit({"classes": [], "counts": []}, [])
            return

        # Gather subdirectories = classes
        class_list = sorted(
            d for d in os.listdir(self.data_folder)
            if os.path.isdir(os.path.join(self.data_folder, d))
        )
        if not class_list:
            self.progress_signal.emit("No subfolders found. Possibly empty dataset.", 0)
            self.done_signal.emit({"classes": [], "counts": []}, [])
            return

        # Gather all images in one pass
        all_image_paths = []
        for c in class_list:
            sub_dir = os.path.join(self.data_folder, c)
            for ext in self.file_types:
                pattern = os.path.join(sub_dir, "**", ext) if self.recursive else os.path.join(sub_dir, ext)
                all_image_paths.extend(glob.glob(pattern, recursive=self.recursive))

        total_images = len(all_image_paths)
        if total_images == 0:
            self.progress_signal.emit("No images found under given folder(s).", 0)
            self.done_signal.emit({"classes": [], "counts": []}, [])
            return

        self.progress_signal.emit(f"Total images found: {total_images}. Analyzing...", 0)

        # We track duplicates in a structure: hash -> {"first": <path>, "duplicates": []}
        duplicate_map: Dict[str, Dict[str, List[str]]] = {}

        processed_count = 0

        class_info_list = []
        overall_counts = []
        global_width_sum = 0
        global_height_sum = 0
        global_image_count = 0
        global_disk_usage = 0

        # Process each class
        for c in class_list:
            if self._stop_requested:
                # Early stop if requested
                break

            sub_dir = os.path.join(self.data_folder, c)
            file_list = []
            for ext in self.file_types:
                pattern = os.path.join(sub_dir, "**", ext) if self.recursive else os.path.join(sub_dir, ext)
                file_list.extend(glob.glob(pattern, recursive=self.recursive))

            valid_images = []
            corrupted_count = 0
            total_width = 0
            total_height = 0
            disk_usage = 0

            for fpath in file_list:
                if self._stop_requested:
                    break

                processed_count += 1
                progress_percent = int((processed_count / total_images) * 100)
                self.progress_signal.emit(f"Processing: {fpath}", progress_percent)

                try:
                    size_bytes = os.path.getsize(fpath)
                    disk_usage += size_bytes
                except Exception:
                    corrupted_count += 1
                    continue

                # Detect duplicates by MD5 hashing
                try:
                    file_hash = md5_hash_file(fpath)
                except Exception:
                    corrupted_count += 1
                    continue

                if file_hash not in duplicate_map:
                    duplicate_map[file_hash] = {"first": fpath, "duplicates": []}
                else:
                    duplicate_map[file_hash]["duplicates"].append(fpath)

                img_ = cv2.imread(fpath)
                if img_ is None:
                    corrupted_count += 1
                    continue

                h, w = img_.shape[:2]
                if w < self.min_width or h < self.min_height:
                    continue

                valid_images.append(fpath)
                total_width += w
                total_height += h

            # Summarize stats for this class
            count_images = len(valid_images)
            if count_images > 0:
                avg_w = total_width / count_images
                avg_h = total_height / count_images
            else:
                avg_w = 0
                avg_h = 0

            if self.random_sample:
                random.shuffle(valid_images)
            sample_files = valid_images[:self.max_images]

            class_info = {
                "class_name": c,
                "count": count_images,
                "avg_width": avg_w,
                "avg_height": avg_h,
                "disk_usage": disk_usage,
                "corrupted_count": corrupted_count,
                "sample_files": sample_files,
                "valid_files": valid_images  # store all for potential CSV or splitting
            }
            class_info_list.append(class_info)
            overall_counts.append(count_images)

            global_image_count += count_images
            global_width_sum += total_width
            global_height_sum += total_height
            global_disk_usage += disk_usage

            self.progress_signal.emit(
                f"Class '{c}': {count_images} valid images, {corrupted_count} corrupted/skipped.",
                min(progress_percent, 100)
            )

        # Count duplicates (beyond first occurrences)
        duplicate_count = 0
        for hval, paths_dict in duplicate_map.items():
            if len(paths_dict["duplicates"]) > 0:
                duplicate_count += len(paths_dict["duplicates"])

        if global_image_count > 0:
            global_avg_width = global_width_sum / global_image_count
            global_avg_height = global_height_sum / global_image_count
        else:
            global_avg_width = 0
            global_avg_height = 0

        stats_dict = {
            "classes": [ci["class_name"] for ci in class_info_list],
            "counts": overall_counts,
            "duplicate_count": duplicate_count,
            "total_images": global_image_count,
            "global_avg_width": global_avg_width,
            "global_avg_height": global_avg_height,
            "global_disk_usage": global_disk_usage
        }

        if self._stop_requested:
            self.progress_signal.emit("Analysis canceled by user.", 100)
            self.done_signal.emit({"classes": [], "counts": []}, [])
        else:
            self.done_signal.emit(stats_dict, class_info_list)


class DatasetSplitThread(QThread):
    """
    Thread to split a dataset (already validated by DataExplorationThread)
    into train/val/test subfolders with a given ratio.
    """
    progress_signal = pyqtSignal(str, int)  # (message, progress_value)
    done_signal = pyqtSignal(bool)          # (finished_successfully)

    def __init__(
        self,
        class_info_list: List[dict],
        data_folder: str,
        train_pct: int,
        val_pct: int,
        test_pct: int,
        output_folder: str,
        recursive: bool,
        min_width: int,
        min_height: int,
        parent=None
    ):
        super().__init__(parent)
        self.class_info_list = class_info_list
        self.data_folder = data_folder
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.output_folder = output_folder
        self.recursive = recursive
        self.min_width = min_width
        self.min_height = min_height

        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        train_dir = os.path.join(self.output_folder, "train")
        val_dir = os.path.join(self.output_folder, "val")
        test_dir = os.path.join(self.output_folder, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        total_images = sum(len(ci["valid_files"]) for ci in self.class_info_list)
        copied_count = 0

        for ci in self.class_info_list:
            if self._stop_requested:
                break

            class_name = ci["class_name"]
            valid_images = ci["valid_files"]

            random.shuffle(valid_images)
            total_count = len(valid_images)
            if total_count == 0:
                continue

            train_count = int((self.train_pct / 100.0) * total_count)
            val_count = int((self.val_pct / 100.0) * total_count)
            test_count = total_count - train_count - val_count

            train_files = valid_images[:train_count]
            val_files = valid_images[train_count:train_count + val_count]
            test_files = valid_images[train_count + val_count:]

            class_train_dir = os.path.join(train_dir, class_name)
            class_val_dir = os.path.join(val_dir, class_name)
            class_test_dir = os.path.join(test_dir, class_name)

            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_val_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            for f in train_files:
                if self._stop_requested:
                    break
                shutil.copy2(f, class_train_dir)
                copied_count += 1
                self.progress_signal.emit(
                    f"Copying to train: {os.path.basename(f)}",
                    int((copied_count / total_images) * 100)
                )
            for f in val_files:
                if self._stop_requested:
                    break
                shutil.copy2(f, class_val_dir)
                copied_count += 1
                self.progress_signal.emit(
                    f"Copying to val: {os.path.basename(f)}",
                    int((copied_count / total_images) * 100)
                )
            for f in test_files:
                if self._stop_requested:
                    break
                shutil.copy2(f, class_test_dir)
                copied_count += 1
                self.progress_signal.emit(
                    f"Copying to test: {os.path.basename(f)}",
                    int((copied_count / total_images) * 100)
                )

        if self._stop_requested:
            self.progress_signal.emit("Splitting canceled by user.", 100)
            self.done_signal.emit(False)
        else:
            self.done_signal.emit(True)


class Plugin(BasePlugin):
    """
    Data Exploration Plugin with a new "Export Ground Truth CSV" option
    that creates a CSV: 'image_path,label' for all valid_files
    in each class's subfolder.

    Modified so that the 'image_path' column is the *base filename* only,
    matching what the inference plugin checks with os.path.basename(...).
    """

    def __init__(self):
        super().__init__()
        self.plugin_name = "Data Exploration"

        self.widget_main = None

        # Folder + scanning
        self.explore_data_dir_edit = None
        self.checkbox_recursive = None
        self.file_types_edit = None
        self.spin_min_width = None
        self.spin_min_height = None

        # Visualization
        self.chart_type_combo = None
        self.sort_combo = None
        self.spin_max_images = None
        self.checkbox_random_sample = None
        self.spin_preview_size = None
        self.checkbox_log_scale = None

        self.chart_label = None
        self.gallery_area = None

        # Logging + progress
        self.text_log = None
        self.progress_bar = None

        # Buttons
        self.btn_run_explore = None
        self.btn_stop_explore = None
        self.btn_export_csv = None
        self.btn_export_gt_csv = None
        self.btn_split_dataset = None
        self.btn_stop_split = None

        # Splitting controls
        self.split_train_spin = None
        self.split_val_spin = None
        self.split_test_spin = None

        # Thread references
        self.explore_thread = None
        self.split_thread = None

        # We'll store results here so we can export/split
        self.stats_dict: Dict[str, Any] = {}
        self.class_info_list: List[dict] = []

    def create_tab(self) -> QWidget:
        self.widget_main = QWidget()
        main_layout = QVBoxLayout(self.widget_main)

        # 1. FOLDER SELECTION & OPTIONS
        folder_layout = QHBoxLayout()
        self.explore_data_dir_edit = QLineEdit()
        btn_browse_exp = QPushButton("Browse Data Folder...")
        btn_browse_exp.clicked.connect(self.browse_explore_data_folder)
        folder_layout.addWidget(QLabel("Data Folder:"))
        folder_layout.addWidget(self.explore_data_dir_edit)
        folder_layout.addWidget(btn_browse_exp)
        main_layout.addLayout(folder_layout)

        # 2. SCANNING OPTIONS
        options_grid = QGridLayout()
        row = 0

        self.checkbox_recursive = QCheckBox("Recursive Search")
        self.checkbox_recursive.setChecked(True)
        options_grid.addWidget(self.checkbox_recursive, row, 0)
        row += 1

        options_grid.addWidget(QLabel("File Types (comma-separated):"), row, 0)
        self.file_types_edit = QLineEdit("*.tiff, *.tif, *.png, *.jpg, *.jpeg")
        options_grid.addWidget(self.file_types_edit, row, 1)
        row += 1

        options_grid.addWidget(QLabel("Min Width:"), row, 0)
        self.spin_min_width = QSpinBox()
        self.spin_min_width.setRange(0, 10000)
        self.spin_min_width.setValue(0)
        options_grid.addWidget(self.spin_min_width, row, 1)
        row += 1

        options_grid.addWidget(QLabel("Min Height:"), row, 0)
        self.spin_min_height = QSpinBox()
        self.spin_min_height.setRange(0, 10000)
        self.spin_min_height.setValue(0)
        options_grid.addWidget(self.spin_min_height, row, 1)
        row += 1

        options_grid.addWidget(QLabel("Max Images/Class:"), row, 0)
        self.spin_max_images = QSpinBox()
        self.spin_max_images.setRange(1, 100)
        self.spin_max_images.setValue(3)
        options_grid.addWidget(self.spin_max_images, row, 1)
        row += 1

        self.checkbox_random_sample = QCheckBox("Random Sample per Class")
        self.checkbox_random_sample.setChecked(False)
        options_grid.addWidget(self.checkbox_random_sample, row, 0, 1, 2)
        row += 1

        main_layout.addLayout(options_grid)

        # 3. VISUALIZATION PREFERENCES
        vis_layout = QHBoxLayout()
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Bar Chart", "Pie Chart", "Horizontal Bar"])
        vis_layout.addWidget(QLabel("Chart Type:"))
        vis_layout.addWidget(self.chart_type_combo)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["By Name (A-Z)", "By Count (Descending)"])
        vis_layout.addWidget(QLabel("Sort Classes:"))
        vis_layout.addWidget(self.sort_combo)

        self.checkbox_log_scale = QCheckBox("Use Log Scale (Bar/Horizontal)")
        self.checkbox_log_scale.setChecked(False)
        vis_layout.addWidget(self.checkbox_log_scale)
        main_layout.addLayout(vis_layout)

        # Preview size
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Preview Image Size:"))
        self.spin_preview_size = QSpinBox()
        self.spin_preview_size.setRange(50, 500)
        self.spin_preview_size.setValue(100)
        preview_layout.addWidget(self.spin_preview_size)
        main_layout.addLayout(preview_layout)

        # 4. ACTION BUTTONS
        actions_layout = QHBoxLayout()
        self.btn_run_explore = QPushButton("Analyze Dataset")
        self.btn_run_explore.clicked.connect(self.analyze_dataset)
        actions_layout.addWidget(self.btn_run_explore)

        self.btn_stop_explore = QPushButton("Stop Analysis")
        self.btn_stop_explore.setEnabled(False)
        self.btn_stop_explore.clicked.connect(self.stop_analysis)
        actions_layout.addWidget(self.btn_stop_explore)

        self.btn_export_csv = QPushButton("Export Stats CSV")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self.export_csv)
        actions_layout.addWidget(self.btn_export_csv)

        # Export Ground Truth CSV
        self.btn_export_gt_csv = QPushButton("Export Ground Truth CSV")
        self.btn_export_gt_csv.setEnabled(False)
        self.btn_export_gt_csv.clicked.connect(self.export_ground_truth_csv)
        actions_layout.addWidget(self.btn_export_gt_csv)

        main_layout.addLayout(actions_layout)

        # Data splitting controls
        split_layout = QHBoxLayout()
        self.split_train_spin = QSpinBox()
        self.split_train_spin.setRange(0, 100)
        self.split_train_spin.setValue(80)
        self.split_val_spin = QSpinBox()
        self.split_val_spin.setRange(0, 100)
        self.split_val_spin.setValue(10)
        self.split_test_spin = QSpinBox()
        self.split_test_spin.setRange(0, 100)
        self.split_test_spin.setValue(10)

        split_layout.addWidget(QLabel("Train %:"))
        split_layout.addWidget(self.split_train_spin)
        split_layout.addWidget(QLabel("Val %:"))
        split_layout.addWidget(self.split_val_spin)
        split_layout.addWidget(QLabel("Test %:"))
        split_layout.addWidget(self.split_test_spin)

        self.btn_split_dataset = QPushButton("Split Dataset")
        self.btn_split_dataset.setEnabled(False)
        self.btn_split_dataset.clicked.connect(self.split_dataset)
        split_layout.addWidget(self.btn_split_dataset)

        self.btn_stop_split = QPushButton("Stop Splitting")
        self.btn_stop_split.setEnabled(False)
        self.btn_stop_split.clicked.connect(self.stop_splitting)
        split_layout.addWidget(self.btn_stop_split)

        main_layout.addLayout(split_layout)

        # 5. CHART DISPLAY
        self.chart_label = QLabel()
        self.chart_label.setFixedSize(600, 400)
        main_layout.addWidget(self.chart_label, alignment=Qt.AlignCenter)

        # 6. GALLERY SCROLL AREA
        self.gallery_area = QScrollArea()
        self.gallery_area.setWidgetResizable(True)
        main_layout.addWidget(self.gallery_area)

        # 7. LOGGING + PROGRESS
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        main_layout.addWidget(self.text_log)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        return self.widget_main

    ########################################################################
    # FOLDER BROWSE
    ########################################################################
    def browse_explore_data_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Data Folder")
        if folder:
            self.explore_data_dir_edit.setText(folder)

    ########################################################################
    # ANALYZE DATASET
    ########################################################################
    def analyze_dataset(self) -> None:
        folder = self.explore_data_dir_edit.text().strip()
        if not os.path.isdir(folder):
            QMessageBox.warning(self.widget_main, "Error", "Invalid folder.")
            return

        self.stats_dict = {}
        self.class_info_list = []
        self.text_log.clear()
        self.chart_label.clear()
        self.gallery_area.takeWidget()

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.btn_run_explore.setEnabled(False)
        self.btn_stop_explore.setEnabled(True)

        max_imgs = self.spin_max_images.value()
        recursive = self.checkbox_recursive.isChecked()
        file_types = self.file_types_edit.text()
        min_w = self.spin_min_width.value()
        min_h = self.spin_min_height.value()
        random_sample = self.checkbox_random_sample.isChecked()

        self.explore_thread = DataExplorationThread(
            data_folder=folder,
            max_images_per_class=max_imgs,
            recursive=recursive,
            file_types=file_types,
            min_width=min_w,
            min_height=min_h,
            random_sample=random_sample
        )
        self.explore_thread.progress_signal.connect(self.on_thread_log)
        self.explore_thread.done_signal.connect(self.on_analysis_done)
        self.explore_thread.start()

    def stop_analysis(self) -> None:
        if self.explore_thread is not None:
            self.explore_thread.request_stop()
            self.text_log.append("Stop requested for analysis...")

    def on_thread_log(self, msg: str, progress_value: int) -> None:
        self.text_log.append(msg)
        self.text_log.ensureCursorVisible()
        self.progress_bar.setValue(progress_value)

    def on_analysis_done(self, stats_dict: dict, class_info_list: list) -> None:
        self.progress_bar.setVisible(False)
        self.btn_run_explore.setEnabled(True)
        self.btn_stop_explore.setEnabled(False)

        self.stats_dict = stats_dict
        self.class_info_list = class_info_list

        classes = stats_dict.get("classes", [])
        counts = stats_dict.get("counts", [])

        if not classes:
            self.chart_label.clear()
            self.gallery_area.takeWidget()
            QMessageBox.information(self.widget_main, "Done", "No classes found or no images after filtering.")
            return

        # Sort if requested
        sort_mode = self.sort_combo.currentText()
        combined = list(zip(classes, counts, class_info_list))
        if sort_mode == "By Count (Descending)":
            combined.sort(key=lambda x: x[1], reverse=True)
        else:
            combined.sort(key=lambda x: x[0].lower())

        classes, counts, sorted_class_info_list = zip(*combined)

        # Plot the distribution
        from io import BytesIO
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))

        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

        chart_type = self.chart_type_combo.currentText()
        if chart_type == "Bar Chart":
            ax.bar(classes, counts, color=colors)
            ax.set_xlabel("Class")
            ax.set_ylabel("Number of Images")
            if self.checkbox_log_scale.isChecked():
                ax.set_yscale("log")
        elif chart_type == "Pie Chart":
            ax.pie(counts, labels=classes, autopct="%1.1f%%", colors=colors)
        else:  # "Horizontal Bar"
            ax.barh(classes, counts, color=colors)
            ax.set_ylabel("Class")
            ax.set_xlabel("Number of Images")
            ax.invert_yaxis()
            if self.checkbox_log_scale.isChecked():
                ax.set_xscale("log")

        ax.set_title("Class Distribution")
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

        # Build the gallery
        gallery_widget = QWidget()
        g_layout = QVBoxLayout(gallery_widget)
        preview_size = self.spin_preview_size.value()

        for ci in sorted_class_info_list:
            class_name = ci["class_name"]
            sample_files = ci["sample_files"]
            count_images = ci["count"]
            corrupted_count = ci["corrupted_count"]
            avg_w = ci["avg_width"]
            avg_h = ci["avg_height"]
            disk_usage = ci["disk_usage"]

            label_class = QLabel(
                f"Class: {class_name}\n"
                f"  Total valid images: {count_images}, Corrupted: {corrupted_count}\n"
                f"  Avg WxH: {avg_w:.1f} x {avg_h:.1f}, Disk: {disk_usage/1024:.2f} KB\n"
                f"  Showing {len(sample_files)} sample(s)"
            )
            label_class.setStyleSheet("font-weight: bold; margin-top: 10px;")
            g_layout.addWidget(label_class)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            g_layout.addWidget(row_widget)

            if not sample_files:
                label_none = QLabel("No images found or selected.")
                row_layout.addWidget(label_none)
            else:
                for sf in sample_files:
                    img_ = cv2.imread(sf)
                    if img_ is not None:
                        pm_ = convert_to_qpixmap(img_)
                        label_img = QLabel()
                        label_img.setPixmap(pm_.scaled(preview_size, preview_size, Qt.KeepAspectRatio))
                        row_layout.addWidget(label_img)

        self.gallery_area.setWidget(gallery_widget)

        summary_msg = (
            f"Analysis Complete.\n"
            f" - Total classes: {len(classes)}\n"
            f" - Total valid images: {stats_dict.get('total_images', 0)}\n"
            f" - Duplicate images (beyond first occurrences): {stats_dict.get('duplicate_count', 0)}\n"
            f" - Global Avg WxH: {stats_dict.get('global_avg_width',0):.1f} x {stats_dict.get('global_avg_height',0):.1f}\n"
            f" - Global Disk Usage: {stats_dict.get('global_disk_usage',0)/1024:.2f} KB\n"
        )
        self.text_log.append(summary_msg)

        self.btn_export_csv.setEnabled(True)
        self.btn_export_gt_csv.setEnabled(True)
        self.btn_split_dataset.setEnabled(True)

        QMessageBox.information(self.widget_main, "Analysis Complete", "Dataset analysis finished.")

    ########################################################################
    # EXPORT STATS CSV
    ########################################################################
    def export_csv(self) -> None:
        if not self.class_info_list:
            QMessageBox.information(self.widget_main, "No Data", "No analysis data to export. Run analysis first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self.widget_main, "Export CSV", "", "CSV Files (*.csv)")
        if not save_path:
            return

        if os.path.exists(save_path):
            confirm = QMessageBox.question(
                self.widget_main,
                "Overwrite File?",
                f"The file '{save_path}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if confirm != QMessageBox.Yes:
                return

        import csv
        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Class", "Count", "Corrupted", "AvgWidth", "AvgHeight", "DiskUsage(KB)"])
                for ci in self.class_info_list:
                    writer.writerow([
                        ci["class_name"],
                        ci["count"],
                        ci["corrupted_count"],
                        f"{ci['avg_width']:.1f}",
                        f"{ci['avg_height']:.1f}",
                        f"{ci['disk_usage']/1024:.2f}"
                    ])

                writer.writerow([])
                writer.writerow(["TOTAL VALID IMAGES", self.stats_dict.get("total_images", 0)])
                writer.writerow(["DUPLICATE IMAGES", self.stats_dict.get("duplicate_count", 0)])
                writer.writerow([
                    "GLOBAL AVG WxH",
                    f"{self.stats_dict.get('global_avg_width',0):.1f} x {self.stats_dict.get('global_avg_height',0):.1f}"
                ])
                writer.writerow([
                    "GLOBAL DISK USAGE (KB)",
                    f"{self.stats_dict.get('global_disk_usage',0)/1024:.2f}"
                ])

            QMessageBox.information(self.widget_main, "Export CSV", f"CSV exported to {save_path}")
        except Exception as e:
            QMessageBox.warning(self.widget_main, "Error", f"Failed to export CSV:\n{str(e)}")

    ########################################################################
    # EXPORT GROUND TRUTH CSV (new feature)
    ########################################################################
    def export_ground_truth_csv(self) -> None:
        """
        Exports a "ground truth" CSV where each line is:
            base_filename,label
        rather than the full path, so the inference plugin can match via os.path.basename(...).
        """
        if not self.class_info_list:
            QMessageBox.information(
                self.widget_main,
                "No Data",
                "No analysis data to export. Run analysis first."
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self.widget_main,
            "Export Ground Truth CSV",
            "",
            "CSV Files (*.csv)"
        )
        if not save_path:
            return

        # Confirm overwriting if file exists
        if os.path.exists(save_path):
            confirm = QMessageBox.question(
                self.widget_main,
                "Overwrite File?",
                f"The file '{save_path}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if confirm != QMessageBox.Yes:
                return

        try:
            import csv
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label"])  # header row (optional)

                # For each class_info, we have "class_name" and "valid_files"
                for ci in self.class_info_list:
                    class_name = ci["class_name"]
                    for vf in ci["valid_files"]:
                        # Only write the BASE filename:
                        base_name = os.path.basename(vf)
                        writer.writerow([base_name, class_name])

            QMessageBox.information(
                self.widget_main,
                "Export GT CSV",
                f"Ground truth CSV (with BASE filenames) exported to:\n{save_path}"
            )

        except Exception as e:
            QMessageBox.warning(
                self.widget_main,
                "Error",
                f"Failed to export GT CSV:\n{str(e)}"
            )

    ########################################################################
    # SPLIT DATASET
    ########################################################################
    def split_dataset(self) -> None:
        if not self.class_info_list:
            QMessageBox.information(self.widget_main, "No Data", "No analysis data to split. Run analysis first.")
            return

        train_pct = self.split_train_spin.value()
        val_pct = self.split_val_spin.value()
        test_pct = self.split_test_spin.value()
        if (train_pct + val_pct + test_pct) != 100:
            QMessageBox.warning(self.widget_main, "Error", "Train/Val/Test percentages must sum to 100.")
            return

        output_folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Output Folder for Splits")
        if not output_folder:
            return

        confirm = QMessageBox.question(
            self.widget_main,
            "Confirm",
            f"Split dataset into {train_pct}% train, {val_pct}% val, {test_pct}% test?\n\n"
            f"Output to: {output_folder}",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        self.btn_split_dataset.setEnabled(False)
        self.btn_stop_split.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.split_thread = DatasetSplitThread(
            class_info_list=self.class_info_list,
            data_folder=self.explore_data_dir_edit.text(),
            train_pct=train_pct,
            val_pct=val_pct,
            test_pct=test_pct,
            output_folder=output_folder,
            recursive=self.checkbox_recursive.isChecked(),
            min_width=self.spin_min_width.value(),
            min_height=self.spin_min_height.value()
        )
        self.split_thread.progress_signal.connect(self.on_split_progress)
        self.split_thread.done_signal.connect(self.on_split_done)
        self.split_thread.start()

    def stop_splitting(self) -> None:
        if self.split_thread is not None:
            self.split_thread.request_stop()
            self.text_log.append("Stop requested for splitting...")

    def on_split_progress(self, msg: str, progress_value: int) -> None:
        self.text_log.append(msg)
        self.text_log.ensureCursorVisible()
        self.progress_bar.setValue(progress_value)

    def on_split_done(self, success: bool) -> None:
        self.btn_split_dataset.setEnabled(True)
        self.btn_stop_split.setEnabled(False)
        self.progress_bar.setVisible(False)

        if success:
            QMessageBox.information(self.widget_main, "Split Complete", "Dataset splitting finished.")
            self.text_log.append("Dataset splitting complete.")
        else:
            QMessageBox.warning(self.widget_main, "Split Canceled", "Dataset splitting was canceled.")
            self.text_log.append("Dataset splitting canceled.")
