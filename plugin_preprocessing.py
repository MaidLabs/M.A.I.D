# plugins/plugin_preprocessing.py

import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QCheckBox, QFormLayout,
    QSpinBox, QFileDialog, QComboBox, QMessageBox, QScrollArea, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QGroupBox, QTextEdit, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from base_plugin import BasePlugin

################################################
#       HELPER FUNCTIONS (unchanged below)     #
################################################

def convert_to_qpixmap(numpy_image: np.ndarray) -> QPixmap:
    """
    Convert a BGR (OpenCV) or RGB numpy array to QPixmap safely.
    """
    array_copied = np.ascontiguousarray(numpy_image)
    h, w, channels = array_copied.shape
    bytes_per_line = channels * w
    q_image = QImage(array_copied.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_image.copy())

def apply_hist_equal_channelwise(image_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image_bgr)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    return cv2.merge([b, g, r])

def apply_clahe_channelwise(image_bgr: np.ndarray, clip=2.0, tile=8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    b, g, r = cv2.split(image_bgr)
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    return cv2.merge([b, g, r])

def threshold_channelwise(image_bgr: np.ndarray, thr_val: int = 128) -> np.ndarray:
    b, g, r = cv2.split(image_bgr)
    _, bth = cv2.threshold(b, thr_val, 255, cv2.THRESH_BINARY)
    _, gth = cv2.threshold(g, thr_val, 255, cv2.THRESH_BINARY)
    _, rth = cv2.threshold(r, thr_val, 255, cv2.THRESH_BINARY)
    return cv2.merge([bth, gth, rth])

def adaptive_threshold_channelwise(image_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image_bgr)
    b_ad = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    g_ad = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    r_ad = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.merge([b_ad, g_ad, r_ad])

def otsu_threshold_channelwise(image_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image_bgr)
    _, bth = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, gth = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, rth = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.merge([bth, gth, rth])

def morphological_op_channelwise(image_bgr: np.ndarray, op: str, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    b, g, r = cv2.split(image_bgr)

    def do_op(chan):
        if op == "erode":
            return cv2.erode(chan, kernel, iterations=1)
        elif op == "dilate":
            return cv2.dilate(chan, kernel, iterations=1)
        elif op == "opening":
            return cv2.morphologyEx(chan, cv2.MORPH_OPEN, kernel)
        elif op == "closing":
            return cv2.morphologyEx(chan, cv2.MORPH_CLOSE, kernel)
        elif op == "tophat":
            return cv2.morphologyEx(chan, cv2.MORPH_TOPHAT, kernel)
        elif op == "blackhat":
            return cv2.morphologyEx(chan, cv2.MORPH_BLACKHAT, kernel)
        else:
            return chan

    b = do_op(b)
    g = do_op(g)
    r = do_op(r)
    return cv2.merge([b, g, r])

def enhance_red_channel(image_bgr: np.ndarray, bright_factor: float, contrast_factor: float) -> np.ndarray:
    b, g, r = cv2.split(image_bgr)
    r = r.astype(np.float32)
    r *= bright_factor
    r = (r - 128.0) * contrast_factor + 128.0
    r = np.clip(r, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def filter_images_by_channel(in_dir: str, out_dir: str, channel_substring: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if channel_substring.lower() not in file.lower():
                continue
            src_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, in_dir)
            dest_sub_dir = os.path.join(out_dir, relative_path)
            os.makedirs(dest_sub_dir, exist_ok=True)
            dest_path = os.path.join(dest_sub_dir, file)
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dest_path, img)

def overlay_two_channels(
    img_ch01_path: str,
    img_ch02_path: str,
    hist_equal: bool = False,
    clahe: bool = False,
    normalize_channels: bool = False
) -> np.ndarray:
    ch01_gray = cv2.imread(img_ch01_path, cv2.IMREAD_GRAYSCALE)
    ch02_gray = cv2.imread(img_ch02_path, cv2.IMREAD_GRAYSCALE)
    if ch01_gray is None or ch02_gray is None:
        return None

    if normalize_channels:
        ch01_gray = normalize_channel_to_255(ch01_gray)
        ch02_gray = normalize_channel_to_255(ch02_gray)

    if hist_equal:
        ch01_gray = cv2.equalizeHist(ch01_gray)
        ch02_gray = cv2.equalizeHist(ch02_gray)

    if clahe:
        clahe_func = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ch01_gray = clahe_func.apply(ch01_gray)
        ch02_gray = clahe_func.apply(ch02_gray)

    if ch01_gray.shape != ch02_gray.shape:
        h = min(ch01_gray.shape[0], ch02_gray.shape[0])
        w = min(ch01_gray.shape[1], ch02_gray.shape[1])
        ch01_gray = ch01_gray[:h, :w]
        ch02_gray = ch02_gray[:h, :w]

    overlay_bgr = np.zeros((ch01_gray.shape[0], ch01_gray.shape[1], 3), dtype=np.uint8)
    overlay_bgr[..., 2] = ch01_gray  # Red
    overlay_bgr[..., 1] = ch02_gray  # Green
    return overlay_bgr

def normalize_channel_to_255(gray: np.ndarray) -> np.ndarray:
    mn, mx = gray.min(), gray.max()
    if mx > mn:
        norm = ((gray - mn) / (mx - mn)) * 255.0
        return norm.astype(np.uint8)
    else:
        return gray

###############################################
#   QThreads for Bulk Preprocessing Tasks     #
###############################################
class BulkPreprocessingThread(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str)  # emits output dir or "ERROR"

    def __init__(self, in_dir, out_dir, out_suffix, file_ext, preprocess_params, parent=None):
        super().__init__(parent)
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.out_suffix = out_suffix
        self.file_ext = file_ext
        self.params = preprocess_params

    def run(self):
        try:
            self._process_bulk_images()
        except Exception as e:
            msg = f"BulkPreprocessingThread Error: {e}"
            self.log_signal.emit(msg)
            self.done_signal.emit("ERROR")

    def _process_bulk_images(self):
        # Gather valid images
        valid_exts = (".tiff", ".tif", ".png", ".jpg", ".jpeg")
        all_files = []
        for root, dirs, files in os.walk(self.in_dir):
            for fname in files:
                if any(fname.lower().endswith(ext) for ext in valid_exts):
                    all_files.append(os.path.join(root, fname))

        if not all_files:
            self.log_signal.emit("No images found in input folder.")
            self.done_signal.emit(self.out_dir)
            return

        total = len(all_files)
        self.log_signal.emit(f"Found {total} images. Starting bulk preprocessing...")

        for idx, fpath in enumerate(all_files, start=1):
            img = cv2.imread(fpath)
            if img is None:
                self.log_signal.emit(f"Skipping unreadable file: {fpath}")
                self.progress_signal.emit(idx)
                continue

            proc = self.params["pipeline_func"](img, self.params)
            # Build output path
            relative_path = os.path.relpath(os.path.dirname(fpath), self.in_dir)
            os.makedirs(os.path.join(self.out_dir, relative_path), exist_ok=True)
            base_name = os.path.splitext(os.path.basename(fpath))[0]
            out_name = base_name + self.out_suffix + f".{self.file_ext}"
            out_path = os.path.join(self.out_dir, relative_path, out_name)
            cv2.imwrite(out_path, proc)
            self.progress_signal.emit(idx)

        self.log_signal.emit("Bulk preprocessing completed.")
        self.done_signal.emit(self.out_dir)


class OverlayThread(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str)  # out_dir or ERROR

    def __init__(self, in_dir, out_dir, out_suffix, overlay_params, parent=None):
        super().__init__(parent)
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.out_suffix = out_suffix
        self.params = overlay_params

    def run(self):
        try:
            self._process_overlay()
        except Exception as e:
            msg = f"OverlayThread Error: {e}"
            self.log_signal.emit(msg)
            self.done_signal.emit("ERROR")

    def _process_overlay(self):
        from collections import defaultdict
        valid_exts = (".tiff", ".tif", ".png", ".jpg", ".jpeg")
        # We'll pair ch01 and ch02
        channel_dict = defaultdict(lambda: {})

        # Find all images
        all_files = []
        for root, dirs, files in os.walk(self.in_dir):
            for fname in files:
                lf = fname.lower()
                if any(lf.endswith(ext) for ext in valid_exts):
                    all_files.append(os.path.join(root, fname))

        if not all_files:
            self.log_signal.emit("No images in overlay input.")
            self.done_signal.emit(self.out_dir)
            return

        for f in all_files:
            lf = os.path.basename(f).lower()
            ch01_tag = self.params["ch01_substring"]
            ch02_tag = self.params["ch02_substring"]
            if ch01_tag in lf:
                prefix = lf.split(ch01_tag)[0]
                channel_dict[(os.path.dirname(f), prefix)]["ch01"] = f
            elif ch02_tag in lf:
                prefix = lf.split(ch02_tag)[0]
                channel_dict[(os.path.dirname(f), prefix)]["ch02"] = f

        pairs = list(channel_dict.items())
        total = len(pairs)
        self.log_signal.emit(f"Found {total} ch01-ch02 pairs (or partial pairs). Processing...")

        idx = 0
        for (root, prefix), ch_map in pairs:
            idx += 1
            if "ch01" in ch_map and "ch02" in ch_map:
                path_ch01 = ch_map["ch01"]
                path_ch02 = ch_map["ch02"]
                hist_eq = self.params["hist_equal"]
                clahe_ = self.params["clahe"]
                normalize_ = self.params["normalize_"]

                overlay_bgr = overlay_two_channels(
                    img_ch01_path=path_ch01,
                    img_ch02_path=path_ch02,
                    hist_equal=hist_eq,
                    clahe=clahe_,
                    normalize_channels=normalize_
                )
                if overlay_bgr is None:
                    self.log_signal.emit(f"Skipping pair due to read error: {path_ch01}, {path_ch02}")
                    self.progress_signal.emit(idx)
                    continue

                relative = os.path.relpath(root, self.in_dir)
                out_sub_dir = os.path.join(self.out_dir, relative)
                os.makedirs(out_sub_dir, exist_ok=True)
                out_fname = prefix + self.out_suffix + "_overlay.png"
                out_path = os.path.join(out_sub_dir, out_fname)
                cv2.imwrite(out_path, overlay_bgr)
            else:
                self.log_signal.emit(f"No complete pair for prefix={prefix} in {root}. Skipping.")
            self.progress_signal.emit(idx)

        self.done_signal.emit(self.out_dir)


class FilterThread(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str)

    def __init__(self, in_dir, out_dir, substring, parent=None):
        super().__init__(parent)
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.substring = substring

    def run(self):
        try:
            filter_images_by_channel(self.in_dir, self.out_dir, self.substring)
            self.done_signal.emit(self.out_dir)
        except Exception as e:
            self.log_signal.emit(f"Filter Error: {e}")
            self.done_signal.emit("ERROR")


################################################
#             MAIN PLUGIN CLASS                #
################################################

class Plugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.plugin_name = "Preprocessing"
        self.widget_main = None

        # Single-image references
        self.original_image = None
        self.processed_image = None
        self.current_image_index = 0
        self.current_folder_images = []

        # UI
        self.preprocess_img_path = None
        self.label_original = None
        self.label_processed = None

        # Thresholding
        self.rb_thresh_none = None
        self.rb_thresh_manual = None
        self.rb_thresh_adaptive = None
        self.rb_thresh_otsu = None
        self.thresh_spin = None
        self.btn_group_thresh = None

        # Morphology
        self.morph_combo = None
        self.morph_kernel_spin = None

        # Denoise
        self.denoise_combo = None
        self.denoise_kernel_spin = None

        # Checkboxes
        self.cb_grayscale = None
        self.cb_invert = None
        self.cb_hist_equal = None
        self.cb_clahe = None

        # Channel
        self.channel_combo = None

        # Resize
        self.resize_w_spin = None
        self.resize_h_spin = None

        # Red channel
        self.cb_enhance_red = None
        self.red_brightness_spin = None
        self.red_contrast_spin = None

        # Bulk
        self.bulk_input_dir = None
        self.bulk_output_dir = None
        self.bulk_suffix_edit = None
        self.bulk_format_combo = None
        self.bulk_thread = None

        # Overlay
        self.overlay_input_dir = None
        self.overlay_output_dir = None
        self.overlay_suffix_edit = None
        self.cb_overlay_normalize = None
        self.cb_overlay_hist_equal = None
        self.cb_overlay_clahe = None
        self.overlay_thread = None
        self.overlay_ch01_edit = None
        self.overlay_ch02_edit = None

        # Filter
        self.filter_input_dir = None
        self.filter_output_dir = None
        self.filter_substring_edit = None
        self.filter_thread = None

        # Log + Progress
        self.text_log = None
        self.progress_bar = None

    def create_tab(self) -> QWidget:
        self.widget_main = QWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        scroll_area.setWidget(container)

        main_layout = QVBoxLayout(self.widget_main)
        main_layout.addWidget(scroll_area)

        # ---------- SINGLE IMAGE UI -------------
        single_box = QVBoxLayout()
        single_box.addWidget(QLabel("<b>Single Image Preprocessing</b>"))

        # File row + next/prev buttons
        h_file = QHBoxLayout()
        self.preprocess_img_path = QLineEdit()
        btn_browse = QPushButton("Browse Image...")
        btn_browse.clicked.connect(self.browse_preprocess_image)
        btn_prev = QPushButton("Prev Image")
        btn_prev.clicked.connect(self.prev_image)
        btn_next = QPushButton("Next Image")
        btn_next.clicked.connect(self.next_image)

        h_file.addWidget(self.preprocess_img_path)
        h_file.addWidget(btn_browse)
        h_file.addWidget(btn_prev)
        h_file.addWidget(btn_next)
        single_box.addLayout(h_file)

        # Basic checkboxes
        basic_box = QGroupBox("Basic Options")
        basic_layout = QVBoxLayout(basic_box)
        self.cb_grayscale = QCheckBox("Grayscale")
        self.cb_invert = QCheckBox("Invert")
        basic_layout.addWidget(self.cb_grayscale)
        basic_layout.addWidget(self.cb_invert)

        # Threshold group (radio)
        thresh_group_box = QGroupBox("Threshold")
        tg_layout = QHBoxLayout(thresh_group_box)
        self.rb_thresh_none = QRadioButton("None")
        self.rb_thresh_manual = QRadioButton("Manual")
        self.rb_thresh_adaptive = QRadioButton("Adaptive")
        self.rb_thresh_otsu = QRadioButton("Otsu")
        self.rb_thresh_none.setChecked(True)

        self.btn_group_thresh = QButtonGroup()
        for rb in [self.rb_thresh_none, self.rb_thresh_manual, self.rb_thresh_adaptive, self.rb_thresh_otsu]:
            self.btn_group_thresh.addButton(rb)

        # manual threshold spin
        self.thresh_spin = QSpinBox()
        self.thresh_spin.setRange(0, 255)
        self.thresh_spin.setValue(128)

        tg_layout.addWidget(self.rb_thresh_none)
        tg_layout.addWidget(self.rb_thresh_manual)
        tg_layout.addWidget(self.rb_thresh_adaptive)
        tg_layout.addWidget(self.rb_thresh_otsu)
        tg_layout.addWidget(QLabel("Value:"))
        tg_layout.addWidget(self.thresh_spin)

        # Hist eq + CLAHE
        hist_clahe_box = QGroupBox("Histogram / CLAHE")
        hc_layout = QVBoxLayout(hist_clahe_box)
        self.cb_hist_equal = QCheckBox("Histogram Equalization")
        self.cb_clahe = QCheckBox("CLAHE")
        hc_layout.addWidget(self.cb_hist_equal)
        hc_layout.addWidget(self.cb_clahe)

        # Morph group
        morph_box = QGroupBox("Morphological")
        morph_layout = QFormLayout(morph_box)
        self.morph_combo = QComboBox()
        self.morph_combo.addItems(["none", "erode", "dilate", "opening", "closing", "tophat", "blackhat"])
        self.morph_kernel_spin = QSpinBox()
        self.morph_kernel_spin.setRange(1, 31)
        self.morph_kernel_spin.setValue(3)
        morph_layout.addRow("Operation:", self.morph_combo)
        morph_layout.addRow("Kernel Size:", self.morph_kernel_spin)

        # Denoise
        denoise_box = QGroupBox("Denoise")
        denoise_layout = QFormLayout(denoise_box)
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["None", "Gaussian", "Median", "Bilateral", "NLM"])
        self.denoise_kernel_spin = QSpinBox()
        self.denoise_kernel_spin.setRange(1, 31)
        self.denoise_kernel_spin.setValue(3)
        denoise_layout.addRow("Type:", self.denoise_combo)
        denoise_layout.addRow("Kernel:", self.denoise_kernel_spin)

        # Channel extraction
        channel_box = QGroupBox("Channel Extraction")
        channel_layout = QHBoxLayout(channel_box)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["None", "B", "G", "R"])
        channel_layout.addWidget(self.channel_combo)

        # Resize
        resize_box = QGroupBox("Resize")
        resize_layout = QFormLayout(resize_box)
        self.resize_w_spin = QSpinBox()
        self.resize_w_spin.setRange(0, 5000)
        self.resize_w_spin.setValue(0)
        self.resize_h_spin = QSpinBox()
        self.resize_h_spin.setRange(0, 5000)
        self.resize_h_spin.setValue(0)
        resize_layout.addRow("Width:", self.resize_w_spin)
        resize_layout.addRow("Height:", self.resize_h_spin)

        # Enhance Red
        red_box = QGroupBox("Enhance Red Channel")
        red_layout = QFormLayout(red_box)
        self.cb_enhance_red = QCheckBox("Enable")
        self.red_brightness_spin = QDoubleSpinBox()
        self.red_brightness_spin.setRange(0.1, 5.0)
        self.red_brightness_spin.setValue(1.0)
        self.red_contrast_spin = QDoubleSpinBox()
        self.red_contrast_spin.setRange(0.1, 5.0)
        self.red_contrast_spin.setValue(1.0)
        red_layout.addRow("Activate:", self.cb_enhance_red)
        red_layout.addRow("Brightness:", self.red_brightness_spin)
        red_layout.addRow("Contrast:", self.red_contrast_spin)

        # Put them all in single_box
        single_box.addWidget(basic_box)
        single_box.addWidget(thresh_group_box)
        single_box.addWidget(hist_clahe_box)
        single_box.addWidget(morph_box)
        single_box.addWidget(denoise_box)
        single_box.addWidget(channel_box)
        single_box.addWidget(resize_box)
        single_box.addWidget(red_box)

        btn_apply = QPushButton("Apply Preprocessing")
        btn_apply.clicked.connect(self.apply_preprocessing)
        single_box.addWidget(btn_apply)

        # Display original/processed
        h_disp = QHBoxLayout()
        self.label_original = QLabel("Original")
        self.label_original.setFixedSize(300, 300)
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_processed = QLabel("Processed")
        self.label_processed.setFixedSize(300, 300)
        self.label_processed.setAlignment(Qt.AlignCenter)
        h_disp.addWidget(self.label_original)
        h_disp.addWidget(self.label_processed)
        single_box.addLayout(h_disp)

        btn_save_proc = QPushButton("Save Processed Image")
        btn_save_proc.clicked.connect(self.save_processed_image)
        single_box.addWidget(btn_save_proc)

        container_layout.addLayout(single_box)
        container_layout.addSpacing(20)

        # ---------- BULK PREPROCESSING -------------
        bulk_box = QVBoxLayout()
        bulk_box.addWidget(QLabel("<b>Bulk Preprocessing</b>"))

        h_bulk1 = QHBoxLayout()
        self.bulk_input_dir = QLineEdit()
        btn_bulk_in = QPushButton("Input Folder...")
        btn_bulk_in.clicked.connect(self.browse_bulk_input_folder)
        self.bulk_output_dir = QLineEdit()
        btn_bulk_out = QPushButton("Output Folder...")
        btn_bulk_out.clicked.connect(self.browse_bulk_output_folder)
        h_bulk1.addWidget(QLabel("Input:"))
        h_bulk1.addWidget(self.bulk_input_dir)
        h_bulk1.addWidget(btn_bulk_in)
        h_bulk1.addSpacing(20)
        h_bulk1.addWidget(QLabel("Output:"))
        h_bulk1.addWidget(self.bulk_output_dir)
        h_bulk1.addWidget(btn_bulk_out)
        bulk_box.addLayout(h_bulk1)

        h_bulk2 = QHBoxLayout()
        self.bulk_format_combo = QComboBox()
        self.bulk_format_combo.addItems(["jpg", "png", "tiff"])
        h_bulk2.addWidget(QLabel("Convert to Format:"))
        h_bulk2.addWidget(self.bulk_format_combo)

        h_bulk2.addWidget(QLabel("Output Suffix:"))
        self.bulk_suffix_edit = QLineEdit()
        self.bulk_suffix_edit.setText("_processed")
        h_bulk2.addWidget(self.bulk_suffix_edit)

        bulk_box.addLayout(h_bulk2)

        btn_bulk_proc = QPushButton("Process Bulk Images")
        btn_bulk_proc.clicked.connect(self.process_bulk_images)
        bulk_box.addWidget(btn_bulk_proc)

        container_layout.addLayout(bulk_box)
        container_layout.addSpacing(20)

        # ----------- BULK OVERLAY --------------
        overlay_box = QVBoxLayout()
        overlay_box.addWidget(QLabel("<b>Bulk Overlay (ch01 -> Red, ch02 -> Green)</b>"))

        # input/output
        h_ov1 = QHBoxLayout()
        self.overlay_input_dir = QLineEdit()
        btn_ov_in = QPushButton("Overlay Input...")
        btn_ov_in.clicked.connect(self.browse_overlay_input_folder)
        self.overlay_output_dir = QLineEdit()
        btn_ov_out = QPushButton("Overlay Output...")
        btn_ov_out.clicked.connect(self.browse_overlay_output_folder)
        h_ov1.addWidget(QLabel("Input:"))
        h_ov1.addWidget(self.overlay_input_dir)
        h_ov1.addWidget(btn_ov_in)
        h_ov1.addSpacing(20)
        h_ov1.addWidget(QLabel("Output:"))
        h_ov1.addWidget(self.overlay_output_dir)
        h_ov1.addWidget(btn_ov_out)
        overlay_box.addLayout(h_ov1)

        # overlay options
        form_ov = QFormLayout()
        self.overlay_ch01_edit = QLineEdit()
        self.overlay_ch01_edit.setText("-ch01")
        self.overlay_ch02_edit = QLineEdit()
        self.overlay_ch02_edit.setText("-ch02")

        self.cb_overlay_normalize = QCheckBox("Normalize Channels")
        self.cb_overlay_hist_equal = QCheckBox("Histogram Equalization")
        self.cb_overlay_clahe = QCheckBox("CLAHE")

        self.overlay_suffix_edit = QLineEdit()
        self.overlay_suffix_edit.setText("")

        form_ov.addRow("ch01 substring:", self.overlay_ch01_edit)
        form_ov.addRow("ch02 substring:", self.overlay_ch02_edit)
        form_ov.addRow("Overlay Suffix:", self.overlay_suffix_edit)
        form_ov.addRow(self.cb_overlay_normalize)
        form_ov.addRow(self.cb_overlay_hist_equal)
        form_ov.addRow(self.cb_overlay_clahe)

        overlay_box.addLayout(form_ov)

        btn_ov_process = QPushButton("Process Overlay in Bulk")
        btn_ov_process.clicked.connect(self.process_overlay_in_bulk)
        overlay_box.addWidget(btn_ov_process)

        container_layout.addLayout(overlay_box)
        container_layout.addSpacing(20)

        # ---------- CHANNEL FILTER -------------
        filter_box = QVBoxLayout()
        filter_box.addWidget(QLabel("<b>Filter Images by Channel (or substring)</b>"))

        h_f1 = QHBoxLayout()
        self.filter_input_dir = QLineEdit()
        btn_filter_in = QPushButton("Filter Input Folder...")
        btn_filter_in.clicked.connect(self.browse_filter_input_folder)
        self.filter_output_dir = QLineEdit()
        btn_filter_out = QPushButton("Filter Output Folder...")
        btn_filter_out.clicked.connect(self.browse_filter_output_folder)

        h_f1.addWidget(QLabel("Input:"))
        h_f1.addWidget(self.filter_input_dir)
        h_f1.addWidget(btn_filter_in)
        h_f1.addSpacing(20)
        h_f1.addWidget(QLabel("Output:"))
        h_f1.addWidget(self.filter_output_dir)
        h_f1.addWidget(btn_filter_out)
        filter_box.addLayout(h_f1)

        h_f2 = QHBoxLayout()
        self.filter_substring_edit = QLineEdit()
        self.filter_substring_edit.setText("ch01")
        h_f2.addWidget(QLabel("Substring to Keep:"))
        h_f2.addWidget(self.filter_substring_edit)

        filter_box.addLayout(h_f2)

        btn_filter_go = QPushButton("Filter and Copy Images")
        btn_filter_go.clicked.connect(self.filter_and_copy_images)
        filter_box.addWidget(btn_filter_go)

        container_layout.addLayout(filter_box)
        container_layout.addSpacing(20)

        # ---------- LOG + PROGRESS -----------
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        container_layout.addWidget(self.text_log)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        container_layout.addWidget(self.progress_bar)

        container_layout.addStretch()

        return self.widget_main

    ##########################################
    # Single-image logic
    ##########################################
    def browse_preprocess_image(self):
        folder = os.path.dirname(self.preprocess_img_path.text())
        fpath, _ = QFileDialog.getOpenFileName(
            self.widget_main, "Select an Image", folder,
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if fpath:
            self.load_image_for_preview(fpath)
            # also load siblings for next/prev
            self.load_folder_images(os.path.dirname(fpath), fpath)

    def load_folder_images(self, folder: str, current_file: str):
        """
        Loads all valid images from 'folder' into self.current_folder_images
        and sets self.current_image_index to the index of current_file.
        Normalizes paths for Windows compatibility.
        """
        folder = os.path.normpath(folder)
        current_file_norm = os.path.normpath(current_file)
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        self.current_folder_images = []
        for fn in os.listdir(folder):
            full_path = os.path.join(folder, fn)
            if not os.path.isfile(full_path):
                continue
            lf = fn.lower()
            if any(lf.endswith(ext) for ext in valid_exts):
                self.current_folder_images.append(os.path.normpath(full_path))

        self.current_folder_images.sort()
        try:
            self.current_image_index = self.current_folder_images.index(current_file_norm)
        except ValueError:
            # If somehow the current_file isn't in the folder list,
            # just set the index to 0 (or handle as you prefer).
            self.current_image_index = 0

    def next_image(self):
        if not self.current_folder_images:
            return
        if self.current_image_index < len(self.current_folder_images) - 1:
            self.current_image_index += 1
        path = self.current_folder_images[self.current_image_index]
        self.load_image_for_preview(path)

    def prev_image(self):
        if not self.current_folder_images:
            return
        if self.current_image_index > 0:
            self.current_image_index -= 1
        path = self.current_folder_images[self.current_image_index]
        self.load_image_for_preview(path)

    def load_image_for_preview(self, fpath: str):
        self.preprocess_img_path.setText(fpath)
        img = cv2.imread(fpath)
        if img is None:
            QMessageBox.warning(self.widget_main, "Error", f"Could not load {fpath}")
            return
        self.original_image = img
        pm = convert_to_qpixmap(img)
        self.label_original.setPixmap(
            pm.scaled(self.label_original.width(), self.label_original.height(), Qt.KeepAspectRatio)
        )
        self.processed_image = None
        self.label_processed.setPixmap(QPixmap())  # clear previous

    def apply_preprocessing(self):
        if self.original_image is None:
            QMessageBox.warning(self.widget_main, "Error", "No image loaded.")
            return
        self.processed_image = self.run_preprocessing_pipeline(self.original_image)
        pm = convert_to_qpixmap(self.processed_image)
        self.label_processed.setPixmap(
            pm.scaled(self.label_processed.width(), self.label_processed.height(), Qt.KeepAspectRatio)
        )

    def run_preprocessing_pipeline(self, img: np.ndarray) -> np.ndarray:
        # Gather user params
        grayscale = self.cb_grayscale.isChecked()
        invert_ = self.cb_invert.isChecked()

        # threshold radio
        if self.rb_thresh_none.isChecked():
            thr_mode = "none"
        elif self.rb_thresh_manual.isChecked():
            thr_mode = "manual"
        elif self.rb_thresh_adaptive.isChecked():
            thr_mode = "adaptive"
        else:
            thr_mode = "otsu"
        thr_val = self.thresh_spin.value()

        do_hist = self.cb_hist_equal.isChecked()
        do_clahe = self.cb_clahe.isChecked()

        morph_op = self.morph_combo.currentText().lower()
        morph_k = self.morph_kernel_spin.value()

        denoise_type = self.denoise_combo.currentText()
        denoise_k = self.denoise_kernel_spin.value()

        channel_ = self.channel_combo.currentText()
        rw = self.resize_w_spin.value()
        rh = self.resize_h_spin.value()

        enh_red = self.cb_enhance_red.isChecked()
        red_b = self.red_brightness_spin.value()
        red_c = self.red_contrast_spin.value()

        return self.do_preprocess(
            img, grayscale, invert_, thr_mode, thr_val,
            do_hist, do_clahe, morph_op, morph_k,
            denoise_type, denoise_k, channel_, rw, rh,
            enh_red, red_b, red_c
        )

    def do_preprocess(self, image: np.ndarray,
                      grayscale: bool, invert_: bool,
                      thr_mode: str, thr_val: int,
                      do_hist: bool, do_clahe: bool,
                      morph_op: str, morph_k: int,
                      denoise_type: str, denoise_k: int,
                      channel_: str,
                      rw: int, rh: int,
                      enhance_red: bool,
                      red_b: float, red_c: float) -> np.ndarray:
        """Single universal function that does the pipeline steps."""
        img = image.copy()

        # channel extraction if user picks
        if channel_ in ["R", "G", "B"]:
            idx = {"B": 0, "G": 1, "R": 2}[channel_]
            single = img[..., idx]
            img = cv2.cvtColor(single, cv2.COLOR_GRAY2BGR)

        # denoise
        if denoise_type == "Gaussian":
            img = cv2.GaussianBlur(img, (denoise_k, denoise_k), 0)
        elif denoise_type == "Median":
            img = cv2.medianBlur(img, denoise_k)
        elif denoise_type == "Bilateral":
            img = cv2.bilateralFilter(img, d=denoise_k, sigmaColor=75, sigmaSpace=75)
        elif denoise_type == "NLM":
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # grayscale
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # invert
        if invert_:
            img = 255 - img

        # threshold
        if thr_mode != "none":
            if thr_mode == "otsu":
                if grayscale:
                    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    img = mask
                else:
                    img = otsu_threshold_channelwise(img)
            elif thr_mode == "adaptive":
                if grayscale:
                    adapt = cv2.adaptiveThreshold(
                        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    img = adapt
                else:
                    img = adaptive_threshold_channelwise(img)
            elif thr_mode == "manual":
                if grayscale:
                    _, mask = cv2.threshold(img, thr_val, 255, cv2.THRESH_BINARY)
                    img = mask
                else:
                    img = threshold_channelwise(img, thr_val)

        # hist eq
        if do_hist:
            if grayscale:
                img = cv2.equalizeHist(img)
            else:
                img = apply_hist_equal_channelwise(img)

        # clahe
        if do_clahe:
            if grayscale:
                clahe_func = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img = clahe_func.apply(img)
            else:
                img = apply_clahe_channelwise(img)

        # morphology
        if morph_op != "none":
            if grayscale:
                kernel = np.ones((morph_k, morph_k), np.uint8)
                if morph_op == "erode":
                    img = cv2.erode(img, kernel, iterations=1)
                elif morph_op == "dilate":
                    img = cv2.dilate(img, kernel, iterations=1)
                elif morph_op == "opening":
                    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                elif morph_op == "closing":
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                elif morph_op == "tophat":
                    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
                elif morph_op == "blackhat":
                    img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            else:
                img = morphological_op_channelwise(img, morph_op, morph_k)

        # convert back to 3-channel if grayscale
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # enhance red channel
        if (not grayscale) and enhance_red and channel_ not in ["B","G"]:
            img = enhance_red_channel(img, bright_factor=red_b, contrast_factor=red_c)

        # resize if user specified
        if rw > 0 and rh > 0:
            img = cv2.resize(img, (rw, rh))

        return img

    def save_processed_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self.widget_main, "Error", "No processed image to save.")
            return
        fpath, _ = QFileDialog.getSaveFileName(
            self.widget_main, "Save Processed Image", "",
            "Images (*.png *.jpg *.tiff)"
        )
        if fpath:
            cv2.imwrite(fpath, self.processed_image)
            QMessageBox.information(self.widget_main, "Saved", f"Saved to {fpath}")

    ##############################################
    # Bulk Preprocessing
    ##############################################
    def browse_bulk_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Input Folder")
        if folder:
            self.bulk_input_dir.setText(folder)

    def browse_bulk_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Output Folder")
        if folder:
            self.bulk_output_dir.setText(folder)

    def process_bulk_images(self):
        in_dir = self.bulk_input_dir.text().strip()
        out_dir = self.bulk_output_dir.text().strip()
        if not os.path.isdir(in_dir):
            QMessageBox.warning(self.widget_main, "Error", "Invalid input folder.")
            return
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_fmt = self.bulk_format_combo.currentText()
        out_suffix = self.bulk_suffix_edit.text().strip()

        # gather the same parameters from single
        # wrap them in a dict
        params = {
            "pipeline_func": self._bulk_pipeline_wrapper,
        }

        # We'll run it in a QThread
        self.bulk_thread = BulkPreprocessingThread(
            in_dir, out_dir, out_suffix, out_fmt, params, parent=self.widget_main
        )
        self.bulk_thread.progress_signal.connect(self.update_progress)
        self.bulk_thread.log_signal.connect(self.append_log)
        self.bulk_thread.done_signal.connect(self.bulk_done)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indefinite
        self.bulk_thread.start()

    def _bulk_pipeline_wrapper(self, img: np.ndarray, _: dict) -> np.ndarray:
        """Just calls run_preprocessing_pipeline with current UI settings"""
        return self.run_preprocessing_pipeline(img)

    def bulk_done(self, out_str: str):
        self.progress_bar.setVisible(False)
        if out_str == "ERROR":
            self.append_log("Bulk preprocessing encountered an error.")
            return
        self.append_log("Bulk preprocessing finished.")
        QMessageBox.information(self.widget_main, "Done", f"Bulk preprocessing done. Output: {out_str}")

    ##############################################
    # Bulk Overlay
    ##############################################
    def browse_overlay_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Overlay Input Folder")
        if folder:
            self.overlay_input_dir.setText(folder)

    def browse_overlay_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Overlay Output Folder")
        if folder:
            self.overlay_output_dir.setText(folder)

    def process_overlay_in_bulk(self):
        in_dir = self.overlay_input_dir.text().strip()
        out_dir = self.overlay_output_dir.text().strip()
        if not os.path.isdir(in_dir):
            QMessageBox.warning(self.widget_main, "Error", "Invalid input overlay folder.")
            return
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # gather params
        hist_eq = self.cb_overlay_hist_equal.isChecked()
        clahe_ = self.cb_overlay_clahe.isChecked()
        norm_ = self.cb_overlay_normalize.isChecked()
        ch01_sub = self.overlay_ch01_edit.text().strip()
        ch02_sub = self.overlay_ch02_edit.text().strip()
        out_suffix = self.overlay_suffix_edit.text().strip()

        params = {
            "hist_equal": hist_eq,
            "clahe": clahe_,
            "normalize_": norm_,
            "ch01_substring": ch01_sub,
            "ch02_substring": ch02_sub
        }

        self.overlay_thread = OverlayThread(
            in_dir, out_dir, out_suffix, params, parent=self.widget_main
        )
        self.overlay_thread.progress_signal.connect(self.update_progress)
        self.overlay_thread.log_signal.connect(self.append_log)
        self.overlay_thread.done_signal.connect(self.overlay_done)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.overlay_thread.start()

    def overlay_done(self, out_str: str):
        self.progress_bar.setVisible(False)
        if out_str == "ERROR":
            self.append_log("Overlay encountered an error.")
            return
        self.append_log("Overlay operation completed.")
        QMessageBox.information(self.widget_main, "Done", f"Overlay done. Output: {out_str}")

    ##############################################
    # Channel Filter
    ##############################################
    def browse_filter_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Filter Input Folder")
        if folder:
            self.filter_input_dir.setText(folder)

    def browse_filter_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Filter Output Folder")
        if folder:
            self.filter_output_dir.setText(folder)

    def filter_and_copy_images(self):
        in_dir = self.filter_input_dir.text().strip()
        out_dir = self.filter_output_dir.text().strip()
        substring = self.filter_substring_edit.text().strip()

        if not os.path.isdir(in_dir):
            QMessageBox.warning(self.widget_main, "Error", "Invalid filter input folder.")
            return
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.filter_thread = FilterThread(in_dir, out_dir, substring, parent=self.widget_main)
        self.filter_thread.progress_signal.connect(self.update_progress)
        self.filter_thread.log_signal.connect(self.append_log)
        self.filter_thread.done_signal.connect(self.filter_done)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.filter_thread.start()

    def filter_done(self, out_str: str):
        self.progress_bar.setVisible(False)
        if out_str == "ERROR":
            self.append_log("Filtering encountered an error.")
            return
        self.append_log("Filtering completed.")
        QMessageBox.information(self.widget_main, "Done", f"Images filtered to: {out_str}")

    ##############################################
    # Shared Log & Progress
    ##############################################
    def update_progress(self, val: int):
        # If we want indefinite, we keep range(0,0),
        # but if we had a total, we could set range(0, total).
        # For demonstration, just ensure progress moves forward:
        self.progress_bar.setRange(0, max(val, self.progress_bar.maximum()))
        self.progress_bar.setValue(val)

    def append_log(self, msg: str):
        self.text_log.append(msg)
        self.text_log.ensureCursorVisible()
