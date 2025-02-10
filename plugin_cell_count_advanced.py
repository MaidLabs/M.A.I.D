# plugin_cell_count_cellpose_cuda.py

import os
import glob
import csv
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFormLayout, QProgressBar, QScrollArea
)
from PyQt5.QtCore import Qt

# Make sure you have installed Cellpose with CUDA-enabled PyTorch if you want GPU.
# Also install cellpose[omni] if you want Omnipose support.
from cellpose import models
from cellpose.utils import outlines_list

# If your framework requires it, adjust the import of BasePlugin accordingly.
from base_plugin import BasePlugin


# -------------------------------------------------------------------------
# Helper function: Create a segmentation outline overlay on top of the image
# -------------------------------------------------------------------------
def create_segmentation_overlay(image_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Create an RGB overlay with red outlines around each segmented mask.

    Args:
        image_rgb (np.ndarray): The original RGB image, scaled 0..1 or 0..255.
        masks (np.ndarray): The integer mask labels from Cellpose.

    Returns:
        np.ndarray: The overlay image in RGB (uint8) format.
    """
    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb * 255).astype(np.uint8)

    outlines = outlines_list(masks)
    overlay = image_rgb.copy()
    for ol in outlines:
        overlay[ol[:, 0], ol[:, 1]] = (255, 0, 0)  # red outline

    return overlay


# -------------------------------------------------------------------------
# Main segmentation function with user-configurable parameters
# -------------------------------------------------------------------------
def run_cellpose_segmentation(
    image_bgr: np.ndarray,
    model_type: str = "cyto",
    custom_model_path: str = "",
    diameter: float = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    mask_threshold: float = 0.0,
    stitch_threshold: float = 0.0,
    do_3d: bool = False,
    normalize: bool = True,
    invert: bool = False,
    net_avg: bool = True,
    batch_size: int = 8,
    channel1: int = 0,
    channel2: int = 0,
    use_gpu: bool = True,
    use_omnipose: bool = False,
    min_size: int = 15,
) -> (int, np.ndarray):
    """
    Run Cellpose (or Omnipose) segmentation with user-configurable parameters.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if model_type.lower() == "custom" and custom_model_path:
        model_to_use = custom_model_path
    else:
        model_to_use = model_type

    if use_omnipose:
        try:
            model = models.OmniPoseModel(gpu=use_gpu, model_type=model_to_use)
        except AttributeError:
            raise RuntimeError(
                "Your Cellpose version does not support Omnipose. "
                "Install or upgrade with: pip install cellpose[omni] --upgrade"
            )
    else:
        model = models.Cellpose(gpu=use_gpu, model_type=model_to_use)

    masks, flows, styles, diams = model.eval(
        image_rgb,
        channels=[channel1, channel2],
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        mask_threshold=mask_threshold,
        stitch_threshold=stitch_threshold,
        do_3D=do_3d,
        normalize=normalize,
        invert=invert,
        net_avg=net_avg,
        batch_size=batch_size,
        min_size=min_size
    )

    cell_count = masks.max()
    return int(cell_count), masks


# -------------------------------------------------------------------------
# Plugin class
# -------------------------------------------------------------------------
class Plugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.plugin_name = "CellPose"

        # We will create our widget_main as a scrollable area
        self.container_widget = None

        # Single image fields
        self.single_image_edit = None
        self.single_count_label = None

        # Bulk fields
        self.bulk_input_edit = None
        self.bulk_results = []

        # Cellpose parameter widgets
        self.model_type_combo = None
        self.custom_model_path_edit = None
        self.diameter_edit = None
        self.flow_threshold_edit = None
        self.cellprob_threshold_edit = None
        self.mask_threshold_edit = None
        self.stitch_threshold_edit = None
        self.do_3d_checkbox = None
        self.normalize_checkbox = None
        self.invert_checkbox = None
        self.net_avg_checkbox = None
        self.batch_size_spin = None
        self.channel1_combo = None
        self.channel2_combo = None
        self.use_gpu_checkbox = None
        self.use_omnipose_checkbox = None
        self.min_size_spin = None

        # Output options
        self.save_overlay_single_checkbox = None
        self.save_overlay_bulk_checkbox = None
        self.save_labelmask_single_checkbox = None
        self.save_labelmask_bulk_checkbox = None
        self.save_outlines_single_checkbox = None
        self.save_outlines_bulk_checkbox = None

        # Progress bar
        self.bulk_progress_bar = None

    def create_tab(self) -> QWidget:
        """
        Create and return a scrollable QScrollArea that contains
        all of our plugin widgets.
        """
        # -- The inner container for our UI --
        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)

        # ------------------- Parameter Group -------------------
        param_group = QGroupBox("Cellpose Parameters")
        param_layout = QFormLayout(param_group)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["cyto", "nuclei", "Custom"])
        self.model_type_combo.setCurrentText("cyto")

        self.custom_model_path_edit = QLineEdit()
        self.custom_model_path_edit.setPlaceholderText("Browse or enter path to .pth if using 'Custom'")
        btn_browse_model = QPushButton("Browse Model...")
        btn_browse_model.clicked.connect(self.browse_model_path)

        self.diameter_edit = QLineEdit()
        self.diameter_edit.setPlaceholderText("None (auto-estimate)")

        self.flow_threshold_edit = QDoubleSpinBox()
        self.flow_threshold_edit.setRange(-10.0, 10.0)
        self.flow_threshold_edit.setValue(0.4)
        self.flow_threshold_edit.setSingleStep(0.1)

        self.cellprob_threshold_edit = QDoubleSpinBox()
        self.cellprob_threshold_edit.setRange(-10.0, 10.0)
        self.cellprob_threshold_edit.setValue(0.0)
        self.cellprob_threshold_edit.setSingleStep(0.1)

        self.mask_threshold_edit = QDoubleSpinBox()
        self.mask_threshold_edit.setRange(-10.0, 10.0)
        self.mask_threshold_edit.setValue(0.0)
        self.mask_threshold_edit.setSingleStep(0.1)

        self.stitch_threshold_edit = QDoubleSpinBox()
        self.stitch_threshold_edit.setRange(0.0, 1.0)
        self.stitch_threshold_edit.setValue(0.0)
        self.stitch_threshold_edit.setSingleStep(0.1)

        self.do_3d_checkbox = QCheckBox("Perform 3D segmentation")
        self.do_3d_checkbox.setChecked(False)

        self.normalize_checkbox = QCheckBox("Normalize Intensity")
        self.normalize_checkbox.setChecked(True)

        self.invert_checkbox = QCheckBox("Invert Intensity")
        self.invert_checkbox.setChecked(False)

        self.net_avg_checkbox = QCheckBox("Use net averaging")
        self.net_avg_checkbox.setChecked(True)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 9999)
        self.batch_size_spin.setValue(8)

        self.channel1_combo = QComboBox()
        self.channel1_combo.addItems(["0", "1", "2"])
        self.channel1_combo.setCurrentIndex(0)

        self.channel2_combo = QComboBox()
        self.channel2_combo.addItems(["0", "1", "2"])
        self.channel2_combo.setCurrentIndex(0)

        self.use_gpu_checkbox = QCheckBox("Use GPU (CUDA)")
        self.use_gpu_checkbox.setChecked(True)

        self.use_omnipose_checkbox = QCheckBox("Use Omnipose (if installed)")
        self.use_omnipose_checkbox.setChecked(False)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 999999)
        self.min_size_spin.setValue(15)

        param_layout.addRow("Model type:", self.model_type_combo)
        h_modelpath = QHBoxLayout()
        h_modelpath.addWidget(self.custom_model_path_edit)
        h_modelpath.addWidget(btn_browse_model)
        param_layout.addRow("Custom Model Path:", h_modelpath)
        param_layout.addRow("Diameter:", self.diameter_edit)
        param_layout.addRow("Flow threshold:", self.flow_threshold_edit)
        param_layout.addRow("Cell prob threshold:", self.cellprob_threshold_edit)
        param_layout.addRow("Mask threshold:", self.mask_threshold_edit)
        param_layout.addRow("Stitch threshold:", self.stitch_threshold_edit)
        param_layout.addRow("Do 3D:", self.do_3d_checkbox)
        param_layout.addRow("Normalize:", self.normalize_checkbox)
        param_layout.addRow("Invert:", self.invert_checkbox)
        param_layout.addRow("Net Avg:", self.net_avg_checkbox)
        param_layout.addRow("Batch Size:", self.batch_size_spin)
        param_layout.addRow("Channel 1:", self.channel1_combo)
        param_layout.addRow("Channel 2:", self.channel2_combo)
        param_layout.addRow("Use GPU:", self.use_gpu_checkbox)
        param_layout.addRow("Use Omnipose:", self.use_omnipose_checkbox)
        param_layout.addRow("Min object size:", self.min_size_spin)

        main_layout.addWidget(param_group)

        # ------------------- Output Options -------------------
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout(output_group)

        self.save_overlay_single_checkbox = QCheckBox("Save overlay image (single)")
        self.save_overlay_single_checkbox.setChecked(False)

        self.save_overlay_bulk_checkbox = QCheckBox("Save overlay images (bulk)")
        self.save_overlay_bulk_checkbox.setChecked(False)

        self.save_labelmask_single_checkbox = QCheckBox("Save label mask (single)")
        self.save_labelmask_single_checkbox.setChecked(False)

        self.save_labelmask_bulk_checkbox = QCheckBox("Save label masks (bulk)")
        self.save_labelmask_bulk_checkbox.setChecked(False)

        self.save_outlines_single_checkbox = QCheckBox("Save outlines (npy) (single)")
        self.save_outlines_single_checkbox.setChecked(False)

        self.save_outlines_bulk_checkbox = QCheckBox("Save outlines (npy) (bulk)")
        self.save_outlines_bulk_checkbox.setChecked(False)

        output_layout.addRow(self.save_overlay_single_checkbox)
        output_layout.addRow(self.save_overlay_bulk_checkbox)
        output_layout.addRow(self.save_labelmask_single_checkbox)
        output_layout.addRow(self.save_labelmask_bulk_checkbox)
        output_layout.addRow(self.save_outlines_single_checkbox)
        output_layout.addRow(self.save_outlines_bulk_checkbox)

        main_layout.addWidget(output_group)

        # ------------------- Single Image Counting -------------------
        main_layout.addWidget(QLabel("Single Image Cell Count (Cellpose)", alignment=Qt.AlignLeft))
        h_single = QHBoxLayout()
        self.single_image_edit = QLineEdit()
        btn_single_browse = QPushButton("Browse Image...")
        btn_single_browse.clicked.connect(self.browse_single_image)
        h_single.addWidget(QLabel("Image:"))
        h_single.addWidget(self.single_image_edit)
        h_single.addWidget(btn_single_browse)
        main_layout.addLayout(h_single)

        btn_single_run = QPushButton("Count Cells (Single)")
        btn_single_run.clicked.connect(self.run_single_count)
        main_layout.addWidget(btn_single_run)

        self.single_count_label = QLabel("Count: N/A")
        main_layout.addWidget(self.single_count_label)

        # ------------------- Bulk Counting -------------------
        main_layout.addSpacing(20)
        main_layout.addWidget(QLabel("Bulk Cell Count (recursive, Cellpose)", alignment=Qt.AlignLeft))

        h_bulk = QHBoxLayout()
        self.bulk_input_edit = QLineEdit()
        btn_bulk_folder = QPushButton("Browse Folder...")
        btn_bulk_folder.clicked.connect(self.browse_bulk_folder)
        h_bulk.addWidget(QLabel("Folder:"))
        h_bulk.addWidget(self.bulk_input_edit)
        h_bulk.addWidget(btn_bulk_folder)
        main_layout.addLayout(h_bulk)

        self.bulk_progress_bar = QProgressBar()
        self.bulk_progress_bar.setValue(0)
        main_layout.addWidget(self.bulk_progress_bar)

        btn_bulk_run = QPushButton("Count Cells (Bulk)")
        btn_bulk_run.clicked.connect(self.run_bulk_count)
        main_layout.addWidget(btn_bulk_run)

        btn_export_csv = QPushButton("Export Bulk Results to CSV")
        btn_export_csv.clicked.connect(self.export_bulk_csv)
        main_layout.addWidget(btn_export_csv)

        # -- Wrap our container_widget in a QScrollArea to allow scrolling --
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.container_widget)
        scroll_area.setWidgetResizable(True)

        return scroll_area

    # ---------------------------------------------------------------------
    # Browse for custom model path
    # ---------------------------------------------------------------------
    def browse_model_path(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self.container_widget,
            "Select Custom Cellpose Model",
            filter="Model Files (*.pth *.pt *.pkl)"
        )
        if fpath:
            self.custom_model_path_edit.setText(fpath)

    # ---------------------------------------------------------------------
    # Single image operations
    # ---------------------------------------------------------------------
    def browse_single_image(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self.container_widget,
            "Select an Image for Cell Counting",
            filter="Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if fpath:
            self.single_image_edit.setText(fpath)

    def run_single_count(self):
        fpath = self.single_image_edit.text().strip()
        if not os.path.isfile(fpath):
            QMessageBox.warning(self.container_widget, "Error", "Invalid image file.")
            return

        img = cv2.imread(fpath)
        if img is None:
            QMessageBox.warning(self.container_widget, "Error", f"Could not load {fpath}")
            return

        model_type = self.model_type_combo.currentText()
        custom_model_path = self.custom_model_path_edit.text().strip()
        dia_text = self.diameter_edit.text().strip()
        diameter = float(dia_text) if dia_text else None
        flow_threshold = self.flow_threshold_edit.value()
        cellprob_threshold = self.cellprob_threshold_edit.value()
        mask_threshold = self.mask_threshold_edit.value()
        stitch_threshold = self.stitch_threshold_edit.value()
        do_3d = self.do_3d_checkbox.isChecked()
        normalize = self.normalize_checkbox.isChecked()
        invert = self.invert_checkbox.isChecked()
        net_avg = self.net_avg_checkbox.isChecked()
        batch_size = self.batch_size_spin.value()
        channel1 = int(self.channel1_combo.currentText())
        channel2 = int(self.channel2_combo.currentText())
        use_gpu = self.use_gpu_checkbox.isChecked()
        use_omnipose = self.use_omnipose_checkbox.isChecked()
        min_size = self.min_size_spin.value()

        try:
            print(f"Running Cellpose on single image: {fpath}")
            count, masks = run_cellpose_segmentation(
                img,
                model_type=model_type,
                custom_model_path=custom_model_path,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                mask_threshold=mask_threshold,
                stitch_threshold=stitch_threshold,
                do_3d=do_3d,
                normalize=normalize,
                invert=invert,
                net_avg=net_avg,
                batch_size=batch_size,
                channel1=channel1,
                channel2=channel2,
                use_gpu=use_gpu,
                use_omnipose=use_omnipose,
                min_size=min_size
            )
            self.single_count_label.setText(f"Count: {count}")
            print(f"Cell count result: {count}")

            # If the user wants an overlay for single images
            if self.save_overlay_single_checkbox.isChecked():
                overlay = create_segmentation_overlay(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), masks)
                save_path = os.path.splitext(fpath)[0] + "_overlay.png"
                cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"Saved overlay image to: {save_path}")

            if self.save_labelmask_single_checkbox.isChecked():
                labelmask_path = os.path.splitext(fpath)[0] + "_labels.tif"
                cv2.imwrite(labelmask_path, masks.astype(np.uint16))
                print(f"Saved label mask to: {labelmask_path}")

            if self.save_outlines_single_checkbox.isChecked():
                ol_list = outlines_list(masks)
                outlines_save_path = os.path.splitext(fpath)[0] + "_outlines.npy"
                np.save(outlines_save_path, ol_list)
                print(f"Saved outlines to: {outlines_save_path}")

        except Exception as e:
            QMessageBox.critical(self.container_widget, "Cell Counting Error", str(e))
            print(f"Error during single image counting: {str(e)}")

    # ---------------------------------------------------------------------
    # Bulk operations
    # ---------------------------------------------------------------------
    def browse_bulk_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self.container_widget, "Select Folder for Bulk Cell Counting"
        )
        if folder:
            self.bulk_input_edit.setText(folder)

    def run_bulk_count(self):
        in_dir = self.bulk_input_edit.text().strip()
        if not os.path.isdir(in_dir):
            QMessageBox.warning(self.container_widget, "Error", "Invalid folder.")
            return

        self.bulk_results = []

        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(in_dir, "**", ext), recursive=True))

        if not image_files:
            QMessageBox.information(self.container_widget, "No Images", "No images found in that folder.")
            return

        model_type = self.model_type_combo.currentText()
        custom_model_path = self.custom_model_path_edit.text().strip()
        dia_text = self.diameter_edit.text().strip()
        diameter = float(dia_text) if dia_text else None
        flow_threshold = self.flow_threshold_edit.value()
        cellprob_threshold = self.cellprob_threshold_edit.value()
        mask_threshold = self.mask_threshold_edit.value()
        stitch_threshold = self.stitch_threshold_edit.value()
        do_3d = self.do_3d_checkbox.isChecked()
        normalize = self.normalize_checkbox.isChecked()
        invert = self.invert_checkbox.isChecked()
        net_avg = self.net_avg_checkbox.isChecked()
        batch_size = self.batch_size_spin.value()
        channel1 = int(self.channel1_combo.currentText())
        channel2 = int(self.channel2_combo.currentText())
        use_gpu = self.use_gpu_checkbox.isChecked()
        use_omnipose = self.use_omnipose_checkbox.isChecked()
        min_size = self.min_size_spin.value()

        total_files = len(image_files)
        self.bulk_progress_bar.setValue(0)

        print(f"Starting bulk cell counting in: {in_dir}")
        for i, fpath in enumerate(image_files):
            img = cv2.imread(fpath)
            if img is None:
                print(f"Skipping unreadable file: {fpath}")
                continue

            try:
                count, masks = run_cellpose_segmentation(
                    img,
                    model_type=model_type,
                    custom_model_path=custom_model_path,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    mask_threshold=mask_threshold,
                    stitch_threshold=stitch_threshold,
                    do_3d=do_3d,
                    normalize=normalize,
                    invert=invert,
                    net_avg=net_avg,
                    batch_size=batch_size,
                    channel1=channel1,
                    channel2=channel2,
                    use_gpu=use_gpu,
                    use_omnipose=use_omnipose,
                    min_size=min_size
                )
                print(f"[{i+1}/{total_files}] {os.path.basename(fpath)}: count={count}")

                if self.save_overlay_bulk_checkbox.isChecked():
                    overlay = create_segmentation_overlay(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), masks)
                    overlay_path = os.path.splitext(fpath)[0] + "_overlay.png"
                    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                if self.save_labelmask_bulk_checkbox.isChecked():
                    labelmask_path = os.path.splitext(fpath)[0] + "_labels.tif"
                    cv2.imwrite(labelmask_path, masks.astype(np.uint16))
                    print(f"Saved label mask to: {labelmask_path}")

                if self.save_outlines_bulk_checkbox.isChecked():
                    ol_list = outlines_list(masks)
                    outlines_save_path = os.path.splitext(fpath)[0] + "_outlines.npy"
                    np.save(outlines_save_path, ol_list)

                self.bulk_results.append((fpath, count))
            except Exception as e:
                print(f"Error on {fpath}: {e}")
                self.bulk_results.append((fpath, -1))

            progress_value = int((i + 1) / total_files * 100)
            self.bulk_progress_bar.setValue(progress_value)

        QMessageBox.information(
            self.container_widget,
            "Bulk Complete",
            f"Finished counting on {len(self.bulk_results)} images.\n"
            "You can now export CSV if you like."
        )
        print("Bulk processing complete.")

    def export_bulk_csv(self):
        if not self.bulk_results:
            QMessageBox.information(self.container_widget, "No Results", "No bulk results to export. Run Bulk Counting first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self.container_widget, "Save CSV", filter="CSV Files (*.csv)"
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Filepath", "CellCount"])
                for (fpath, count) in self.bulk_results:
                    writer.writerow([fpath, count])

            QMessageBox.information(self.container_widget, "Exported", f"CSV exported to {save_path}")
            print(f"Bulk results exported to {save_path}")
        except Exception as e:
            QMessageBox.critical(self.container_widget, "Export Error", str(e))
            print(f"Error exporting CSV: {str(e)}")
