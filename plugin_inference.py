# plugin_inference.py

import os
import glob
import time
import traceback
import json
import csv
import cv2
import torch
import logging
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QProgressBar,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime, QUrl
from PyQt5.QtGui import QPixmap, QImage, QDesktopServices

from base_plugin import BasePlugin
from plugins.plugin_training import MaidClassifier

# Try importing torchvision for Model Zoo integration
try:
    import torchvision
    from torchvision.models import (
        resnet18, resnet50, resnet101, mobilenet_v2, vit_b_16,
        densenet121, alexnet, inception_v3
    )
    ZOO_MODELS = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "mobilenet_v2": mobilenet_v2,
        "densenet121": densenet121,
        "alexnet": alexnet,
        "inception_v3": inception_v3,
        "vit_b_16": vit_b_16
    }
    HAVE_TORCHVISION = True
except ImportError:
    ZOO_MODELS = {}
    HAVE_TORCHVISION = False

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Classification metrics
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# SHAP
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

# Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

import tempfile


def convert_to_qpixmap(numpy_image: np.ndarray) -> QPixmap:
    """
    Converts a BGR or RGB NumPy image into a QPixmap for PyQt display.
    """
    array_copied = np.ascontiguousarray(numpy_image)
    height, width, channels = array_copied.shape
    bytes_per_line = channels * width
    q_image = QImage(array_copied.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    q_image_final = q_image.copy()
    return QPixmap.fromImage(q_image_final)


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Applies a Grad-CAM or Grad-CAM++ heatmap to the original BGR image.
    'cam' should be normalized between 0 and 1.
    """
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), colormap)
    heatmap = heatmap.astype(np.float32) / 255.0
    image_float = image.astype(np.float32) / 255.0
    overlay = alpha * heatmap + (1 - alpha) * image_float

    if overlay.max() > 0:
        overlay /= overlay.max()
    overlay = (overlay * 255).astype(np.uint8)
    return overlay


class GradCAM:
    """
    Basic Grad-CAM implementation, hooking a specified model layer.
    Reference: https://arxiv.org/abs/1610.02391
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for the specified target_class.
        """
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling of gradients
        alpha = gradients.mean(dim=[1, 2], keepdim=True)
        weighted_activations = alpha * activations
        cam = weighted_activations.sum(dim=0)
        cam = torch.relu(cam)

        # Normalize to [0,1]
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam.cpu().numpy()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation: https://arxiv.org/abs/1710.11063
    """
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        grads_power_2 = gradients ** 2
        grads_power_3 = gradients ** 3
        eps = 1e-10

        # Denominator from the Grad-CAM++ formula
        denom = 2 * grads_power_2 + \
                torch.sum(activations * grads_power_3, dim=[1, 2], keepdim=True) / \
                (torch.sum(activations * gradients, dim=[1, 2], keepdim=True) + eps)

        alpha = grads_power_2 / (denom + eps)
        pos_grad = torch.relu(gradients)
        weights = (alpha * pos_grad).mean(dim=[1, 2], keepdim=True)

        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)

        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam.cpu().numpy()


class XGradCAM(GradCAM):
    """
    Another Grad-CAM variant (XGrad-CAM).
    Reference: https://arxiv.org/abs/1904.00645
    """
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        # XGradCAM weighting
        alpha = (gradients * activations).mean(dim=[1, 2], keepdim=True)
        cam = torch.sum(alpha * activations, dim=0)
        cam = torch.relu(cam)

        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam.cpu().numpy()


class QTextEditLogger(logging.Handler):
    """
    A logging handler that sends logs to a PyQt QTextEdit widget.
    """
    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self.text_edit.append(msg)
        self.text_edit.ensureCursorVisible()


class SingleInferenceThread(QThread):
    """
    Threaded single-image inference.
    """
    progress_signal = pyqtSignal(str)
    done_signal = pyqtSignal(object, object, str)

    def __init__(
        self,
        bgr_image: np.ndarray,
        model: torch.nn.Module,
        device: torch.device,
        transform: A.Compose,
        cam_variant: str,
        target_layer: torch.nn.Module,
        top_k: int = 1,
        min_conf: float = 10.0,
        overlay_alpha: float = 0.5,
        colormap_id: int = cv2.COLORMAP_JET,
        do_gradcam: bool = True,
        class_names: list = None,
        parent=None
    ):
        super().__init__(parent)
        self.bgr_image = bgr_image
        self.model = model
        self.device = device
        self.transform = transform
        self.cam_variant = cam_variant
        self.target_layer = target_layer
        self.top_k = top_k
        self.min_conf = min_conf
        self.overlay_alpha = overlay_alpha
        self.colormap_id = colormap_id
        self.do_gradcam = do_gradcam
        self.class_names = class_names

        self.overlay_result = None
        self.label_str = None

    def run(self):
        try:
            rgb_img = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=rgb_img)
            input_tensor = aug["image"].unsqueeze(0).to(self.device)

            self.progress_signal.emit("Running forward pass...")
            with torch.no_grad():
                if self.device.type == 'cuda':
                    try:
                        with torch.cuda.amp.autocast():
                            logits = self.model(input_tensor)
                    except RuntimeError as oom_e:
                        if "out of memory" in str(oom_e).lower():
                            self.progress_signal.emit("[ERROR] CUDA OOM in single inference.")
                            self.done_signal.emit(None, self.bgr_image, "OOM Error.")
                            return
                        else:
                            raise
                else:
                    logits = self.model(input_tensor)

                probs = torch.softmax(logits, dim=1)

            topk_vals, topk_indices = torch.topk(probs, k=self.top_k, dim=1)
            result_items = []

            if self.class_names:
                n_classes = len(self.class_names)
            else:
                n_classes = None

            for rank in range(topk_vals.size(1)):
                cls_idx = topk_indices[0, rank].item()
                conf = topk_vals[0, rank].item() * 100.0
                if n_classes and cls_idx < n_classes:
                    cls_name = self.class_names[cls_idx]
                else:
                    cls_name = f"Class {cls_idx}"

                if conf < self.min_conf:
                    label_conf_str = f"Uncertain (<{self.min_conf:.1f}%)"
                else:
                    label_conf_str = f"{cls_name} ({conf:.1f}%)"
                result_items.append(label_conf_str)

            self.label_str = " | ".join(result_items)

            # Grad-CAM overlay if requested
            if self.do_gradcam and self.target_layer is not None:
                if self.cam_variant == "Grad-CAM++":
                    gradcam_obj = GradCAMPlusPlus(self.model, self.target_layer)
                elif self.cam_variant == "XGradCAM":
                    gradcam_obj = XGradCAM(self.model, self.target_layer)
                else:
                    gradcam_obj = GradCAM(self.model, self.target_layer)

                pred_class = topk_indices[0, 0].item()
                input_tensor.requires_grad = True
                cam_map = gradcam_obj.generate_cam(input_tensor, target_class=pred_class)
                cam_resized = cv2.resize(cam_map, (self.bgr_image.shape[1], self.bgr_image.shape[0]))
                overlay = overlay_cam_on_image(
                    self.bgr_image, cam_resized,
                    alpha=self.overlay_alpha,
                    colormap=self.colormap_id
                )
                self.overlay_result = overlay
            else:
                self.overlay_result = None

            self.done_signal.emit(self.overlay_result, self.bgr_image, self.label_str)

        except Exception as e:
            traceback_str = traceback.format_exc()
            self.progress_signal.emit(f"[ERROR] Single Inference failed: {e}\n{traceback_str}")
            self.done_signal.emit(None, self.bgr_image, f"Error: {e}")


class BatchInferenceThread(QThread):
    """
    Threaded batch inference.
    """
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str, dict)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params
        self._cancelled = False
        self.gradcam_cache = {}

    def run(self):
        try:
            self.run_batch_inference()
        except Exception as e:
            msg = f"[ERROR] Batch Inference Error: {e}\n{traceback.format_exc()}"
            self.log_signal.emit(msg)
            self.done_signal.emit("ERROR", {})
            return

    def run_batch_inference(self):
        logger = logging.getLogger(__name__)

        model = self.params["model"].to(self.params["device"]).eval()
        target_layer = self.params["target_layer"]
        device = self.params["device"]
        input_dir = self.params["input_dir"]
        gt_csv_path = self.params["gt_csv_path"]
        out_dir = self.params["out_dir"]
        transform = self.params["transform"]
        class_names = self.params["class_names"]
        top_k = self.params["top_k"]
        min_confidence = self.params["min_confidence"]
        do_gradcam = self.params["do_gradcam"]
        use_gradcam_pp = self.params["use_gradcam_pp"]
        overlay_alpha = self.params["overlay_alpha"]
        export_csv = self.params["export_csv"]
        average_mode = self.params["average_mode"]
        do_shap = self.params["do_shap"]
        shap_samples = self.params["shap_samples"]
        shap_bg = self.params["shap_bg"]
        batch_size = self.params.get("batch_size", 1)
        compute_advanced_metrics = self.params.get("compute_advanced_metrics", True)
        skip_shap_if_large = self.params.get("skip_shap_if_large", 1024)
        cam_variant = self.params.get("cam_variant", "Grad-CAM")
        colormap_id = self.params.get("colormap_id", cv2.COLORMAP_JET)

        gt_dict = self.load_ground_truth_dict(gt_csv_path)
        exts = ("*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg")
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

        if not image_files:
            self.log_signal.emit("[WARNING] No images found for batch inference.")
            self.done_signal.emit("DONE", {})
            return

        os.makedirs(out_dir, exist_ok=True)
        cam_dir = os.path.join(out_dir, "inference_cam")
        if do_gradcam:
            os.makedirs(cam_dir, exist_ok=True)

        txt_file = os.path.join(out_dir, "predictions.txt")
        csv_file = os.path.join(out_dir, "predictions.csv") if export_csv else None

        all_preds = []
        all_probs = []
        all_targets = []
        misclassified_samples = []

        if class_names:
            class_to_idx = {cname: i for i, cname in enumerate(class_names)}
            num_classes = len(class_names)
        else:
            class_to_idx = {}
            num_classes = None

        pdf_export = True

        log_file = os.path.join(out_dir, "batch_inference_log.txt")
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write("Batch Inference Log\n\n")

        with open(txt_file, "w", encoding="utf-8") as f_txt:
            if csv_file:
                csv_out = open(csv_file, "w", encoding="utf-8", newline="")
                csv_writer = csv.writer(csv_out)
                csv_writer.writerow(["image_path", "top_k_predictions", "gt_label", "inference_time_ms"])
            else:
                csv_out = None
                csv_writer = None

            for start_idx in range(0, len(image_files), batch_size):
                if self._cancelled:
                    self.log_signal.emit("[INFO] Batch inference was cancelled.")
                    break

                end_idx = start_idx + batch_size
                batch_paths = image_files[start_idx:end_idx]
                input_tensors = []
                valid_indices_in_batch = []

                for i, img_path in enumerate(batch_paths):
                    if self._cancelled:
                        self.log_signal.emit("[INFO] Batch inference was cancelled mid-batch.")
                        break
                    try:
                        bgr_img = cv2.imread(img_path)
                        if bgr_img is None:
                            self.log_with_timestamp(
                                f"[WARNING] Skipping unreadable file: {img_path}",
                                log_file
                            )
                            self.progress_signal.emit(start_idx + i + 1)
                            continue

                        # Check dimension if skipping SHAP for large images
                        if do_shap and (bgr_img.shape[0] > skip_shap_if_large or bgr_img.shape[1] > skip_shap_if_large):
                            self.log_signal.emit(f"[INFO] Skipping SHAP for large image: {img_path}")

                        tensor = self._transform_image(bgr_img, transform)
                        input_tensors.append(tensor)
                        valid_indices_in_batch.append(i)
                    except Exception as e:
                        self.log_with_timestamp(
                            f"[ERROR] Error reading/transforming {img_path}: {e}",
                            log_file
                        )
                        self.progress_signal.emit(start_idx + i + 1)
                        continue

                if not input_tensors:
                    continue

                input_batch = torch.stack(input_tensors).to(device)

                start_time = time.time()
                try:
                    with torch.no_grad():
                        if device.type == 'cuda':
                            try:
                                with torch.cuda.amp.autocast():
                                    logits = model(input_batch)
                            except RuntimeError as oom_e:
                                if "out of memory" in str(oom_e).lower():
                                    self.log_signal.emit("[ERROR] CUDA OOM in batch. Attempting to continue.")
                                    self.progress_signal.emit(start_idx + len(batch_paths))
                                    del input_batch
                                    torch.cuda.empty_cache()
                                    continue
                                else:
                                    raise
                        else:
                            logits = model(input_batch)
                        probs = torch.softmax(logits, dim=1)
                except Exception as e:
                    self.log_signal.emit(f"[ERROR] Failure during forward pass: {e}")
                    del input_batch
                    torch.cuda.empty_cache()
                    continue

                end_time = time.time()
                elapsed_ms_total = (end_time - start_time) * 1000.0
                elapsed_per_item = elapsed_ms_total / max(len(valid_indices_in_batch), 1)

                del input_batch, logits
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                for local_i, global_i in enumerate(valid_indices_in_batch):
                    idx_global = start_idx + global_i
                    img_path = batch_paths[global_i]
                    prob_vector = probs[local_i, :]
                    topk_vals, topk_indices = torch.topk(prob_vector, k=top_k, dim=0)

                    result_items = []
                    for rank in range(topk_vals.size(0)):
                        cls_idx = topk_indices[rank].item()
                        conf = topk_vals[rank].item() * 100.0
                        if class_names and cls_idx < len(class_names):
                            pred_label_str = class_names[cls_idx]
                        else:
                            pred_label_str = f"Class {cls_idx}"

                        if conf < min_confidence:
                            label_conf_str = f"Uncertain (<{min_confidence:.1f}%)"
                        else:
                            label_conf_str = f"{pred_label_str} ({conf:.1f}%)"
                        result_items.append(label_conf_str)

                    pred_class = topk_indices[0].item()
                    top1_conf = topk_vals[0].item()

                    all_preds.append(pred_class)
                    all_probs.append(prob_vector.cpu().numpy())

                    base_name = os.path.basename(img_path)
                    gt_label_str = None
                    gt_index = -1
                    if base_name in gt_dict:
                        gt_label_str = gt_dict[base_name]
                    elif img_path in gt_dict:
                        gt_label_str = gt_dict[img_path]

                    if gt_label_str is not None:
                        try:
                            gt_index = int(gt_label_str)
                        except ValueError:
                            gt_index = class_to_idx.get(gt_label_str, -1)
                    all_targets.append(gt_index)

                    topk_str = " | ".join(result_items)
                    line_txt = f"{img_path} => {topk_str} [Time: {elapsed_per_item:.1f} ms]"
                    f_txt.write(line_txt + "\n")
                    self.log_with_timestamp(line_txt, log_file, level="INFO")

                    if csv_writer:
                        csv_pred = "; ".join(result_items)
                        csv_writer.writerow([
                            img_path, csv_pred,
                            gt_label_str if gt_label_str else "",
                            f"{elapsed_per_item:.1f}"
                        ])

                    # Check misclassification
                    if gt_index != -1 and gt_index != pred_class:
                        if class_names and pred_class < len(class_names):
                            mc_pred_label = class_names[pred_class]
                        else:
                            mc_pred_label = f"Class {pred_class}"
                        misclassified_samples.append((
                            img_path,
                            gt_label_str if gt_label_str else str(gt_index),
                            mc_pred_label,
                            f"{top1_conf*100:.1f}%"
                        ))

                    # Grad-CAM if requested
                    if do_gradcam and target_layer is not None:
                        cam_key = (img_path, pred_class, cam_variant)
                        if cam_key in self.gradcam_cache:
                            cam_map = self.gradcam_cache[cam_key]
                        else:
                            if cam_variant == "Grad-CAM++":
                                gradcam_obj = GradCAMPlusPlus(model, target_layer)
                            elif cam_variant == "XGradCAM":
                                gradcam_obj = XGradCAM(model, target_layer)
                            else:
                                gradcam_obj = GradCAM(model, target_layer)

                            single_tensor = self._transform_image(cv2.imread(img_path), transform)
                            single_input = single_tensor.unsqueeze(0).to(device)
                            with torch.no_grad():
                                single_input.requires_grad = True

                            cam_map = gradcam_obj.generate_cam(single_input, target_class=pred_class)
                            self.gradcam_cache[cam_key] = cam_map

                            del single_input
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()

                        bgr_img = cv2.imread(img_path)
                        if bgr_img is not None:
                            cam_resized = cv2.resize(cam_map, (bgr_img.shape[1], bgr_img.shape[0]))
                            overlay_img = overlay_cam_on_image(
                                bgr_img, cam_resized,
                                alpha=overlay_alpha,
                                colormap=colormap_id
                            )
                            out_path = os.path.join(cam_dir, f"CAM_{os.path.basename(img_path)}")
                            cv2.imwrite(out_path, overlay_img)

                    self.progress_signal.emit(idx_global + 1)

            if csv_out:
                csv_out.close()

        if misclassified_samples:
            mc_file = os.path.join(out_dir, "misclassified.csv")
            with open(mc_file, "w", encoding="utf-8", newline="") as f_mc:
                writer = csv.writer(f_mc)
                writer.writerow(["image_path", "ground_truth", "predicted", "pred_confidence"])
                for row in misclassified_samples:
                    writer.writerow(row)

        metrics_dict = {}
        valid_inds = [i for i, t in enumerate(all_targets) if t != -1]
        if valid_inds and compute_advanced_metrics:
            y_true = [all_targets[i] for i in valid_inds]
            y_pred = [all_preds[i] for i in valid_inds]
            y_prob = [all_probs[i] for i in valid_inds]

            cm = confusion_matrix(y_true, y_pred)
            cm_path = os.path.join(out_dir, "confusion_matrix.txt")
            with open(cm_path, "w") as f_cm:
                f_cm.write("Confusion Matrix:\n")
                f_cm.write(str(cm))
                f_cm.write("\n")

            if class_names and max(y_true) < len(class_names):
                cls_rep_str = classification_report(y_true, y_pred, target_names=class_names)
                cls_rep_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            else:
                cls_rep_str = classification_report(y_true, y_pred)
                cls_rep_dict = classification_report(y_true, y_pred, output_dict=True)

            cr_path = os.path.join(out_dir, "classification_report.txt")
            with open(cr_path, "w") as f_rep:
                f_rep.write("Classification Report:\n")
                f_rep.write(cls_rep_str)
                f_rep.write("\n")

            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            if class_names and max(y_true) < len(class_names):
                tick_marks = range(len(class_names))
                plt.xticks(tick_marks, class_names, rotation=45, ha="right")
                plt.yticks(tick_marks, class_names)
            plt.tight_layout()
            cm_png = os.path.join(out_dir, "confusion_matrix.png")
            plt.savefig(cm_png)
            if pdf_export:
                plt.savefig(os.path.join(out_dir, "confusion_matrix.pdf"))
            plt.close()

            if HAVE_PLOTLY and class_names and max(y_true) < len(class_names):
                self.plotly_confusion_matrix(cm, class_names, out_dir)

            top1_conf_all = [max(prob_vec) for prob_vec in y_prob]
            plt.figure()
            plt.hist(top1_conf_all, bins=20, range=(0, 1), color='green', alpha=0.7)
            plt.title("Top-1 Confidence Distribution")
            plt.xlabel("Confidence")
            plt.ylabel("Frequency")
            hist_png = os.path.join(out_dir, "confidence_histogram.png")
            plt.savefig(hist_png)
            if pdf_export:
                plt.savefig(os.path.join(out_dir, "confidence_histogram.pdf"))
            plt.close()

            roc_auc_overall = None
            pr_auc_val = None
            mcc_val = matthews_corrcoef(y_true, y_pred)

            if num_classes == 2:
                y_prob_pos = [p[1] for p in y_prob]
                fpr, tpr, _ = roc_curve(y_true, y_prob_pos, pos_label=1)
                roc_auc_val = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={roc_auc_val:.3f}")
                plt.plot([0, 1], [0, 1], "r--")
                plt.title("ROC Curve (Binary)")
                plt.legend()
                roc_png = os.path.join(out_dir, "roc_curve.png")
                plt.savefig(roc_png)
                if pdf_export:
                    plt.savefig(os.path.join(out_dir, "roc_curve.pdf"))
                plt.close()

                if HAVE_PLOTLY:
                    self.plotly_roc_curve(fpr, tpr, roc_auc_val, out_dir)

                precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
                avg_prec = average_precision_score(y_true, y_prob_pos)
                plt.figure()
                plt.plot(recall, precision, label=f"AP={avg_prec:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve (Binary)")
                plt.legend()
                pr_png = os.path.join(out_dir, "pr_curve.png")
                plt.savefig(pr_png)
                if pdf_export:
                    plt.savefig(os.path.join(out_dir, "pr_curve.pdf"))
                plt.close()

                fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob_pos, n_bins=10)
                plt.figure()
                plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
                plt.plot([0, 1], [0, 1], "r--", label="Perfect")
                plt.title("Calibration Curve (Binary)")
                plt.xlabel("Mean Predicted Value")
                plt.ylabel("Fraction of Positives")
                plt.legend()
                cal_png = os.path.join(out_dir, "calibration_curve.png")
                plt.savefig(cal_png)
                if pdf_export:
                    plt.savefig(os.path.join(out_dir, "calibration_curve.pdf"))
                plt.close()

                try:
                    roc_auc_overall = roc_auc_score(y_true, y_prob_pos)
                except:
                    roc_auc_overall = None
                pr_auc_val = avg_prec

            elif num_classes and num_classes > 2:
                roc_auc_overall = self.plot_multiclass_roc(
                    np.array(y_true),
                    np.array(y_prob),
                    out_dir,
                    average=average_mode,
                    class_names=class_names,
                    pdf_export=pdf_export
                )
                pr_auc_val = self.plot_multiclass_pr(
                    np.array(y_true),
                    np.array(y_prob),
                    out_dir,
                    average=average_mode,
                    class_names=class_names,
                    pdf_export=pdf_export
                )

            if do_shap:
                if HAVE_SHAP:
                    shap_files = [image_files[i] for i in valid_inds]
                    shap_file_out = os.path.join(out_dir, "shap_summary.png")
                    try:
                        self.run_shap_analysis(
                            model, device, shap_files, transform,
                            shap_samples, shap_bg, shap_file_out, skip_shap_if_large
                        )
                    except Exception as e:
                        self.log_signal.emit(f"[WARNING] SHAP error: {e}")
                else:
                    self.log_signal.emit("[WARNING] SHAP not available. Skipping SHAP analysis.")

            metrics_dict["confusion_matrix"] = cm
            metrics_dict["class_report"] = cls_rep_str

            self.save_metrics_to_json(cm, cls_rep_dict, mcc_val, roc_auc_overall, pr_auc_val, out_dir)

        self.log_signal.emit("[INFO] Batch Inference Done.")
        self.done_signal.emit(out_dir, metrics_dict)

    def _transform_image(self, bgr_img: np.ndarray, transform: A.Compose) -> torch.Tensor:
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        aug = transform(image=rgb_img)
        return aug["image"]

    def load_ground_truth_dict(self, csv_file: str) -> dict:
        gt_dict = {}
        if not csv_file or not os.path.isfile(csv_file):
            return gt_dict
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue
                    img_name, lbl_str = parts[0].strip(), parts[1].strip()
                    gt_dict[img_name] = lbl_str
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Error reading CSV: {e}")
        return gt_dict

    def plot_multiclass_roc(self, y_true, y_prob, out_dir, average="macro", class_names=None, pdf_export=False):
        num_classes = y_prob.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        try:
            roc_auc = roc_auc_score(y_true_bin, y_prob, average=average, multi_class="ovr")
        except:
            roc_auc = None

        plt.figure()
        for c in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
            if class_names and c < len(class_names):
                plt.plot(fpr, tpr, label=f"{class_names[c]}")
            else:
                plt.plot(fpr, tpr, label=f"Class {c}")
        plt.plot([0, 1], [0, 1], "r--")
        if roc_auc is not None:
            plt.title(f"Multi-class ROC ({average} avg AUC={roc_auc:.3f})")
        else:
            plt.title("Multi-class ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        roc_png = os.path.join(out_dir, "roc_curve.png")
        plt.savefig(roc_png)
        if pdf_export:
            plt.savefig(os.path.join(out_dir, "roc_curve.pdf"))
        plt.close()
        return roc_auc

    def plot_multiclass_pr(self, y_true, y_prob, out_dir, average="macro", class_names=None, pdf_export=False):
        num_classes = y_prob.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        try:
            pr_auc = average_precision_score(y_true_bin, y_prob, average=average)
        except:
            pr_auc = None

        plt.figure()
        for c in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, c], y_prob[:, c])
            if class_names and c < len(class_names):
                label_str = f"{class_names[c]}"
            else:
                label_str = f"Class {c}"
            plt.plot(recall, precision, label=label_str)

        if pr_auc is not None:
            plt.title(f"Multi-class PR Curve ({average} AP={pr_auc:.3f})")
        else:
            plt.title("Multi-class PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        pr_png = os.path.join(out_dir, "pr_curve.png")
        plt.savefig(pr_png)
        if pdf_export:
            plt.savefig(os.path.join(out_dir, "pr_curve.pdf"))
        plt.close()
        return pr_auc

    def run_shap_analysis(
            self, model, device, image_files, transform,
            shap_samples, shap_bg, shap_outfile, skip_shap_if_large
    ):
        sample_files = []
        for fimg in image_files[:shap_samples]:
            bgr = cv2.imread(fimg)
            if bgr is None:
                continue
            if bgr.shape[0] > skip_shap_if_large or bgr.shape[1] > skip_shap_if_large:
                continue
            sample_files.append(fimg)

        background_files = sample_files[:shap_bg] if shap_bg > 0 else sample_files[:1]

        shap_list = []
        for fpath in sample_files:
            bgr = cv2.imread(fpath)
            if bgr is None:
                continue
            shap_list.append(self._transform_image(bgr, transform))

        bg_list = []
        for fpath in background_files:
            bgr = cv2.imread(fpath)
            if bgr is None:
                continue
            bg_list.append(self._transform_image(bgr, transform))

        if not shap_list:
            self.log_signal.emit("[WARNING] No valid images for SHAP.")
            return

        shap_tensor = torch.stack(shap_list, dim=0).to(device)
        if not bg_list:
            bg_list = shap_list[:1]
        bg_tensor = torch.stack(bg_list, dim=0).to(device)

        self.log_signal.emit("[INFO] Running SHAP DeepExplainer...")
        explainer = shap.DeepExplainer(model, bg_tensor)
        shap_values = explainer.shap_values(shap_tensor)

        plt.figure()
        if isinstance(shap_values, list) and len(shap_values) > 1:
            import torch
            st = [torch.tensor(sv) for sv in shap_values]
            stacked = torch.stack(st, dim=0)
            mean_across_classes = stacked.mean(dim=0)
            shap.image_plot(mean_across_classes.numpy(), shap_tensor.cpu().numpy())
        else:
            if isinstance(shap_values, list):
                sv = shap_values[0]
            else:
                sv = shap_values
            shap.image_plot(sv, shap_tensor.cpu().numpy())

        plt.savefig(shap_outfile, bbox_inches="tight")
        plt.close()
        self.log_signal.emit(f"[INFO] SHAP summary saved to {shap_outfile}")

        del shap_tensor, bg_tensor, shap_values
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    def save_metrics_to_json(self, cm, cls_rep_dict, mcc_val, roc_auc_val, pr_auc_val, out_dir):
        data = {
            "confusion_matrix": cm.tolist(),
            "classification_report": cls_rep_dict,
            "mcc": mcc_val,
            "roc_auc": roc_auc_val,
            "pr_auc": pr_auc_val
        }
        out_path = os.path.join(out_dir, "metrics.json")
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

    def plotly_confusion_matrix(self, cm, class_names, out_dir):
        if not HAVE_PLOTLY:
            return
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=class_names, y=class_names
        )
        fig.update_layout(title="Confusion Matrix (Interactive)")
        plot_file = os.path.join(out_dir, "confusion_matrix_plotly.html")
        fig.write_html(plot_file)

    def plotly_roc_curve(self, fpr, tpr, roc_auc_val, out_dir):
        if not HAVE_PLOTLY:
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc_val:.3f})'))
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                      line=dict(color='red', dash='dash'))
        fig.update_layout(
            title="ROC Curve (Interactive)",
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        plot_file = os.path.join(out_dir, "roc_curve_plotly.html")
        fig.write_html(plot_file)

    def log_with_timestamp(self, message, logfile, level="INFO"):
        now_str = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        log_str = f"[{now_str}][{level}] {message}"
        self.log_signal.emit(log_str)
        with open(logfile, "a", encoding="utf-8") as lf:
            lf.write(log_str + "\n")

    def cancel(self):
        self._cancelled = True


class Plugin(BasePlugin):
    """
    Updated plugin for post-training evaluation & inference:
    - Model Zoo support
    - Transform pipeline (basic or advanced text-based)
    - Memory usage limit
    - Single-image inference in thread
    - SHAP & Grad-CAM
    - UI enhancements (open output folder, advanced metrics, etc.)
    - Extended class names
    - Logging via Python's logging module
    - Additional Grad-CAM variants
    - Unit test placeholders
    """
    def __init__(self):
        super().__init__()
        self.plugin_name = "Evaluation/Inference"

        self.widget_main = None
        self.ckpt_edit = None
        self.infer_img_edit = None
        self.infer_result_label = None
        self.label_infer_orig = None
        self.label_infer_cam = None
        self.inference_input_dir = None
        self.progress_bar = None

        self.gt_csv_edit = None
        self.cb_gradcam_pp = None
        self.cb_enable_shap = None
        self.cbx_roc_average = None
        self.spin_shap_samples = None
        self.spin_shap_bg_samples = None
        self.dspin_overlay_alpha = None

        self.resize_check = None
        self.resize_width_spin = None
        self.resize_height_spin = None
        self.mean_spin = []
        self.std_spin = []
        self.maintain_ar_check = None

        self.spin_top_k = None
        self.dspin_min_confidence = None
        self.cb_batch_gradcam = None
        self.cb_export_csv = None
        self.cb_advanced_metrics = None

        self.text_log = None
        self.layer_combo = None
        self.available_layers = []
        self.device_combo = None

        self.spin_batch_size = None

        self.model = None
        self.target_layer = None
        self.class_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_thread = None
        self.single_thread = None

        self.model_zoo_combo = None
        self.btn_load_zoo_model = None
        self.gradcam_variant_combo = None
        self.colormap_combo = None
        self.btn_cancel_batch = None
        self.btn_load_class_names = None
        self.spin_gpu_usage = None
        self.check_adv_transform = None
        self.adv_transform_textedit = None
        self.spin_skip_shap_if_large = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.last_single_inference_img = None
        self.last_single_inference_overlay = None
        self.last_single_inference_label_str = None

    def create_tab(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        container_widget = QWidget()
        main_layout = QVBoxLayout(container_widget)

        help_label = QLabel(
            "Evaluation/Inference Plugin\n"
            "1. Load a custom checkpoint or select a pretrained model.\n"
            "2. Adjust transforms & Grad-CAM settings.\n"
            "3. Run single-image or batch inference.\n"
            "Hover over elements for tooltips."
        )
        help_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(help_label)

        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        text_edit_handler = QTextEditLogger(self.text_log)
        text_edit_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        text_edit_handler.setFormatter(formatter)
        self.logger.addHandler(text_edit_handler)

        main_layout.addLayout(self._create_ckpt_section())
        main_layout.addLayout(self._create_model_zoo_section())
        main_layout.addLayout(self._create_transform_section())
        main_layout.addLayout(self._create_single_infer_section())
        main_layout.addLayout(self._create_batch_infer_section())

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        main_layout.addWidget(self.text_log)

        scroll_area.setWidget(container_widget)
        return scroll_area

    def _create_ckpt_section(self) -> QHBoxLayout:
        h_ckpt = QHBoxLayout()
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setToolTip("Path to your trained .ckpt file.")
        btn_ckpt = QPushButton("Browse Checkpoint...")
        btn_ckpt.clicked.connect(self.browse_ckpt)
        h_ckpt.addWidget(QLabel("Checkpoint:"))
        h_ckpt.addWidget(self.ckpt_edit)
        h_ckpt.addWidget(btn_ckpt)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto (cuda if available)", "CPU", "CUDA"])
        self.device_combo.setToolTip("Select inference device.")
        lbl_device = QLabel("Device:")
        h_ckpt.addWidget(lbl_device)
        h_ckpt.addWidget(self.device_combo)

        lbl_gpu_usage = QLabel("Max GPU Usage %:")
        self.spin_gpu_usage = QSpinBox()
        self.spin_gpu_usage.setRange(1, 100)
        self.spin_gpu_usage.setValue(100)
        self.spin_gpu_usage.setToolTip("Limit GPU memory usage fraction.")
        h_ckpt.addWidget(lbl_gpu_usage)
        h_ckpt.addWidget(self.spin_gpu_usage)

        self.btn_load_class_names = QPushButton("Load Class Names from File...")
        self.btn_load_class_names.setToolTip("Optional: load class names from a text or CSV.")
        self.btn_load_class_names.clicked.connect(self.load_class_names_file)
        h_ckpt.addWidget(self.btn_load_class_names)

        return h_ckpt

    def _create_model_zoo_section(self) -> QHBoxLayout:
        h_zoo = QHBoxLayout()
        self.model_zoo_combo = QComboBox()
        self.model_zoo_combo.setToolTip("Select a pretrained model from torchvision model zoo.")
        if HAVE_TORCHVISION and ZOO_MODELS:
            self.model_zoo_combo.addItems(sorted(ZOO_MODELS.keys()))
        else:
            self.model_zoo_combo.addItem("No torchvision available")

        self.btn_load_zoo_model = QPushButton("Load Pretrained")
        self.btn_load_zoo_model.clicked.connect(self.load_zoo_model)
        h_zoo.addWidget(QLabel("Pretrained Model:"))
        h_zoo.addWidget(self.model_zoo_combo)
        h_zoo.addWidget(self.btn_load_zoo_model)
        return h_zoo

    def _create_transform_section(self) -> QVBoxLayout:
        transform_box = QVBoxLayout()

        self.resize_check = QCheckBox("Resize Images?")
        self.resize_check.setChecked(True)
        transform_box.addWidget(self.resize_check)

        self.maintain_ar_check = QCheckBox("Maintain Aspect Ratio (pad)")
        self.maintain_ar_check.setChecked(False)
        transform_box.addWidget(self.maintain_ar_check)

        h_resize = QHBoxLayout()
        h_resize.addWidget(QLabel("Width:"))
        self.resize_width_spin = QSpinBox()
        self.resize_width_spin.setRange(1, 5000)
        self.resize_width_spin.setValue(224)
        h_resize.addWidget(self.resize_width_spin)
        h_resize.addWidget(QLabel("Height:"))
        self.resize_height_spin = QSpinBox()
        self.resize_height_spin.setRange(1, 5000)
        self.resize_height_spin.setValue(224)
        h_resize.addWidget(self.resize_height_spin)
        transform_box.addLayout(h_resize)

        norm_layout = QHBoxLayout()
        norm_layout.addWidget(QLabel("Mean:"))
        for _ in range(3):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.01)
            sp.setValue(0.5)
            self.mean_spin.append(sp)
            norm_layout.addWidget(sp)

        norm_layout.addWidget(QLabel("Std:"))
        for _ in range(3):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 10.0)
            sp.setSingleStep(0.01)
            sp.setValue(0.5)
            self.std_spin.append(sp)
            norm_layout.addWidget(sp)

        transform_box.addLayout(norm_layout)

        self.check_adv_transform = QCheckBox("Use advanced text-based transform config?")
        self.check_adv_transform.setChecked(False)
        transform_box.addWidget(self.check_adv_transform)

        self.adv_transform_textedit = QTextEdit()
        self.adv_transform_textedit.setPlaceholderText(
            "Example:\nA.Compose([\n    A.RandomCrop(200, 200),\n    A.HorizontalFlip(p=0.5),\n    A.Normalize(...),\n    ToTensorV2(),\n])"
        )
        self.adv_transform_textedit.setFixedHeight(100)
        transform_box.addWidget(self.adv_transform_textedit)

        return transform_box

    def _create_single_infer_section(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        h_img = QHBoxLayout()
        self.infer_img_edit = QLineEdit()
        btn_img = QPushButton("Browse Image...")
        btn_img.clicked.connect(self.browse_infer_image)
        h_img.addWidget(QLabel("Test Image:"))
        h_img.addWidget(self.infer_img_edit)
        h_img.addWidget(btn_img)
        layout.addLayout(h_img)

        gradcam_variant_layout = QHBoxLayout()
        gradcam_variant_layout.addWidget(QLabel("Grad-CAM Variant:"))
        self.gradcam_variant_combo = QComboBox()
        self.gradcam_variant_combo.addItems(["Grad-CAM", "Grad-CAM++", "XGradCAM"])
        gradcam_variant_layout.addWidget(self.gradcam_variant_combo)
        layout.addLayout(gradcam_variant_layout)

        self.cb_gradcam_pp = QCheckBox("Use Grad-CAM++ (legacy toggle)")
        layout.addWidget(self.cb_gradcam_pp)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Grad-CAM Overlay Alpha:"))
        self.dspin_overlay_alpha = QDoubleSpinBox()
        self.dspin_overlay_alpha.setRange(0, 1)
        self.dspin_overlay_alpha.setValue(0.5)
        alpha_layout.addWidget(self.dspin_overlay_alpha)
        layout.addLayout(alpha_layout)

        h_cmap = QHBoxLayout()
        h_cmap.addWidget(QLabel("Grad-CAM Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["JET", "HOT", "BONE", "HSV"])
        h_cmap.addWidget(self.colormap_combo)
        layout.addLayout(h_cmap)

        tk_layout = QHBoxLayout()
        tk_layout.addWidget(QLabel("Top-k:"))
        self.spin_top_k = QSpinBox()
        self.spin_top_k.setRange(1, 10)
        self.spin_top_k.setValue(1)
        tk_layout.addWidget(self.spin_top_k)

        tk_layout.addWidget(QLabel("Min Confidence (%):"))
        self.dspin_min_confidence = QDoubleSpinBox()
        self.dspin_min_confidence.setRange(0, 100)
        self.dspin_min_confidence.setValue(10)
        self.dspin_min_confidence.setSingleStep(1)
        tk_layout.addWidget(self.dspin_min_confidence)
        layout.addLayout(tk_layout)

        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Target Layer:"))
        self.layer_combo = QComboBox()
        layer_layout.addWidget(self.layer_combo)
        layout.addLayout(layer_layout)

        btn_infer = QPushButton("Run Inference (Single)")
        btn_infer.clicked.connect(self.run_inference)
        layout.addWidget(btn_infer)

        btn_export_single = QPushButton("Export Single Inference Results")
        btn_export_single.clicked.connect(self.export_single_inference_result)
        layout.addWidget(btn_export_single)

        self.infer_result_label = QLabel("Result: ")
        layout.addWidget(self.infer_result_label)

        h_disp = QHBoxLayout()
        self.label_infer_orig = QLabel("Original")
        self.label_infer_orig.setFixedSize(300, 300)
        self.label_infer_orig.setAlignment(Qt.AlignCenter)
        self.label_infer_cam = QLabel("Grad-CAM/++")
        self.label_infer_cam.setFixedSize(300, 300)
        self.label_infer_cam.setAlignment(Qt.AlignCenter)
        h_disp.addWidget(self.label_infer_orig)
        h_disp.addWidget(self.label_infer_cam)
        layout.addLayout(h_disp)

        return layout

    def _create_batch_infer_section(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        batch_inf_layout = QHBoxLayout()
        self.inference_input_dir = QLineEdit()
        btn_infer_dir = QPushButton("Browse Folder...")
        btn_infer_dir.clicked.connect(self.browse_inference_folder)
        batch_inf_layout.addWidget(QLabel("Inference Folder:"))
        batch_inf_layout.addWidget(self.inference_input_dir)
        batch_inf_layout.addWidget(btn_infer_dir)
        layout.addLayout(batch_inf_layout)

        h_gtcsv = QHBoxLayout()
        self.gt_csv_edit = QLineEdit()
        btn_gtcsv = QPushButton("Browse GT CSV...")
        btn_gtcsv.clicked.connect(self.browse_gt_csv)  # <--- ADDED HERE
        h_gtcsv.addWidget(QLabel("GT CSV (optional):"))
        h_gtcsv.addWidget(self.gt_csv_edit)
        h_gtcsv.addWidget(btn_gtcsv)
        layout.addLayout(h_gtcsv)

        self.cb_advanced_metrics = QCheckBox("Compute Advanced Metrics (ROC, PR, etc.)")
        self.cb_advanced_metrics.setChecked(True)
        layout.addWidget(self.cb_advanced_metrics)

        roc_layout = QHBoxLayout()
        roc_layout.addWidget(QLabel("ROC/PR Average Mode:"))
        self.cbx_roc_average = QComboBox()
        self.cbx_roc_average.addItems(["macro", "micro", "weighted"])
        self.cbx_roc_average.setCurrentText("macro")
        roc_layout.addWidget(self.cbx_roc_average)
        layout.addLayout(roc_layout)

        self.cb_enable_shap = QCheckBox("Enable SHAP analysis")
        layout.addWidget(self.cb_enable_shap)

        shap_layout = QHBoxLayout()
        shap_layout.addWidget(QLabel("SHAP Samples:"))
        self.spin_shap_samples = QSpinBox()
        self.spin_shap_samples.setRange(1, 9999)
        self.spin_shap_samples.setValue(20)
        shap_layout.addWidget(self.spin_shap_samples)

        shap_layout.addWidget(QLabel("SHAP BG Samples:"))
        self.spin_shap_bg_samples = QSpinBox()
        self.spin_shap_bg_samples.setRange(0, 9999)
        self.spin_shap_bg_samples.setValue(5)
        shap_layout.addWidget(self.spin_shap_bg_samples)
        layout.addLayout(shap_layout)

        h_skip_shap = QHBoxLayout()
        h_skip_shap.addWidget(QLabel("Skip SHAP if dimension >"))
        self.spin_skip_shap_if_large = QSpinBox()
        self.spin_skip_shap_if_large.setRange(256, 9999)
        self.spin_skip_shap_if_large.setValue(1024)
        h_skip_shap.addWidget(self.spin_skip_shap_if_large)
        h_skip_shap.addWidget(QLabel("(either width or height)"))
        layout.addLayout(h_skip_shap)

        self.cb_batch_gradcam = QCheckBox("Generate Grad-CAM in Batch Inference?")
        self.cb_batch_gradcam.setChecked(True)
        layout.addWidget(self.cb_batch_gradcam)

        self.cb_export_csv = QCheckBox("Export CSV for predictions?")
        layout.addWidget(self.cb_export_csv)

        h_batch_size = QHBoxLayout()
        h_batch_size.addWidget(QLabel("Batch Size:"))
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 256)
        self.spin_batch_size.setValue(1)
        h_batch_size.addWidget(self.spin_batch_size)
        layout.addLayout(h_batch_size)

        h_btns = QHBoxLayout()
        btn_batch_infer = QPushButton("Run Batch Inference")
        btn_batch_infer.clicked.connect(self.start_batch_inference)
        h_btns.addWidget(btn_batch_infer)

        self.btn_cancel_batch = QPushButton("Cancel Batch")
        self.btn_cancel_batch.clicked.connect(self.cancel_batch_inference)
        h_btns.addWidget(self.btn_cancel_batch)

        btn_open_output = QPushButton("Open Output Folder")
        btn_open_output.clicked.connect(self.open_output_folder)
        h_btns.addWidget(btn_open_output)

        btn_open_plotly = QPushButton("Open Plotly HTMLs")
        btn_open_plotly.clicked.connect(self.open_plotly_html_results)
        h_btns.addWidget(btn_open_plotly)

        layout.addLayout(h_btns)
        return layout

    def browse_inference_folder(self):
        folder = QFileDialog.getExistingDirectory(None, "Select Inference Folder")
        if folder:
            self.inference_input_dir.setText(folder)

    def browse_gt_csv(self):
        """
        Opens a file dialog to select a ground-truth CSV file, sets path to gt_csv_edit.
        """
        fpath, _ = QFileDialog.getOpenFileName(None, "Select GT CSV", filter="CSV Files (*.csv)")
        if fpath:
            self.gt_csv_edit.setText(fpath)

    def load_zoo_model(self):
        if not HAVE_TORCHVISION or not ZOO_MODELS:
            QMessageBox.warning(None, "Error", "No torchvision / model zoo support available.")
            return
        model_name = self.model_zoo_combo.currentText()
        if model_name not in ZOO_MODELS:
            QMessageBox.warning(None, "Error", f"Invalid model name: {model_name}")
            return
        self.device = self.get_selected_device()

        fraction = self.spin_gpu_usage.value() / 100.0
        if self.device.type == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(fraction)
            except AttributeError:
                self.logger.warning("set_per_process_memory_fraction not supported.")

        try:
            self.model = ZOO_MODELS[model_name](pretrained=True)
            self.model.eval().to(self.device)
            self.class_names = None

            self.available_layers = self.discover_conv_layers(self.model)
            self.layer_combo.clear()
            for name_layer in self.available_layers:
                self.layer_combo.addItem(name_layer[0])
            if self.available_layers:
                self.layer_combo.setCurrentIndex(len(self.available_layers) - 1)

            QMessageBox.information(
                None,
                "Model Loaded",
                f"Pretrained model '{model_name}' loaded.\nFound {len(self.available_layers)} candidate layers."
            )
        except Exception as e:
            QMessageBox.critical(None, "Load Error", f"Failed to load zoo model '{model_name}': {e}")
            self.logger.error(f"Failed to load zoo model '{model_name}': {e}")

    def get_selected_device(self) -> torch.device:
        choice = self.device_combo.currentText()
        if choice == "CPU":
            return torch.device("cpu")
        elif choice == "CUDA":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def browse_ckpt(self):
        fpath, _ = QFileDialog.getOpenFileName(None, "Select Checkpoint File", filter="*.ckpt")
        if fpath:
            self.ckpt_edit.setText(fpath)
            try:
                self.device = self.get_selected_device()
                fraction = self.spin_gpu_usage.value() / 100.0
                if self.device.type == "cuda":
                    try:
                        torch.cuda.set_per_process_memory_fraction(fraction)
                    except AttributeError:
                        self.logger.warning("set_per_process_memory_fraction not supported.")

                self.model = MaidClassifier.load_from_checkpoint(fpath)
                self.model.to(self.device).eval()

                if hasattr(self.model, "class_names"):
                    self.class_names = self.model.class_names
                else:
                    self.class_names = None

                self.available_layers = self.discover_conv_layers(self.model.backbone)
                self.layer_combo.clear()
                for name_layer in self.available_layers:
                    self.layer_combo.addItem(name_layer[0])
                if self.available_layers:
                    self.layer_combo.setCurrentIndex(len(self.available_layers) - 1)

                QMessageBox.information(
                    None,
                    "Model Loaded",
                    f"Model loaded from:\n{fpath}\nFound {len(self.available_layers)} candidate layers."
                )
            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(None, "Load Error", f"Failed to load model:\n{e}")
                self.logger.error(f"Failed to load checkpoint: {e}")

    def discover_conv_layers(self, net: torch.nn.Module) -> list:
        layers_list = []

        def recurse_layers(parent, parent_name):
            for name, module in parent.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                has_param = list(module.parameters())
                if has_param:
                    layers_list.append((full_name, module))
                recurse_layers(module, full_name)

        recurse_layers(net, "")
        return layers_list

    def browse_infer_image(self):
        fpath, _ = QFileDialog.getOpenFileName(None, "Select an Image", filter="Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if fpath:
            self.infer_img_edit.setText(fpath)

    def run_inference(self):
        if not self.model:
            QMessageBox.warning(None, "Error", "Model not loaded.")
            return
        img_path = self.infer_img_edit.text().strip()
        if not os.path.isfile(img_path):
            QMessageBox.warning(None, "Error", "Invalid image path.")
            return

        idx_layer = self.layer_combo.currentIndex()
        if idx_layer < 0 or idx_layer >= len(self.available_layers):
            QMessageBox.warning(None, "Error", "No valid Grad-CAM target layer selected.")
            return
        self.target_layer = self.available_layers[idx_layer][1]

        try:
            bgr_img = cv2.imread(img_path)
            if bgr_img is None:
                QMessageBox.warning(None, "Error", "Failed to read image.")
                return

            pm_orig = convert_to_qpixmap(bgr_img)
            self.label_infer_orig.setPixmap(pm_orig.scaled(
                self.label_infer_orig.width(),
                self.label_infer_orig.height(),
                Qt.KeepAspectRatio
            ))

            transform = self.build_transform()

            colormap_id = self.get_colormap_id()
            gradcam_variant = self.gradcam_variant_combo.currentText()

            self.single_thread = SingleInferenceThread(
                bgr_image=bgr_img,
                model=self.model,
                device=self.get_selected_device(),
                transform=transform,
                cam_variant=gradcam_variant,
                target_layer=self.target_layer,
                top_k=self.spin_top_k.value(),
                min_conf=self.dspin_min_confidence.value(),
                overlay_alpha=self.dspin_overlay_alpha.value(),
                colormap_id=colormap_id,
                do_gradcam=True,
                class_names=self.class_names
            )
            self.single_thread.progress_signal.connect(self._on_single_inference_progress)
            self.single_thread.done_signal.connect(self._on_single_inference_done)
            self.single_thread.start()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Inference Error", str(e))
            self.logger.error(f"Inference error: {e}")

    def _on_single_inference_progress(self, msg: str):
        self.logger.info(msg)

    def _on_single_inference_done(self, overlay, original, label_str):
        self.infer_result_label.setText(f"Result: {label_str}")
        self.last_single_inference_img = original.copy()
        self.last_single_inference_overlay = overlay.copy() if overlay is not None else None
        self.last_single_inference_label_str = label_str

        if overlay is not None:
            pm_cam = convert_to_qpixmap(overlay)
            self.label_infer_cam.setPixmap(pm_cam.scaled(
                self.label_infer_cam.width(),
                self.label_infer_cam.height(),
                Qt.KeepAspectRatio
            ))
        else:
            self.label_infer_cam.setPixmap(QPixmap())

    def start_batch_inference(self):
        if not self.model:
            QMessageBox.warning(None, "Error", "Model not loaded.")
            return

        input_dir = self.inference_input_dir.text().strip()
        if not os.path.isdir(input_dir):
            QMessageBox.warning(None, "Error", "Invalid folder.")
            return

        self.device = self.get_selected_device()
        fraction = self.spin_gpu_usage.value() / 100.0
        if self.device.type == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(fraction)
            except AttributeError:
                self.logger.warning("set_per_process_memory_fraction not supported.")

        transform = self.build_transform()

        idx_layer = self.layer_combo.currentIndex()
        if 0 <= idx_layer < len(self.available_layers):
            self.target_layer = self.available_layers[idx_layer][1]
        else:
            self.target_layer = None

        out_dir = os.path.join(input_dir, "inference_results")
        os.makedirs(out_dir, exist_ok=True)

        cam_variant = self.gradcam_variant_combo.currentText()
        if self.cb_gradcam_pp.isChecked():
            cam_variant = "Grad-CAM++"

        params = {
            "model": self.model,
            "target_layer": self.target_layer,
            "device": self.device,
            "input_dir": input_dir,
            "gt_csv_path": self.gt_csv_edit.text().strip(),
            "out_dir": out_dir,
            "transform": transform,
            "class_names": self.class_names,
            "top_k": self.spin_top_k.value(),
            "min_confidence": self.dspin_min_confidence.value(),
            "do_gradcam": self.cb_batch_gradcam.isChecked(),
            "use_gradcam_pp": self.cb_gradcam_pp.isChecked(),
            "overlay_alpha": self.dspin_overlay_alpha.value(),
            "export_csv": self.cb_export_csv.isChecked(),
            "average_mode": self.cbx_roc_average.currentText(),
            "do_shap": self.cb_enable_shap.isChecked(),
            "shap_samples": self.spin_shap_samples.value(),
            "shap_bg": self.spin_shap_bg_samples.value(),
            "batch_size": self.spin_batch_size.value(),
            "compute_advanced_metrics": self.cb_advanced_metrics.isChecked(),
            "skip_shap_if_large": self.spin_skip_shap_if_large.value(),
            "cam_variant": cam_variant,
            "colormap_id": self.get_colormap_id()
        }

        self.batch_thread = BatchInferenceThread(params=params)
        self.batch_thread.progress_signal.connect(self.update_progress)
        self.batch_thread.log_signal.connect(self._append_log_direct)
        self.batch_thread.done_signal.connect(self.batch_inference_done)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 0)
        self.batch_thread.start()

    def cancel_batch_inference(self):
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.cancel()
            self.logger.info("[INFO] Cancel requested.")

    def update_progress(self, val: int):
        current_max = self.progress_bar.maximum()
        if current_max == 0 or val > current_max:
            self.progress_bar.setRange(0, val)
        self.progress_bar.setValue(val)

    def _append_log_direct(self, msg: str):
        self.text_log.append(msg)
        self.text_log.ensureCursorVisible()

    def batch_inference_done(self, results_dir: str, metrics: dict):
        self.progress_bar.setVisible(False)
        if results_dir == "ERROR":
            self.logger.error("[ERROR] Batch inference encountered an error.")
            return
        self.logger.info("[INFO] Batch inference completed.")
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            self.logger.info(f"Confusion Matrix:\n{cm}\n")
        if "class_report" in metrics:
            cr = metrics["class_report"]
            self.logger.info(f"Classification Report:\n{cr}\n")

        QMessageBox.information(None, "Batch Inference Finished", f"Results saved in: {results_dir}")

    def build_transform(self) -> A.Compose:
        if self.check_adv_transform.isChecked():
            user_code = self.adv_transform_textedit.toPlainText().strip()
            if user_code:
                try:
                    loc = {}
                    safe_namespace = {"A": A, "ToTensorV2": ToTensorV2, "__builtins__": {}}
                    exec(f"transform_obj = {user_code}", safe_namespace, loc)
                    transform_candidate = loc.get("transform_obj", None)
                    if transform_candidate and isinstance(transform_candidate, A.Compose):
                        return transform_candidate
                    else:
                        self.logger.warning("Advanced transform code invalid. Reverting to basic.")
                except Exception as e:
                    self.logger.error(f"Error parsing advanced transform: {e}. Reverting.")
                    QMessageBox.warning(None, "Transform Error", f"Failed to parse advanced transform:\n{e}")

        do_resize = self.resize_check.isChecked()
        w = self.resize_width_spin.value()
        h = self.resize_height_spin.value()
        means = tuple(sp.value() for sp in self.mean_spin)
        stds = tuple(sp.value() for sp in self.std_spin)
        maintain_ar = self.maintain_ar_check.isChecked()

        transforms_list = []
        if do_resize:
            if maintain_ar:
                transforms_list.append(A.LongestMaxSize(max_size=max(w, h)))
                transforms_list.append(
                    A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                )
            else:
                transforms_list.append(A.Resize(height=h, width=w))

        transforms_list.append(A.Normalize(mean=means, std=stds))
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    def get_colormap_id(self) -> int:
        cmap_name = self.colormap_combo.currentText()
        if cmap_name == "HOT":
            return cv2.COLORMAP_HOT
        elif cmap_name == "BONE":
            return cv2.COLORMAP_BONE
        elif cmap_name == "HSV":
            return cv2.COLORMAP_HSV
        return cv2.COLORMAP_JET

    def export_single_inference_result(self):
        if self.last_single_inference_img is None:
            QMessageBox.warning(None, "Error", "No inference result to export.")
            return

        out_dir = QFileDialog.getExistingDirectory(None, "Select Folder to Save Inference Results")
        if not out_dir:
            return

        orig_path = os.path.join(out_dir, "inference_original.png")
        cv2.imwrite(orig_path, self.last_single_inference_img)

        if self.last_single_inference_overlay is not None:
            overlay_path = os.path.join(out_dir, "inference_overlay.png")
            cv2.imwrite(overlay_path, self.last_single_inference_overlay)

        results_csv = os.path.join(out_dir, "inference_results.csv")
        with open(results_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Predictions"])
            writer.writerow([self.last_single_inference_label_str])

        QMessageBox.information(None, "Export Done", f"Inference results saved to:\n{out_dir}")

    def open_output_folder(self):
        input_dir = self.inference_input_dir.text().strip()
        if not os.path.isdir(input_dir):
            QMessageBox.warning(None, "Error", "No valid input folder set.")
            return
        out_dir = os.path.join(input_dir, "inference_results")
        if os.path.isdir(out_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))
        else:
            QMessageBox.warning(None, "Error", f"No results directory found at: {out_dir}")

    def open_plotly_html_results(self):
        input_dir = self.inference_input_dir.text().strip()
        if not os.path.isdir(input_dir):
            QMessageBox.warning(None, "Error", "No valid input folder set.")
            return
        out_dir = os.path.join(input_dir, "inference_results")
        if not os.path.isdir(out_dir):
            QMessageBox.warning(None, "Error", f"No results directory found at: {out_dir}")
            return

        html_files = glob.glob(os.path.join(out_dir, "*.html"))
        if not html_files:
            QMessageBox.information(None, "No Plotly Files", "No HTML files found in output folder.")
            return

        for hf in html_files:
            QDesktopServices.openUrl(QUrl.fromLocalFile(hf))

    def load_class_names_file(self):
        fpath, _ = QFileDialog.getOpenFileName(None, "Select Class Names File", filter="*.txt *.csv")
        if not fpath:
            return
        new_names = []
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        new_names.append(name)
            self.class_names = new_names
            QMessageBox.information(None, "Class Names Loaded", f"Loaded {len(new_names)} class names.")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load class names:\n{e}")

    # -----------------------------
    # Unit Test Placeholders
    # -----------------------------
    def test_convert_to_qpixmap(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        pix = convert_to_qpixmap(arr)
        assert isinstance(pix, QPixmap), "convert_to_qpixmap did not return QPixmap."

    def test_metric_calculations(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2), "Confusion matrix shape mismatch."

    def test_run_shap_analysis_edge_cases(self):
        if HAVE_SHAP:
            pass
        else:
            pass
