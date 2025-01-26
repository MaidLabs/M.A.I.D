# plugin_inference.py

import os
import glob
import time
import traceback
import json
import cv2
import torch
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QProgressBar,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from base_plugin import BasePlugin
from plugins.plugin_training import MaidClassifier

# For plotting metrics & curves
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt

# For classification metrics
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# Try importing SHAP
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False


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


def overlay_cam_on_image(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Applies a Grad-CAM or Grad-CAM++ heatmap to the original BGR image.
    'cam' should be normalized between 0 and 1.
    """
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255.0
    image_float = image.astype(np.float32) / 255.0
    overlay = alpha * heatmap + (1 - alpha) * image_float

    if overlay.max() > 0:
        overlay /= overlay.max()
    overlay = (overlay * 255).astype(np.uint8)
    return overlay


class GradCAM:
    """
    A basic Grad-CAM implementation, hooking a specified model layer.
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

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None):
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

        gradients = self.gradients[0]  # Shape: [C, H, W]
        activations = self.activations[0]  # Shape: [C, H, W]

        alpha = gradients.mean(dim=[1, 2], keepdim=True)  # [C,1,1]
        weighted_activations = alpha * activations
        cam = weighted_activations.sum(dim=0)
        cam = torch.relu(cam)

        # Normalize [0, 1]
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam.cpu().numpy()


class GradCAMPlusPlus(GradCAM):
    """
    A Grad-CAM++ implementation.
    """

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None):
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


class BatchInferenceThread(QThread):
    """
    A QThread to handle potentially large batch inference without freezing the UI.
    """
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str, dict)  # results_dir, metrics_dict

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            self.run_batch_inference()
        except Exception as e:
            msg = f"Batch Inference Error: {e}\n{traceback.format_exc()}"
            self.log_signal.emit(msg)
            self.done_signal.emit("ERROR", {})
            return

    def run_batch_inference(self):
        """
        Actual batch inference logic, pulling from self.params to do the job.
        """
        model = self.params["model"]
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
        average_mode = self.params["average_mode"]  # for ROC/PR
        do_shap = self.params["do_shap"]
        shap_samples = self.params["shap_samples"]
        shap_bg = self.params["shap_bg"]

        # Load ground truth dict
        gt_dict = self.load_ground_truth_dict(gt_csv_path)

        exts = ("*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg")
        image_files = []
        for ext in exts:
            image_files.extend(
                glob.glob(os.path.join(input_dir, "**", ext), recursive=True)
            )
        if not image_files:
            self.log_signal.emit("No images found for batch inference.")
            self.done_signal.emit("DONE", {})
            return

        # create output dir
        os.makedirs(out_dir, exist_ok=True)
        cam_dir = os.path.join(out_dir, "inference_cam")
        os.makedirs(cam_dir, exist_ok=True)

        # text-based results
        txt_file = os.path.join(out_dir, "predictions.txt")

        # optional CSV
        if export_csv:
            csv_file = os.path.join(out_dir, "predictions.csv")
        else:
            csv_file = None

        all_preds = []
        all_probs = []
        all_targets = []

        misclassified_samples = []  # to store (img_path, gt_label, pred_label, pred_conf)

        if class_names:
            class_to_idx = {cname: i for i, cname in enumerate(class_names)}
            num_classes = len(class_names)
        else:
            class_to_idx = {}
            num_classes = None

        with open(txt_file, "w", encoding="utf-8") as f_txt:
            if csv_file:
                import csv
                csv_out = open(csv_file, "w", encoding="utf-8", newline="")
                # Include GT label column if available
                csv_writer = csv.writer(csv_out)
                csv_writer.writerow(["image_path", "top_k_predictions", "gt_label", "inference_time_ms"])
            else:
                csv_out = None
                csv_writer = None

            for idx, img_path in enumerate(image_files, start=1):
                bgr_img = cv2.imread(img_path)
                if bgr_img is None:
                    self.log_signal.emit(f"Skipping unreadable file: {img_path}")
                    self.progress_signal.emit(idx)
                    continue

                # transform
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                aug = transform(image=rgb_img)
                input_tensor = aug["image"].unsqueeze(0).to(device)

                # Inference
                start_time = time.time()
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000

                # top-k predictions
                topk_vals, topk_indices = torch.topk(probs, k=top_k, dim=1)
                result_items = []
                for rank in range(topk_vals.size(1)):
                    cls_idx = topk_indices[0, rank].item()
                    conf = topk_vals[0, rank].item() * 100.0
                    if class_names and cls_idx < len(class_names):
                        pred_label_str = class_names[cls_idx]
                    else:
                        pred_label_str = f"Class {cls_idx}"

                    if conf < min_confidence:
                        label_conf_str = f"Uncertain (<{min_confidence:.1f}%)"
                    else:
                        label_conf_str = f"{pred_label_str} ({conf:.1f}%)"
                    result_items.append(label_conf_str)

                # top-1 predicted class
                pred_class = topk_indices[0, 0].item()
                top1_conf = topk_vals[0, 0].item()

                # Store for metrics
                all_preds.append(pred_class)
                all_probs.append(probs[0].cpu().numpy())

                # ground truth
                base_name = os.path.basename(img_path)
                gt_label_str = None
                gt_index = -1
                if base_name in gt_dict:
                    gt_label_str = gt_dict[base_name]
                elif img_path in gt_dict:
                    gt_label_str = gt_dict[img_path]

                if gt_label_str is not None:
                    try:
                        # numeric label
                        gt_index = int(gt_label_str)
                    except ValueError:
                        # string label
                        gt_index = class_to_idx.get(gt_label_str, -1)
                all_targets.append(gt_index)

                topk_str = " | ".join(result_items)
                line_txt = f"{img_path} => {topk_str} [Time: {elapsed_ms:.1f} ms]"
                f_txt.write(line_txt + "\n")
                self.log_signal.emit(line_txt)

                if csv_writer:
                    # E.g. "Class 1 (82.3%); Class 2 (10.4%)"
                    csv_pred = "; ".join(result_items)
                    csv_writer.writerow([
                        img_path, csv_pred, gt_label_str if gt_label_str else "",
                        f"{elapsed_ms:.1f}"
                    ])

                # Check if misclassified
                if gt_index != -1 and gt_index != pred_class:
                    pred_label_for_mis = (
                        class_names[pred_class] if class_names and pred_class < len(class_names)
                        else f"Class {pred_class}"
                    )
                    misclassified_samples.append((
                        img_path,
                        gt_label_str if gt_label_str else str(gt_index),
                        pred_label_for_mis,
                        f"{top1_conf*100:.1f}%"
                    ))

                # Grad-CAM overlay if user wants
                if do_gradcam and (target_layer is not None):
                    # We'll pick the top-1 predicted class for generating CAM
                    if use_gradcam_pp:
                        gradcam = GradCAMPlusPlus(model, target_layer)
                    else:
                        gradcam = GradCAM(model, target_layer)

                    cam_map = gradcam.generate_cam(input_tensor, target_class=pred_class)
                    cam_resized = cv2.resize(cam_map, (bgr_img.shape[1], bgr_img.shape[0]))
                    overlay_img = overlay_cam_on_image(bgr_img, cam_resized, alpha=overlay_alpha)
                    out_path = os.path.join(cam_dir, f"CAM_{base_name}")
                    cv2.imwrite(out_path, overlay_img)

                self.progress_signal.emit(idx)

            if csv_out:
                csv_out.close()

        # Write misclassified samples to CSV if we found any
        if misclassified_samples:
            mc_file = os.path.join(out_dir, "misclassified.csv")
            import csv
            with open(mc_file, "w", encoding="utf-8", newline="") as f_mc:
                writer = csv.writer(f_mc)
                writer.writerow(["image_path", "ground_truth", "predicted", "pred_confidence"])
                for row in misclassified_samples:
                    writer.writerow(row)

        # done main loop
        metrics_dict = {}

        # Compute metrics if we have valid GT
        valid_inds = [i for i, t in enumerate(all_targets) if t != -1]
        if valid_inds:
            y_true = [all_targets[i] for i in valid_inds]
            y_pred = [all_preds[i] for i in valid_inds]
            y_prob = [all_probs[i] for i in valid_inds]

            # confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_path = os.path.join(out_dir, "confusion_matrix.txt")
            with open(cm_path, "w") as f_cm:
                f_cm.write("Confusion Matrix:\n")
                f_cm.write(str(cm))
                f_cm.write("\n")

            # classification report (as string and dict)
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

            # simple plot of confusion matrix
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
            plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
            plt.close()

            # Confidence histogram (top-1 only)
            top1_conf_all = [max(prob_vec) for prob_vec in y_prob]
            plt.figure()
            plt.hist(top1_conf_all, bins=20, range=(0, 1), color='green', alpha=0.7)
            plt.title("Top-1 Confidence Distribution")
            plt.xlabel("Confidence")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(out_dir, "confidence_histogram.png"))
            plt.close()

            # If we have the number of classes
            if num_classes == 2:
                # binary ROC
                y_prob_pos = [p[1] for p in y_prob]
                fpr, tpr, _ = roc_curve(y_true, y_prob_pos, pos_label=1)
                roc_auc_val = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={roc_auc_val:.3f}")
                plt.plot([0, 1], [0, 1], "r--")
                plt.title("ROC Curve (Binary)")
                plt.legend()
                plt.savefig(os.path.join(out_dir, "roc_curve.png"))
                plt.close()

                # binary PR curve
                precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
                avg_prec = average_precision_score(y_true, y_prob_pos)
                plt.figure()
                plt.plot(recall, precision, label=f"AP={avg_prec:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve (Binary)")
                plt.legend()
                plt.savefig(os.path.join(out_dir, "pr_curve.png"))
                plt.close()

                # Calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob_pos, n_bins=10)
                plt.figure()
                plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
                plt.plot([0, 1], [0, 1], "r--", label="Perfectly calibrated")
                plt.title("Calibration Curve (Binary)")
                plt.xlabel("Mean Predicted Value")
                plt.ylabel("Fraction of Positives")
                plt.legend()
                plt.savefig(os.path.join(out_dir, "calibration_curve.png"))
                plt.close()

                try:
                    roc_auc_overall = roc_auc_score(y_true, y_prob_pos)
                except:
                    roc_auc_overall = None

                # MCC
                mcc_val = matthews_corrcoef(y_true, y_pred)

                # PR AUC
                pr_auc_val = avg_prec

            elif num_classes and num_classes > 2:
                # multi-class ROC
                roc_auc_overall = self.plot_multiclass_roc(
                    y_true=np.array(y_true),
                    y_prob=np.array(y_prob),
                    out_dir=out_dir,
                    average=average_mode,
                    class_names=class_names
                )
                # multi-class PR
                pr_auc_val = self.plot_multiclass_pr(
                    y_true=np.array(y_true),
                    y_prob=np.array(y_prob),
                    out_dir=out_dir,
                    average=average_mode,
                    class_names=class_names
                )
                # MCC
                mcc_val = matthews_corrcoef(y_true, y_pred)
            else:
                # If we can't determine #classes, just skip these advanced plots
                roc_auc_overall = None
                pr_auc_val = None
                mcc_val = matthews_corrcoef(y_true, y_pred)

            # optionally run SHAP
            if do_shap and HAVE_SHAP:
                shap_files = []
                for i in valid_inds:
                    shap_files.append(image_files[i])
                shap_file_out = os.path.join(out_dir, "shap_summary.png")
                try:
                    self.run_shap_analysis(model, device, shap_files, transform,
                                           shap_samples, shap_bg, shap_file_out)
                except Exception as e:
                    self.log_signal.emit(f"SHAP error: {e}")

            metrics_dict["confusion_matrix"] = cm
            metrics_dict["class_report"] = cls_rep_str

            # Save core metrics as JSON (including numeric versions of classification report)
            self.save_metrics_to_json(
                cm,
                cls_rep_dict,
                mcc_val,
                roc_auc_overall,
                pr_auc_val,
                out_dir
            )

        self.log_signal.emit("Batch Inference Done.")
        self.done_signal.emit(out_dir, metrics_dict)

    def load_ground_truth_dict(self, csv_file: str):
        """
        CSV lines: image_filename, label_index (or label_str)
        """
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
            self.log_signal.emit(f"Error reading CSV: {e}")
        return gt_dict

    def plot_multiclass_roc(self, y_true, y_prob, out_dir, average="macro", class_names=None):
        # One-vs-rest approach
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
            plt.title(f"Multi-class ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "roc_curve.png"))
        plt.close()
        return roc_auc

    def plot_multiclass_pr(self, y_true, y_prob, out_dir, average="macro", class_names=None):
        # One-vs-rest approach
        num_classes = y_prob.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(num_classes))

        # Attempt multi-class average precision
        try:
            pr_auc = average_precision_score(y_true_bin, y_prob, average=average)
        except:
            pr_auc = None

        plt.figure()
        for c in range(num_classes):
            # For each class, treat it as "positive"
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
        plt.savefig(os.path.join(out_dir, "pr_curve.png"))
        plt.close()
        return pr_auc

    def run_shap_analysis(self, model, device, image_files, transform,
                          shap_samples, shap_bg, shap_outfile):
        sample_files = image_files[:shap_samples]
        background_files = image_files[:shap_bg] if shap_bg > 0 else sample_files[:1]

        shap_list = []
        for fimg in sample_files:
            bgr = cv2.imread(fimg)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            aug = transform(image=rgb)
            shap_list.append(aug["image"])

        bg_list = []
        for fimg in background_files:
            bgr = cv2.imread(fimg)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            aug = transform(image=rgb)
            bg_list.append(aug["image"])

        if not shap_list:
            self.log_signal.emit("No valid images for SHAP.")
            return

        shap_tensor = torch.stack(shap_list, dim=0).to(device)
        if not bg_list:
            bg_list = shap_list[:1]

        bg_tensor = torch.stack(bg_list, dim=0).to(device)

        explainer = shap.DeepExplainer(model, bg_tensor)
        shap_values = explainer.shap_values(shap_tensor)

        plt.figure()
        # For multi-class models, shap_values can be a list
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
        self.log_signal.emit(f"SHAP summary saved to {shap_outfile}")

    def save_metrics_to_json(self, cm, cls_rep_dict, mcc_val, roc_auc_val, pr_auc_val, out_dir):
        """
        Saves key metrics to a JSON file for machine-readable processing.
        """
        data = {}
        data["confusion_matrix"] = cm.tolist()
        data["classification_report"] = cls_rep_dict  # This is already a dict from output_dict=True
        data["mcc"] = mcc_val if mcc_val is not None else None
        data["roc_auc"] = roc_auc_val if roc_auc_val is not None else None
        data["pr_auc"] = pr_auc_val if pr_auc_val is not None else None

        out_path = os.path.join(out_dir, "metrics.json")
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)


class Plugin(BasePlugin):
    """
    Updated plugin for post-training evaluation & inference, with additional metrics and a scrollable UI.
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

        # Transform controls
        self.resize_check = None
        self.resize_width_spin = None
        self.resize_height_spin = None
        self.mean_spin = []
        self.std_spin = []

        # Check box to maintain aspect ratio
        self.maintain_ar_check = None

        # Top-k spin
        self.spin_top_k = None
        # Min confidence
        self.dspin_min_confidence = None
        # Enable batch grad-cam
        self.cb_batch_gradcam = None
        # Export CSV
        self.cb_export_csv = None

        # Extra text log
        self.text_log = None

        # For layer selection
        self.layer_combo = None
        self.available_layers = []

        # Device selection
        self.device_combo = None

        self.model = None
        self.target_layer = None
        self.class_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_thread = None

    def create_tab(self) -> QWidget:
        """
        Builds the main UI inside a QScrollArea so it becomes scrollable if it doesn't fit.
        """
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        container_widget = QWidget()
        main_layout = QVBoxLayout(container_widget)

        # ------------- Checkpoint selection + device selection -------------
        h_ckpt = QHBoxLayout()
        self.ckpt_edit = QLineEdit()
        btn_ckpt = QPushButton("Browse Checkpoint...")
        btn_ckpt.clicked.connect(self.browse_ckpt)
        h_ckpt.addWidget(QLabel("Checkpoint:"))
        h_ckpt.addWidget(self.ckpt_edit)
        h_ckpt.addWidget(btn_ckpt)
        main_layout.addLayout(h_ckpt)

        # Device selection
        h_device = QHBoxLayout()
        h_device.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto (cuda if available)", "CPU", "CUDA"])
        h_device.addWidget(self.device_combo)
        main_layout.addLayout(h_device)

        # Transform customization
        transform_box = QVBoxLayout()
        self.resize_check = QCheckBox("Resize Images?")
        self.resize_check.setChecked(True)
        transform_box.addWidget(self.resize_check)

        # Maintain aspect ratio
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

        # Normalization
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
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.01)
            sp.setValue(0.5)
            self.std_spin.append(sp)
            norm_layout.addWidget(sp)
        transform_box.addLayout(norm_layout)

        main_layout.addLayout(transform_box)

        # Single image UI
        h_img = QHBoxLayout()
        self.infer_img_edit = QLineEdit()
        btn_img = QPushButton("Browse Image...")
        btn_img.clicked.connect(self.browse_infer_image)
        h_img.addWidget(QLabel("Test Image:"))
        h_img.addWidget(self.infer_img_edit)
        h_img.addWidget(btn_img)
        main_layout.addLayout(h_img)

        # Grad-CAM++ for single image
        self.cb_gradcam_pp = QCheckBox("Use Grad-CAM++ (single image)")
        main_layout.addWidget(self.cb_gradcam_pp)

        # Grad-CAM alpha
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Grad-CAM Overlay Alpha:"))
        self.dspin_overlay_alpha = QDoubleSpinBox()
        self.dspin_overlay_alpha.setRange(0.0, 1.0)
        self.dspin_overlay_alpha.setSingleStep(0.1)
        self.dspin_overlay_alpha.setValue(0.5)
        alpha_layout.addWidget(self.dspin_overlay_alpha)
        main_layout.addLayout(alpha_layout)

        # top-k spin + min confidence
        tk_layout = QHBoxLayout()
        tk_layout.addWidget(QLabel("Top-k for Inference:"))
        self.spin_top_k = QSpinBox()
        self.spin_top_k.setRange(1, 10)
        self.spin_top_k.setValue(1)
        tk_layout.addWidget(self.spin_top_k)

        tk_layout.addWidget(QLabel("Min Confidence (%):"))
        self.dspin_min_confidence = QDoubleSpinBox()
        self.dspin_min_confidence.setRange(0.0, 100.0)
        self.dspin_min_confidence.setValue(10.0)
        self.dspin_min_confidence.setSingleStep(1.0)
        tk_layout.addWidget(self.dspin_min_confidence)

        main_layout.addLayout(tk_layout)

        # model layer combo for Grad-CAM
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Target Layer:"))
        self.layer_combo = QComboBox()
        layer_layout.addWidget(self.layer_combo)
        main_layout.addLayout(layer_layout)

        btn_infer = QPushButton("Run Inference (Single)")
        btn_infer.clicked.connect(self.run_inference)
        main_layout.addWidget(btn_infer)

        self.infer_result_label = QLabel("Result: ")
        main_layout.addWidget(self.infer_result_label)

        # Display Original vs Grad-CAM
        h_disp = QHBoxLayout()
        self.label_infer_orig = QLabel("Original")
        self.label_infer_cam = QLabel("Grad-CAM/++")
        self.label_infer_orig.setFixedSize(300, 300)
        self.label_infer_cam.setFixedSize(300, 300)
        self.label_infer_orig.setAlignment(Qt.AlignCenter)
        self.label_infer_cam.setAlignment(Qt.AlignCenter)
        h_disp.addWidget(self.label_infer_orig)
        h_disp.addWidget(self.label_infer_cam)
        main_layout.addLayout(h_disp)

        # Batch inference
        batch_inf_layout = QHBoxLayout()
        self.inference_input_dir = QLineEdit()
        btn_infer_dir = QPushButton("Browse Folder...")
        btn_infer_dir.clicked.connect(self.browse_inference_folder)
        batch_inf_layout.addWidget(QLabel("Inference Folder:"))
        batch_inf_layout.addWidget(self.inference_input_dir)
        batch_inf_layout.addWidget(btn_infer_dir)
        main_layout.addLayout(batch_inf_layout)

        # GT CSV
        h_gtcsv = QHBoxLayout()
        self.gt_csv_edit = QLineEdit()
        btn_gtcsv = QPushButton("Browse GT CSV...")
        btn_gtcsv.clicked.connect(self.browse_gt_csv)
        h_gtcsv.addWidget(QLabel("GT CSV (optional):"))
        h_gtcsv.addWidget(self.gt_csv_edit)
        h_gtcsv.addWidget(btn_gtcsv)
        main_layout.addLayout(h_gtcsv)

        # ROC averaging
        roc_layout = QHBoxLayout()
        roc_layout.addWidget(QLabel("ROC/PR Average Mode:"))
        self.cbx_roc_average = QComboBox()
        self.cbx_roc_average.addItems(["macro", "micro", "weighted"])
        self.cbx_roc_average.setCurrentText("macro")
        roc_layout.addWidget(self.cbx_roc_average)
        main_layout.addLayout(roc_layout)

        # SHAP
        self.cb_enable_shap = QCheckBox("Enable SHAP analysis")
        main_layout.addWidget(self.cb_enable_shap)

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
        main_layout.addLayout(shap_layout)

        # checkbox for batch grad-cam
        self.cb_batch_gradcam = QCheckBox("Generate Grad-CAM in Batch Inference?")
        self.cb_batch_gradcam.setChecked(True)
        main_layout.addWidget(self.cb_batch_gradcam)

        # Export CSV
        self.cb_export_csv = QCheckBox("Export CSV for predictions?")
        main_layout.addWidget(self.cb_export_csv)

        btn_batch_infer = QPushButton("Run Batch Inference")
        btn_batch_infer.clicked.connect(self.start_batch_inference)
        main_layout.addWidget(btn_batch_infer)

        # Progress + logging
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        main_layout.addWidget(self.text_log)

        scroll_area.setWidget(container_widget)
        return scroll_area

    def get_selected_device(self):
        """
        Helper to return a torch.device based on user selection in the combo box.
        """
        choice = self.device_combo.currentText()
        if choice == "CPU":
            return torch.device("cpu")
        elif choice == "CUDA":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            # Auto
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def browse_ckpt(self):
        fpath, _ = QFileDialog.getOpenFileName(
            None, "Select Checkpoint File", filter="*.ckpt"
        )
        if fpath:
            self.ckpt_edit.setText(fpath)
            try:
                # Update device from user’s selection
                self.device = self.get_selected_device()
                self.model = MaidClassifier.load_from_checkpoint(fpath)
                self.model.to(self.device).eval()

                # check for class_names
                if hasattr(self.model, "class_names"):
                    self.class_names = self.model.class_names
                else:
                    self.class_names = None

                # discover available candidate layers for Grad-CAM
                self.available_layers = self.discover_conv_layers(self.model.backbone)
                self.layer_combo.clear()
                for name_layer in self.available_layers:
                    self.layer_combo.addItem(name_layer[0])  # text
                if self.available_layers:
                    self.layer_combo.setCurrentIndex(len(self.available_layers) - 1)

                QMessageBox.information(
                    None,
                    "Model Loaded",
                    f"Model loaded from:\n{fpath}\nFound {len(self.available_layers)} candidate layers."
                )
            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(
                    None,
                    "Load Error",
                    f"Failed to load model:\n{str(e)}"
                )

    def discover_conv_layers(self, net: torch.nn.Module):
        """
        Recursively discover all modules that might be suitable for hooking Grad-CAM.
        Return a list of tuples (name, module).
        """

        layers_list = []

        def recurse_layers(parent, parent_name):
            for name, module in parent.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                # If module has parameters, consider it a candidate
                has_param = list(module.parameters())
                if has_param:
                    layers_list.append((full_name, module))
                recurse_layers(module, full_name)

        recurse_layers(net, "")
        return layers_list

    def browse_infer_image(self):
        fpath, _ = QFileDialog.getOpenFileName(
            None, "Select an Image",
            filter="Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
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

        transform = self.build_transform()

        idx_layer = self.layer_combo.currentIndex()
        if 0 <= idx_layer < len(self.available_layers):
            self.target_layer = self.available_layers[idx_layer][1]
        else:
            self.target_layer = None

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

            label_str, overlay = self.infer_image_single(bgr_img, transform)
            self.infer_result_label.setText(f"Result: {label_str}")

            if overlay is not None:
                pm_cam = convert_to_qpixmap(overlay)
                self.label_infer_cam.setPixmap(pm_cam.scaled(
                    self.label_infer_cam.width(),
                    self.label_infer_cam.height(),
                    Qt.KeepAspectRatio
                ))
            else:
                self.label_infer_cam.setPixmap(QPixmap())

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Inference Error", str(e))

    def infer_image_single(self, bgr_img: np.ndarray, transform: A.Compose):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        aug = transform(image=rgb_img)
        input_tensor = aug["image"].unsqueeze(0).to(self.device)

        top_k = self.spin_top_k.value()
        min_conf = self.dspin_min_confidence.value()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)

        topk_vals, topk_indices = torch.topk(probs, k=top_k, dim=1)
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

            if conf < min_conf:
                label_conf_str = f"Uncertain (<{min_conf:.1f}%)"
            else:
                label_conf_str = f"{cls_name} ({conf:.1f}%)"
            result_items.append(label_conf_str)

        label_str = " | ".join(result_items)

        overlay = None
        if self.target_layer:
            use_pp = self.cb_gradcam_pp.isChecked()
            if use_pp:
                gradcam = GradCAMPlusPlus(self.model, self.target_layer)
            else:
                gradcam = GradCAM(self.model, self.target_layer)

            # Generate Grad-CAM for the top-1 predicted class
            pred_class = topk_indices[0, 0].item()
            cam_map = gradcam.generate_cam(input_tensor, target_class=pred_class)
            cam_resized = cv2.resize(cam_map, (bgr_img.shape[1], bgr_img.shape[0]))
            alpha_val = self.dspin_overlay_alpha.value()
            overlay = overlay_cam_on_image(bgr_img, cam_resized, alpha=alpha_val)

        return label_str, overlay

    def build_transform(self) -> A.Compose:
        do_resize = self.resize_check.isChecked()
        w = self.resize_width_spin.value()
        h = self.resize_height_spin.value()
        means = tuple(sp.value() for sp in self.mean_spin)
        stds = tuple(sp.value() for sp in self.std_spin)
        maintain_ar = self.maintain_ar_check.isChecked()

        transforms_list = []
        if do_resize:
            if maintain_ar:
                # Resize the smaller dimension, then pad
                transforms_list.append(A.LongestMaxSize(max_size=max(w, h)))
                # Ensure final size is at least w x h by padding
                transforms_list.append(
                    A.PadIfNeeded(
                        min_height=h,
                        min_width=w,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )
                )
            else:
                transforms_list.append(A.Resize(height=h, width=w))

        transforms_list.append(A.Normalize(mean=means, std=stds))
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    def browse_inference_folder(self):
        folder = QFileDialog.getExistingDirectory(None, "Select Inference Folder")
        if folder:
            self.inference_input_dir.setText(folder)

    def browse_gt_csv(self):
        fpath, _ = QFileDialog.getOpenFileName(None, "Select CSV File", filter="*.csv")
        if fpath:
            self.gt_csv_edit.setText(fpath)

    def start_batch_inference(self):
        if not self.model:
            QMessageBox.warning(None, "Error", "Model not loaded.")
            return
        input_dir = self.inference_input_dir.text().strip()
        if not os.path.isdir(input_dir):
            QMessageBox.warning(None, "Error", "Invalid folder.")
            return

        # Update device from user’s selection again, in case changed
        self.device = self.get_selected_device()

        transform = self.build_transform()

        idx_layer = self.layer_combo.currentIndex()
        if 0 <= idx_layer < len(self.available_layers):
            self.target_layer = self.available_layers[idx_layer][1]
        else:
            self.target_layer = None

        out_dir = os.path.join(input_dir, "inference_results")
        os.makedirs(out_dir, exist_ok=True)

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
            "shap_bg": self.spin_shap_bg_samples.value()
        }

        self.batch_thread = BatchInferenceThread(params=params)
        self.batch_thread.progress_signal.connect(self.update_progress)
        self.batch_thread.log_signal.connect(self.append_log)
        self.batch_thread.done_signal.connect(self.batch_inference_done)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        # Indefinite progress until we get a final count
        self.progress_bar.setRange(0, 0)
        self.batch_thread.start()

    def update_progress(self, val: int):
        # Switch from indefinite to definite range if val is large
        current_max = self.progress_bar.maximum()
        if current_max == 0 or val > current_max:
            self.progress_bar.setRange(0, val)
        self.progress_bar.setValue(val)

    def append_log(self, msg: str):
        self.text_log.append(msg)
        self.text_log.ensureCursorVisible()

    def batch_inference_done(self, results_dir: str, metrics: dict):
        self.progress_bar.setVisible(False)
        if results_dir == "ERROR":
            self.append_log("Batch inference encountered an error.")
            return
        self.append_log("Batch inference completed successfully.")
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            self.append_log(f"Confusion Matrix:\n{cm}\n")
        if "class_report" in metrics:
            cr = metrics["class_report"]
            self.append_log(f"Classification Report:\n{cr}\n")
        QMessageBox.information(
            None,
            "Batch Inference Finished",
            f"Results saved in: {results_dir}"
        )
