# plugin_training.py

import os
import glob
import traceback
import shutil
import random
import time
import gc
import logging
import importlib.util
from collections import Counter
from typing import Tuple, Dict, Any, List, Optional, Union
import unittest

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, Callback
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import (
    SequentialLR, LambdaLR, StepLR, ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
)

# === Optuna additions ===
import optuna
from pytorch_lightning.tuner.tuning import Tuner

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QDoubleSpinBox, QSpinBox,
    QCheckBox, QComboBox, QTextEdit, QProgressBar, QMessageBox
)

# If you have a local base_plugin.py for your system:
try:
    from base_plugin import BasePlugin
except ImportError:
    # Fallback stub if it doesn't exist in your environment
    class BasePlugin:
        def __init__(self):
            pass

try:
    from pytorch_optimizer import Lamb as LambClass
    HAVE_LAMB = True
except ImportError:
    LambClass = None
    HAVE_LAMB = False

try:
    import pynvml  # For VRAM checks
    HAVE_PYNVML = True
except ImportError:
    HAVE_PYNVML = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# Optional: For performance
torch.backends.cudnn.benchmark = True

# -----------------------------------
#  LOGGING SETUP
# -----------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# ==============================
# 1) CONFIG OBJECTS & HELPERS
# ==============================
class DataConfig:
    """
    Holds dataset-related configuration.
    """
    def __init__(
        self,
        root_dir: str,
        val_split: float,
        test_split: float,
        batch_size: int,
        allow_grayscale: bool
    ):
        self.root_dir = root_dir
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.allow_grayscale = allow_grayscale


class TrainConfig:
    """
    Holds all training-related parameters.
    """
    def __init__(
        self,
        max_epochs: int,
        architecture: str,
        lr: float,
        momentum: float,
        weight_decay: float,
        use_weighted_loss: bool,
        optimizer_name: str,
        scheduler_name: str,
        scheduler_params: Dict[str, float],
        do_early_stopping: bool,
        early_stopping_monitor: str,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
        early_stopping_mode: str,
        brightness_contrast: bool,
        hue_saturation: bool,
        gaussian_noise: bool,
        use_rotation: bool,
        use_flip: bool,
        flip_mode: str,
        flip_prob: float,
        use_crop: bool,
        use_elastic: bool,
        normalize_pixel_intensity: bool,
        use_grid_distortion: bool,
        use_optical_distortion: bool,
        use_mixup: bool,
        use_cutmix: bool,
        mix_alpha: float,
        dropout_rate: float,
        label_smoothing: float,
        freeze_backbone: bool,
        loss_function: str,
        gradient_clip_val: float,
        use_lr_finder: bool,
        accept_lr_suggestion: bool,
        use_tensorboard: bool,
        use_mixed_precision: bool,
        warmup_epochs: int,
        use_inception_299: bool,
        enable_gradient_checkpointing: bool,
        enable_grad_accum: bool,
        accumulate_grad_batches: int,
        check_val_every_n_epoch: int,
        freeze_config: Dict[str, bool],
        num_workers: int,
        val_center_crop: bool,
        random_crop_prob: float,
        random_crop_scale_min: float,
        random_crop_scale_max: float,
        pretrained_weights: bool = True,
        run_gc: bool = False,
        enable_tta: bool = False,
        profile_memory: bool = False,
        # For custom model import:
        load_custom_model: bool = False,
        custom_model_path: str = "",
        # Optionally support dynamic architecture file
        custom_architecture_file: str = ""
    ):
        self.max_epochs = max_epochs
        self.architecture = architecture
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_weighted_loss = use_weighted_loss
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.do_early_stopping = do_early_stopping

        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_mode = early_stopping_mode

        self.brightness_contrast = brightness_contrast
        self.hue_saturation = hue_saturation
        self.gaussian_noise = gaussian_noise
        self.use_rotation = use_rotation
        self.use_flip = use_flip
        self.flip_mode = flip_mode
        self.flip_prob = flip_prob
        self.use_crop = use_crop
        self.use_elastic = use_elastic
        self.normalize_pixel_intensity = normalize_pixel_intensity

        self.use_grid_distortion = use_grid_distortion
        self.use_optical_distortion = use_optical_distortion
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mix_alpha = mix_alpha

        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.freeze_backbone = freeze_backbone
        self.loss_function = loss_function

        self.gradient_clip_val = gradient_clip_val
        self.use_lr_finder = use_lr_finder
        self.accept_lr_suggestion = accept_lr_suggestion
        self.use_tensorboard = use_tensorboard
        self.use_mixed_precision = use_mixed_precision
        self.warmup_epochs = warmup_epochs
        self.use_inception_299 = use_inception_299

        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_grad_accum = enable_grad_accum
        self.accumulate_grad_batches = accumulate_grad_batches
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.freeze_config = freeze_config
        self.num_workers = num_workers
        self.val_center_crop = val_center_crop

        self.random_crop_prob = random_crop_prob
        self.random_crop_scale_min = random_crop_scale_min
        self.random_crop_scale_max = random_crop_scale_max

        self.pretrained_weights = pretrained_weights
        self.run_gc = run_gc
        self.enable_tta = enable_tta
        self.profile_memory = profile_memory

        self.load_custom_model = load_custom_model
        self.custom_model_path = custom_model_path
        self.custom_architecture_file = custom_architecture_file


def get_available_vram() -> float:
    """
    Returns an approximate available VRAM in MB for the current GPU.
    If CUDA is not available or cannot fetch, returns a large number.
    """
    if not torch.cuda.is_available():
        return 999999.0
    if not HAVE_PYNVML:
        return 999999.0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem_mb = info.free / 1024**2
        pynvml.nvmlShutdown()
        return free_mem_mb
    except Exception:
        return 999999.0


# ==============================
# 2) DATA FUNCTIONS & CLASSES
# ==============================
def gather_samples_and_classes(root_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Gather image samples and class names from the given root directory.
    Skips corrupt or non-image files more gracefully.
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"Invalid root_dir: {root_dir}")

    classes = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    if not classes:
        raise ValueError(f"No class subfolders found in {root_dir}.")

    valid_extensions = (".tiff", ".tif", ".png", ".jpg", ".jpeg")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    samples = []
    for cls_name in classes:
        sub_dir = os.path.join(root_dir, cls_name)
        for ext in valid_extensions:
            pattern = os.path.join(sub_dir, "**", f"*{ext}")
            file_list = glob.glob(pattern, recursive=True)
            for fpath in file_list:
                if not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, "rb") as fp:
                        _ = fp.read(20)
                except OSError:
                    continue
                samples.append((fpath, class_to_idx[cls_name]))

    return samples, classes


class AlbumentationsDataset(Dataset):
    """
    A PyTorch Dataset wrapper that applies Albumentations transforms.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[A.Compose] = None,
        classes: Optional[List[str]] = None,
        allow_grayscale: bool = False
    ):
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.classes = classes if classes is not None else []
        self.allow_grayscale = allow_grayscale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fpath, label = self.samples[idx]
        image_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"Failed to load or corrupt image: {fpath}")

        if len(image_bgr.shape) == 2:  # single channel
            if not self.allow_grayscale:
                raise ValueError(
                    f"Encountered a single-channel image but allow_grayscale=False: {fpath}"
                )
            else:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image_rgb)
            image_tensor = augmented["image"].float()
        else:
            image_tensor = torch.from_numpy(
                np.transpose(image_rgb, (2, 0, 1))
            ).float() / 255.0

        return image_tensor, label


# =====================
# 3) AUGMENTATION UTILS
# =====================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, c, h, w = x.size()
    index = torch.randperm(batch_size, device=x.device)

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bw = int(np.sqrt(1 - lam) * w)
    bh = int(np.sqrt(1 - lam) * h)

    x1 = np.clip(cx - bw // 2, 0, w)
    x2 = np.clip(cx + bw // 2, 0, w)
    y1 = np.clip(cy - bh // 2, 0, h)
    y2 = np.clip(cy + bh // 2, 0, h)

    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    y_a, y_b = y, y[index]
    return x_cut, y_a, y_b, lam


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
    use_mixup: bool = False,
    use_cutmix: bool = False,
    alpha: float = 1.0
):
    images, labels = list(zip(*batch))
    images_tensor = torch.stack(images, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    if use_mixup and use_cutmix:
        # Decide randomly whether to apply MixUp or CutMix
        if random.random() < 0.5:
            mixed_x, y_a, y_b, lam = mixup_data(images_tensor, labels_tensor, alpha=alpha)
            return mixed_x, (y_a, y_b, lam)
        else:
            cut_x, y_a, y_b, lam = cutmix_data(images_tensor, labels_tensor, alpha=alpha)
            return cut_x, (y_a, y_b, lam)
    elif use_mixup:
        mixed_x, y_a, y_b, lam = mixup_data(images_tensor, labels_tensor, alpha=alpha)
        return mixed_x, (y_a, y_b, lam)
    elif use_cutmix:
        cut_x, y_a, y_b, lam = cutmix_data(images_tensor, labels_tensor, alpha=alpha)
        return cut_x, (y_a, y_b, lam)
    else:
        return images_tensor, labels_tensor


# =========================
# 4) LOSSES & UTIL CLASSES
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =========================
# 5) MAIN MODEL CLASS
# =========================
class MaidClassifier(pl.LightningModule):
    """
    Main classifier LightningModule with:
     - Optional partial freezing
     - Support for TTA
     - Weighted loss / Focal / BCE variants
     - Option to load a user-provided custom model or custom architecture
    """
    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 2,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        use_weighted_loss: bool = False,
        class_weights: Optional[List[float]] = None,
        optimizer_name: str = "adam",
        scheduler_name: str = "none",
        scheduler_params: Optional[Dict[str, float]] = None,
        dropout_rate: float = 0.0,
        label_smoothing: float = 0.0,
        freeze_backbone: bool = False,
        loss_function: str = "cross_entropy",
        pretrained: bool = True,
        enable_tta: bool = False,
        load_custom_model: bool = False,
        custom_model_path: str = "",
        custom_architecture_file: str = ""
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["class_weights", "scheduler_params"]
        )

        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_weighted_loss = use_weighted_loss
        self.class_weights = class_weights
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params if scheduler_params else {}
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.freeze_backbone = freeze_backbone
        self.loss_function = loss_function
        self.enable_tta = enable_tta

        self.load_custom_model = load_custom_model
        self.custom_model_path = custom_model_path
        self.custom_architecture_file = custom_architecture_file

        self.class_names: Optional[List[str]] = None

        # If BCE but num_classes > 1, fallback
        if self.loss_function == "bce" and (num_classes > 1):
            logger.warning("BCE with num_classes>1 is not recommended. Using cross_entropy fallback.")
            self.loss_function = "cross_entropy"

        # Attempt dynamic import if requested
        self.custom_model = None
        if self.load_custom_model:
            if self.custom_architecture_file and os.path.isfile(self.custom_architecture_file):
                # Dynamic import scenario
                logger.info(f"Loading custom architecture from: {self.custom_architecture_file}")
                spec = importlib.util.spec_from_file_location("custom_arch", self.custom_architecture_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                # Suppose the user’s file has a function get_model(...)
                if hasattr(mod, "get_model"):
                    self.custom_model = mod.get_model(num_classes=self.num_classes, pretrained=pretrained)
                    if os.path.isfile(self.custom_model_path):
                        self.custom_model.load_state_dict(
                            torch.load(self.custom_model_path, map_location="cpu")
                        )
                    else:
                        logger.warning(f"Custom model weights not found: {self.custom_model_path}")
                else:
                    raise ValueError("No get_model() function found in the custom architecture file.")
            else:
                # Just load .pt / .pth as a full model
                if not os.path.isfile(self.custom_model_path):
                    raise ValueError(f"Custom model path not found: {self.custom_model_path}")
                logger.info(f"Loading entire custom model from: {self.custom_model_path}")
                self.custom_model = torch.load(self.custom_model_path, map_location="cpu")

        if self.custom_model is not None:
            # We assume the user’s model is fully constructed with final layers
            self.backbone = None
            self.head = None
        else:
            # Build a standard torchvision backbone
            self.backbone, in_feats = self._create_backbone(architecture, pretrained)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )

        # Built-in custom loss if needed
        if self.loss_function == "focal":
            self.loss_fn = FocalLoss()
        elif self.loss_function == "bce_single_logit" and num_classes == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = None

    def _create_backbone(self, architecture: str, pretrained: bool) -> Tuple[nn.Module, int]:
        def weight_if_pretrained(res_w):
            return res_w if pretrained else None

        arch = architecture.lower()
        if arch == "resnet18":
            m = models.resnet18(weights=weight_if_pretrained(models.ResNet18_Weights.IMAGENET1K_V1))
            in_feats = m.fc.in_features
            m.fc = nn.Identity()
            return m, in_feats
        elif arch == "resnet50":
            m = models.resnet50(weights=weight_if_pretrained(models.ResNet50_Weights.IMAGENET1K_V2))
            in_feats = m.fc.in_features
            m.fc = nn.Identity()
            return m, in_feats
        elif arch == "resnet101":
            m = models.resnet101(weights=weight_if_pretrained(models.ResNet101_Weights.IMAGENET1K_V2))
            in_feats = m.fc.in_features
            m.fc = nn.Identity()
            return m, in_feats
        elif arch == "densenet":
            densenet = models.densenet121(
                weights=weight_if_pretrained(models.DenseNet121_Weights.IMAGENET1K_V1)
            )
            in_feats = densenet.classifier.in_features
            densenet.classifier = nn.Identity()
            return densenet, in_feats
        elif arch == "vgg":
            vgg = models.vgg16(weights=weight_if_pretrained(models.VGG16_Weights.IMAGENET1K_V1))
            in_feats = vgg.classifier[6].in_features
            vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
            return vgg, in_feats
        elif arch == "inception":
            inception = models.inception_v3(
                weights=weight_if_pretrained(models.Inception_V3_Weights.IMAGENET1K_V1),
                aux_logits=False
            )
            in_feats = inception.fc.in_features
            inception.fc = nn.Identity()
            return inception, in_feats
        elif arch == "mobilenet":
            mbnet = models.mobilenet_v2(
                weights=weight_if_pretrained(models.MobileNet_V2_Weights.IMAGENET1K_V1)
            )
            in_feats = mbnet.classifier[1].in_features
            mbnet.classifier = nn.Identity()
            return mbnet, in_feats
        elif arch == "efficientnet_b0":
            effnet = models.efficientnet_b0(
                weights=weight_if_pretrained(models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            )
            in_feats = effnet.classifier[1].in_features
            effnet.classifier = nn.Identity()
            return effnet, in_feats
        elif arch == "convnext_tiny":
            convnext = models.convnext_tiny(
                weights=weight_if_pretrained(models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            )
            in_feats = convnext.classifier[2].in_features
            convnext.classifier[2] = nn.Identity()
            return convnext, in_feats
        elif arch == "convnext_large":
            convnext = models.convnext_large(
                weights=weight_if_pretrained(models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
            )
            in_feats = convnext.classifier[2].in_features
            convnext.classifier[2] = nn.Identity()
            return convnext, in_feats
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.custom_model is not None:
            return self.custom_model(x)
        else:
            feats = self.backbone(x)
            return self.head(feats)

    def enable_gradient_checkpointing(self) -> None:
        """
        Try enabling gradient checkpointing on each module if available.
        """
        if self.custom_model is not None:
            logger.warning("Skipping gradient checkpointing for a custom model.")
            return

        def recurse_gc(module: nn.Module):
            for _, child in module.named_children():
                if hasattr(child, "gradient_checkpointing_enable"):
                    child.gradient_checkpointing_enable()
                recurse_gc(child)

        recurse_gc(self.backbone)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        if isinstance(y, tuple):
            # mixup or cutmix
            y_a, y_b, lam = y
            logits = self(x)
            if (self.loss_function == "bce_single_logit"
                    and self.num_classes == 1
                    and self.loss_fn is not None):
                y_a = y_a.float().unsqueeze(1)
                y_b = y_b.float().unsqueeze(1)
                loss_a = self.loss_fn(logits, y_a)
                loss_b = self.loss_fn(logits, y_b)
            else:
                if self.loss_fn is None:
                    loss_a = self._compute_loss(logits, y_a)
                    loss_b = self._compute_loss(logits, y_b)
                else:
                    loss_a = self._compute_loss_custom(logits, y_a)
                    loss_b = self._compute_loss_custom(logits, y_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            logits = self(x)
            if (self.loss_function == "bce_single_logit"
                    and self.num_classes == 1
                    and self.loss_fn is not None):
                y_float = y.float().unsqueeze(1)
                loss = self.loss_fn(logits, y_float)
            else:
                if self.loss_fn is None:
                    loss = self._compute_loss(logits, y)
                else:
                    loss = self._compute_loss_custom(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            logits = self(x)
            if (self.loss_function == "bce_single_logit"
                    and self.num_classes == 1
                    and self.loss_fn is not None):
                y_float = y.float().unsqueeze(1)
                loss = self.loss_fn(logits, y_float)
                prob = torch.sigmoid(logits)
                preds = (prob >= 0.5).long()
                acc = (preds.view(-1) == y).float().mean()
            else:
                if self.loss_fn is None:
                    loss = self._compute_loss(logits, y)
                else:
                    loss = self._compute_loss_custom(logits, y)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            logits = self._forward_tta(x) if self.enable_tta else self(x)

            if (self.loss_function == "bce_single_logit"
                    and self.num_classes == 1
                    and self.loss_fn is not None):
                y_float = y.float().unsqueeze(1)
                loss = self.loss_fn(logits, y_float)
                prob = torch.sigmoid(logits)
                preds = (prob >= 0.5).long()
                acc = (preds.view(-1) == y).float().mean()
            else:
                if self.loss_fn is None:
                    loss = self._compute_loss(logits, y)
                else:
                    loss = self._compute_loss_custom(logits, y)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean()

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def _forward_tta(self, x: torch.Tensor) -> torch.Tensor:
        # Basic TTA: average predictions of [original, h-flip, v-flip]
        with torch.no_grad():
            logits_list = []
            logits_list.append(self(x))
            x_hflip = torch.flip(x, dims=[3])
            logits_list.append(self(x_hflip))
            x_vflip = torch.flip(x, dims=[2])
            logits_list.append(self(x_vflip))
            mean_logits = torch.mean(torch.stack(logits_list), dim=0)
        return mean_logits

    def configure_optimizers(self):
        warmup_epochs = self.scheduler_params.pop("warmup_epochs", 0)
        monitor_metric = self.scheduler_params.pop("monitor", "val_loss")

        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "lamb":
            if not HAVE_LAMB:
                raise ImportError("LAMB optimizer not available. Please install pytorch-optimizer.")
            optimizer = LambClass(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        sch_name = self.scheduler_name.lower()
        main_scheduler = None
        if sch_name == "steplr":
            step_size = self.scheduler_params.get("step_size", 10)
            gamma = self.scheduler_params.get("gamma", 0.1)
            main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif sch_name == "reducelronplateau":
            factor = self.scheduler_params.get("factor", 0.1)
            patience = self.scheduler_params.get("patience", 5)
            main_scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
        elif sch_name == "cosineannealing":
            T_max = self.scheduler_params.get("t_max", 10)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        elif sch_name == "cycliclr":
            base_lr = self.scheduler_params.get("base_lr", 1e-4)
            max_lr = self.scheduler_params.get("max_lr", 1e-2)
            step_size_up = self.scheduler_params.get("step_size_up", 2000)
            main_scheduler = CyclicLR(
                optimizer, base_lr, max_lr, step_size_up=step_size_up, mode="triangular2"
            )
        elif sch_name == "none":
            pass

        if warmup_epochs > 0:
            def warmup_lambda(epoch: int):
                return min(1.0, float(epoch + 1) / float(warmup_epochs))
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            if main_scheduler is not None:
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs]
                )
                if isinstance(main_scheduler, ReduceLROnPlateau):
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": scheduler,
                            "monitor": monitor_metric
                        }
                    }
                return [optimizer], [scheduler]
            else:
                return [optimizer], [warmup_scheduler]
        else:
            if main_scheduler is not None:
                if isinstance(main_scheduler, ReduceLROnPlateau):
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": main_scheduler,
                            "monitor": monitor_metric
                        }
                    }
                return [optimizer], [main_scheduler]
            return optimizer

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_weighted_loss and (self.class_weights is not None):
            if any(w == 0 for w in self.class_weights):
                logger.warning("Some class weights are zero; check dataset distribution.")
            wt = torch.tensor(self.class_weights, device=logits.device, dtype=torch.float32)
            return F.cross_entropy(
                logits, targets, weight=wt, label_smoothing=self.label_smoothing
            )
        else:
            return F.cross_entropy(
                logits, targets, label_smoothing=self.label_smoothing
            )

    def _compute_loss_custom(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(self.loss_fn, FocalLoss):
            return self.loss_fn(logits, targets)
        elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            return self.loss_fn(logits, targets.float().unsqueeze(1))
        else:
            return self._compute_loss(logits, targets)

    def export_onnx(self, onnx_path: str = "model.onnx",
                    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)):
        dummy_input = torch.randn(input_size, device=self.device)
        self.eval()
        torch.onnx.export(self, dummy_input, onnx_path, export_params=True)
        logger.info(f"Model exported to ONNX: {onnx_path}")

    def export_torchscript(self, ts_path: str = "model.ts",
                           input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)):
        dummy_input = torch.randn(input_size, device=self.device)
        self.eval()
        traced = torch.jit.trace(self, dummy_input)
        traced.save(ts_path)
        logger.info(f"Model exported to TorchScript: {ts_path}")

    def export_tensorrt(self, trt_path: str = "model.trt",
                        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)):
        logger.info("TensorRT export is not implemented.")
        pass


# ----------------------------------
# PARTIAL FREEZE (ARCH-SPECIFIC)
# ----------------------------------
def apply_partial_freeze(model: MaidClassifier, freeze_config: Dict[str, bool]):
    """
    Dynamically freeze certain submodules based on freeze_config, supporting
    ResNet (conv1_bn1, layer1..4) and ConvNeXt (block0..3) in a single set of
    checkboxes. If the user checks e.g. 'conv1_bn1' but the architecture is
    ConvNeXt, we do nothing.

    If 'freeze_entire_backbone' is True, we freeze all backbone parameters
    and ignore the per-layer flags.
    """
    if model.custom_model is not None:
        logger.info("Skipping partial freeze because a custom model was loaded.")
        return

    if not model.backbone:
        # Possibly custom model has no backbone
        return

    if freeze_config.get("freeze_entire_backbone", False):
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze the entire backbone.")
        return

    arch = model.hparams["architecture"].lower()

    # -------------------
    # FOR RESNET:
    # -------------------
    if arch.startswith("resnet"):
        # If conv1_bn1 is checked, freeze conv1 + bn1
        if freeze_config.get("conv1_bn1", False):
            if hasattr(model.backbone, "conv1"):
                for p in model.backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(model.backbone, "bn1"):
                for p in model.backbone.bn1.parameters():
                    p.requires_grad = False
            logger.info("Froze conv1 + bn1 (ResNet).")

        # For layer1..layer4
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            if freeze_config.get(layer_name, False):
                layer_mod = getattr(model.backbone, layer_name, None)
                if layer_mod is not None:
                    for p in layer_mod.parameters():
                        p.requires_grad = False
                    logger.info(f"Froze {layer_name} (ResNet).")

    # -------------------
    # FOR CONVNEXT:
    # We interpret "block0..3" as model.backbone.features[0..3]
    # -------------------
    elif arch.startswith("convnext"):
        if hasattr(model.backbone, "features"):
            feats = model.backbone.features
            # block0 => features[0]
            if freeze_config.get("block0", False) and len(feats) > 0:
                for p in feats[0].parameters():
                    p.requires_grad = False
                logger.info("Froze block0 (ConvNeXt).")
            if freeze_config.get("block1", False) and len(feats) > 1:
                for p in feats[1].parameters():
                    p.requires_grad = False
                logger.info("Froze block1 (ConvNeXt).")
            if freeze_config.get("block2", False) and len(feats) > 2:
                for p in feats[2].parameters():
                    p.requires_grad = False
                logger.info("Froze block2 (ConvNeXt).")
            if freeze_config.get("block3", False) and len(feats) > 3:
                for p in feats[3].parameters():
                    p.requires_grad = False
                logger.info("Froze block3 (ConvNeXt).")

    # If user selected e.g. "conv1_bn1" but arch is not ResNet, we just ignore it
    # (no error). Similarly for "block0" if not a ConvNeXt, etc.


class CollateFnWrapper:
    def __init__(self, use_mixup: bool = False, use_cutmix: bool = False, alpha: float = 1.0):
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.alpha = alpha

    def __call__(self, batch):
        return custom_collate_fn(batch, self.use_mixup, self.use_cutmix, self.alpha)


# ============================
# 6) CALLBACKS & TRAIN LOOPS
# ============================
class ProgressCallback(Callback):
    """
    A callback to log progress each epoch,
    optionally run garbage collection,
    and optionally profile memory usage.
    """
    def __init__(self, total_epochs: int, run_gc: bool = False, profile_memory: bool = False):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        self.run_gc = run_gc
        self.profile_memory = profile_memory

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()
        logger.info("Training started...")

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        elapsed = time.time() - self.start_time
        epochs_left = self.total_epochs - current_epoch
        if current_epoch > 0:
            time_per_epoch = elapsed / current_epoch
            eta = time_per_epoch * epochs_left
        else:
            eta = 0.0

        metric_msg = (
            f"Epoch {current_epoch}/{self.total_epochs} "
            f"- ETA: {eta:.2f}s "
            f"- Train Loss: {trainer.callback_metrics.get('train_loss', 'N/A')}, "
            f"Val Loss: {trainer.callback_metrics.get('val_loss', 'N/A')}, "
            f"Val Acc: {trainer.callback_metrics.get('val_acc', 'N/A')}"
        )
        logger.info(metric_msg)

        # Clear CUDA memory if requested
        if self.run_gc:
            gc.collect()
            torch.cuda.empty_cache()

        if self.profile_memory and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            logger.info(f"[Memory Profile] GPU allocated: {allocated:.2f} MB")


def run_training_once(
    data_params: DataConfig,
    train_params: TrainConfig
) -> Tuple[str, Dict[str, Any], Optional[float]]:
    """
    Perform a single training run using the given configs.
    Returns (best_ckpt_path, test_metrics, val_loss).
    """
    root_dir = data_params.root_dir
    val_split = data_params.val_split
    test_split = data_params.test_split
    batch_size = data_params.batch_size
    allow_grayscale = data_params.allow_grayscale

    if val_split + test_split > 1.0:
        raise ValueError("Combined val+test split cannot exceed 1.0")

    # Basic fields
    max_epochs = train_params.max_epochs
    architecture = train_params.architecture
    lr = train_params.lr
    momentum = train_params.momentum
    weight_decay = train_params.weight_decay
    use_weighted_loss = train_params.use_weighted_loss
    optimizer_name = train_params.optimizer_name
    scheduler_name = train_params.scheduler_name
    scheduler_params = dict(train_params.scheduler_params)
    do_early_stopping = train_params.do_early_stopping
    early_stopping_monitor = train_params.early_stopping_monitor
    early_stopping_patience = train_params.early_stopping_patience
    early_stopping_min_delta = train_params.early_stopping_min_delta
    early_stopping_mode = train_params.early_stopping_mode

    brightness_contrast = train_params.brightness_contrast
    hue_saturation = train_params.hue_saturation
    gaussian_noise = train_params.gaussian_noise
    rotation = train_params.use_rotation
    flip = train_params.use_flip
    flip_mode = train_params.flip_mode
    flip_prob = train_params.flip_prob
    crop = train_params.use_crop
    elastic = train_params.use_elastic
    normalize_pixel_intensity = train_params.normalize_pixel_intensity
    use_grid_distortion = train_params.use_grid_distortion
    use_optical_distortion = train_params.use_optical_distortion
    use_mixup = train_params.use_mixup
    use_cutmix = train_params.use_cutmix
    mix_alpha = train_params.mix_alpha

    dropout_rate = train_params.dropout_rate
    label_smoothing = train_params.label_smoothing
    freeze_backbone = train_params.freeze_backbone
    loss_function = train_params.loss_function

    gradient_clip_val = train_params.gradient_clip_val
    use_lr_finder = train_params.use_lr_finder
    use_tensorboard = train_params.use_tensorboard
    use_mixed_precision = train_params.use_mixed_precision
    warmup_epochs = train_params.warmup_epochs
    use_inception_299 = train_params.use_inception_299

    enable_gradient_checkpointing = train_params.enable_gradient_checkpointing
    enable_grad_accum = train_params.enable_grad_accum
    accumulate_grad_batches = train_params.accumulate_grad_batches
    check_val_every_n_epoch = train_params.check_val_every_n_epoch

    freeze_config = train_params.freeze_config
    num_workers = train_params.num_workers
    val_center_crop = train_params.val_center_crop
    accept_lr_suggestion = train_params.accept_lr_suggestion
    random_crop_prob = train_params.random_crop_prob
    random_crop_scale_min = train_params.random_crop_scale_min
    random_crop_scale_max = train_params.random_crop_scale_max

    pretrained_weights = train_params.pretrained_weights
    run_gc = train_params.run_gc
    enable_tta = train_params.enable_tta
    profile_memory = train_params.profile_memory

    load_custom_model = train_params.load_custom_model
    custom_model_path = train_params.custom_model_path
    custom_arch_file = train_params.custom_architecture_file

    logger.info("Gathering samples...")
    samples, class_names = gather_samples_and_classes(root_dir)
    n_total = len(samples)
    if n_total < 2:
        raise ValueError("Dataset has insufficient images.")
    logger.info(f"Found {n_total} total images across classes.")

    # Stratified splits
    targets = [lbl for _, lbl in samples]
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
    trainval_index, test_index = next(sss_test.split(np.arange(n_total), targets))

    trainval_samples = [samples[i] for i in trainval_index]
    trainval_targets = [targets[i] for i in trainval_index]

    if val_split > 0.0:
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_split / (1.0 - test_split),
            random_state=42
        )
        train_index, val_index = next(sss_val.split(trainval_samples, trainval_targets))
    else:
        train_index = range(len(trainval_samples))
        val_index = []

    train_samples = [trainval_samples[i] for i in train_index]
    val_samples = [trainval_samples[i] for i in val_index]
    test_samples = [samples[i] for i in test_index]

    logger.info(f"Splits => Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    num_classes = len(class_names)
    logger.info(f"Detected {num_classes} classes: {class_names}")

    class_weights = None
    if use_weighted_loss and (loss_function != "bce_single_logit"):
        # Weighted CE scenario
        all_labels = [lbl for _, lbl in samples]
        label_counter = Counter(all_labels)
        freq = [label_counter[i] for i in range(num_classes)]
        if any(c == 0 for c in freq):
            logger.warning("One or more classes have zero samples.")
        class_weights = [1.0 / c if c > 0 else 0.0 for c in freq]
        logger.info(f"Using Weighted Loss: {class_weights}")

    if loss_function == "bce_single_logit":
        model_num_classes = 1
    else:
        model_num_classes = num_classes

    scheduler_params["warmup_epochs"] = warmup_epochs
    scheduler_params["monitor"] = early_stopping_monitor

    # Construct model
    model = MaidClassifier(
        architecture=architecture,
        num_classes=model_num_classes,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        use_weighted_loss=use_weighted_loss,
        class_weights=class_weights,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        scheduler_params=scheduler_params,
        dropout_rate=dropout_rate,
        label_smoothing=label_smoothing,
        freeze_backbone=freeze_backbone,
        loss_function=loss_function,
        pretrained=pretrained_weights,
        enable_tta=enable_tta,
        load_custom_model=load_custom_model,
        custom_model_path=custom_model_path,
        custom_architecture_file=custom_arch_file
    )
    model.class_names = class_names
    apply_partial_freeze(model, freeze_config)

    if enable_gradient_checkpointing:
        try:
            model.enable_gradient_checkpointing()
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    ckpt_callback = ModelCheckpoint(
        monitor=early_stopping_monitor,
        save_top_k=1,
        mode=early_stopping_mode,
        filename="best-checkpoint"
    )
    callbacks = [ckpt_callback]

    if do_early_stopping:
        early_stop = EarlyStopping(
            monitor=early_stopping_monitor,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode=early_stopping_mode
        )
        callbacks.append(early_stop)

    progress_cb = ProgressCallback(
        total_epochs=max_epochs,
        run_gc=run_gc,
        profile_memory=profile_memory
    )
    callbacks.append(progress_cb)

    logger_obj = None
    tb_log_dir = None
    if use_tensorboard:
        logger_obj = TensorBoardLogger(save_dir="tb_logs", name="experiment")
        tb_log_dir = logger_obj.log_dir

    trainer_device = "gpu" if torch.cuda.is_available() else "cpu"
    devices_to_use = torch.cuda.device_count() if torch.cuda.is_available() else 1

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=trainer_device,
        devices=devices_to_use,
        callbacks=callbacks,
        logger=logger_obj,
        gradient_clip_val=gradient_clip_val,
        precision=16 if use_mixed_precision and torch.cuda.is_available() else 32,
        accumulate_grad_batches=accumulate_grad_batches if enable_grad_accum else 1,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_progress_bar=True
    )

    # Decide final crop dimension
    if architecture.lower() == "inception" and use_inception_299:
        final_crop_dim = 299
        bigger_resize = 320
    else:
        final_crop_dim = 224
        bigger_resize = 256

    # Collate with MixUp/CutMix
    collate_fn = None
    if use_mixup or use_cutmix:
        collate_fn = CollateFnWrapper(use_mixup=use_mixup, use_cutmix=use_cutmix, alpha=mix_alpha)

    # Albumentations transforms
    train_augs = []
    if rotation:
        train_augs.append(A.Rotate(limit=30, p=0.5))
    if flip:
        if flip_mode == "horizontal":
            train_augs.append(A.HorizontalFlip(p=flip_prob))
        elif flip_mode == "vertical":
            train_augs.append(A.VerticalFlip(p=flip_prob))
        elif flip_mode == "both":
            train_augs.append(
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0)
                ], p=flip_prob)
            )
    if crop:
        train_augs.append(
            A.RandomResizedCrop(
                (final_crop_dim, final_crop_dim),
                scale=(random_crop_scale_min, random_crop_scale_max),
                p=random_crop_prob
            )
        )
    else:
        train_augs.append(A.Resize(final_crop_dim, final_crop_dim))

    if elastic:
        train_augs.append(A.ElasticTransform(p=0.2))
    if brightness_contrast:
        train_augs.append(A.RandomBrightnessContrast(p=0.5))
    if hue_saturation:
        train_augs.append(A.HueSaturationValue(p=0.5))
    if gaussian_noise:
        train_augs.append(A.GaussNoise(p=0.3))
    if use_grid_distortion:
        train_augs.append(A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3))
    if use_optical_distortion:
        train_augs.append(A.OpticalDistortion(distort_limit=0.3, shift_limit=0.2, p=0.3))

    if normalize_pixel_intensity:
        train_augs.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    train_augs.append(ToTensorV2())
    train_transform = A.Compose(train_augs)

    val_augs = [
        A.Resize(bigger_resize, bigger_resize)
    ]
    if val_center_crop:
        val_augs.append(A.CenterCrop(final_crop_dim, final_crop_dim, p=1.0))
    if normalize_pixel_intensity:
        val_augs.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    val_augs.append(ToTensorV2())
    val_transform = A.Compose(val_augs)

    test_augs = [
        A.Resize(bigger_resize, bigger_resize)
    ]
    if val_center_crop:
        test_augs.append(A.CenterCrop(final_crop_dim, final_crop_dim, p=1.0))
    if normalize_pixel_intensity:
        test_augs.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    test_augs.append(ToTensorV2())
    test_transform = A.Compose(test_augs)

    train_ds = AlbumentationsDataset(
        train_samples,
        transform=train_transform,
        classes=class_names,
        allow_grayscale=allow_grayscale
    )
    val_ds = AlbumentationsDataset(
        val_samples,
        transform=val_transform,
        classes=class_names,
        allow_grayscale=allow_grayscale
    )
    test_ds = AlbumentationsDataset(
        test_samples,
        transform=test_transform,
        classes=class_names,
        allow_grayscale=allow_grayscale
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2
    )

    # LR Finder if requested
    if use_lr_finder:
        logger.info("Running LR finder...")
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        new_lr = lr_finder.suggestion()
        logger.info(f"LR Finder suggests learning rate: {new_lr}")
        if accept_lr_suggestion:
            logger.info(f"Applying LR Finder suggestion: {new_lr}")
            model.lr = new_lr
        else:
            logger.info("User declined LR suggestion. Keeping original LR.")

    # Train
    logger.info(f"Starting training with batch_size={batch_size} on {trainer_device} (devices={devices_to_use})...")
    trainer.fit(model, train_loader, val_loader)
    logger.info("Training finished.")

    best_ckpt_path = ckpt_callback.best_model_path
    logger.info(f"Best checkpoint: {best_ckpt_path}")

    val_results = trainer.validate(model, val_loader, ckpt_path=best_ckpt_path)
    val_loss = val_results[0]["val_loss"] if len(val_results) > 0 else None

    logger.info("Running test...")
    test_results = trainer.test(ckpt_path=best_ckpt_path, dataloaders=test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = MaidClassifier.load_from_checkpoint(best_ckpt_path)
    best_model.class_names = model.class_names
    best_model.to(device)
    best_model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = best_model._forward_tta(x) if best_model.enable_tta else best_model(x)

            if (best_model.loss_function == "bce_single_logit") and best_model.num_classes == 1:
                prob = torch.sigmoid(logits)
                preds = (prob >= 0.5).long().view(-1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(prob.view(-1).cpu().numpy())
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    cm = None
    cr = None
    auc_roc = None
    try:
        unique_labels = np.unique(all_targets)
        if len(unique_labels) < 2:
            logger.warning("Only one class present in test set; skipping AUC computation.")
            cm = confusion_matrix(all_targets, all_preds)
            if (best_model.loss_function == "bce_single_logit") and best_model.num_classes == 1:
                cr = classification_report(all_targets, all_preds, zero_division=0)
            else:
                cr = classification_report(
                    all_targets, all_preds,
                    target_names=best_model.class_names,
                    zero_division=0
                )
        else:
            if (best_model.loss_function == "bce_single_logit") and best_model.num_classes == 1:
                cm = confusion_matrix(all_targets, all_preds)
                cr = classification_report(all_targets, all_preds, zero_division=0)
                auc_roc = roc_auc_score(all_targets, all_probs)
            else:
                cm = confusion_matrix(all_targets, all_preds)
                cr = classification_report(
                    all_targets, all_preds,
                    target_names=best_model.class_names,
                    zero_division=0
                )
                if best_model.num_classes == 2:
                    auc_roc = roc_auc_score(all_targets, all_probs[:, 1])
                else:
                    auc_roc = roc_auc_score(
                        all_targets, all_probs, multi_class='ovr', average='macro'
                    )
    except Exception as e:
        logger.warning(f"Could not compute some metrics: {e}")

    if cm is not None:
        logger.info(f"Confusion Matrix:\n{cm}")
    if cr is not None:
        logger.info(f"Classification Report:\n{cr}")
    if auc_roc is not None:
        logger.info(f"AUC-ROC: {auc_roc:.4f}")

    # Optionally create a confusion matrix figure
    cm_fig_path = None
    if cm is not None and HAVE_MPL:
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.tight_layout()
            cm_fig_path = "confusion_matrix.png"
            fig.savefig(cm_fig_path)
            plt.close(fig)
            logger.info(f"Confusion matrix plot saved to {cm_fig_path}")
        except Exception as e:
            logger.warning(f"Failed to plot confusion matrix: {e}")

    test_metrics = {
        "test_results": test_results,
        "confusion_matrix": cm.tolist() if cm is not None else None,
        "class_report": cr,
        "auc_roc": auc_roc,
        "tb_log_dir": tb_log_dir,
        "cm_fig_path": cm_fig_path
    }
    return best_ckpt_path, test_metrics, val_loss


# =======================
# 7) THREAD CLASSES
# =======================
class TrainThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str, dict)

    def __init__(self, data_params: Dict[str, Any], train_params: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.data_params = data_params
        self.train_params = train_params

        # For binary-search batch-size fallback
        self.original_bs = self.data_params["batch_size"]

    def run(self):
        try:
            seed_everything(42, workers=True)
            self._execute_training()
        except (ValueError, ImportError, RuntimeError) as e:
            err_msg = f"ERROR during training: {e}\n{traceback.format_exc()}"
            self.log_signal.emit(err_msg)
            self.done_signal.emit("ERROR", {})

    def _execute_training(self):
        data_config = DataConfig(**self.data_params)
        train_config = TrainConfig(**self.train_params)
        try:
            best_ckpt_path, test_metrics, _ = run_training_once(
                data_config, train_config
            )
            self.done_signal.emit(best_ckpt_path, test_metrics)
        except RuntimeError as re:
            if "CUDA out of memory" in str(re):
                self.log_signal.emit("GPU OOM encountered.\n")
                self._auto_adjust_batch_size()
            else:
                raise

    def _auto_adjust_batch_size(self):
        """
        Perform a binary search for the maximum feasible batch size
        between 1 and original_bs.
        """
        low, high = 1, self.original_bs
        feasible_bs = 1
        while low <= high:
            mid = (low + high) // 2
            self.data_params["batch_size"] = mid
            self.log_signal.emit(
                f"Trying batch_size={mid} in binary search.\n"
            )
            try:
                best_ckpt_path, test_metrics, _ = run_training_once(
                    DataConfig(**self.data_params),
                    TrainConfig(**self.train_params)
                )
                feasible_bs = mid
                low = mid + 1
                # If successful, record result but try bigger
            except RuntimeError as re:
                if "CUDA out of memory" in str(re):
                    high = mid - 1
                else:
                    # some other error
                    self.log_signal.emit(f"Other error in attempt: {re}\n")
                    break

        if feasible_bs == 1 and low == 1:
            self.log_signal.emit(
                "All attempts failed at batch_size=1. Training cannot proceed.\n"
            )
            self.done_signal.emit("ERROR", {})
        else:
            self.log_signal.emit(
                f"Binary search found feasible batch_size={feasible_bs}.\n"
            )
            # Final run with that feasible batch size
            self.data_params["batch_size"] = feasible_bs
            try:
                best_ckpt_path, test_metrics, _ = run_training_once(
                    DataConfig(**self.data_params),
                    TrainConfig(**self.train_params)
                )
                self.done_signal.emit(best_ckpt_path, test_metrics)
            except Exception as e:
                err_msg = f"ERROR (final attempt) during training: {e}\n{traceback.format_exc()}"
                self.log_signal.emit(err_msg)
                self.done_signal.emit("ERROR", {})


class OptunaTrainThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str, dict)

    def __init__(
        self,
        data_params: Dict[str, Any],
        base_train_params: Dict[str, Any],
        optuna_n_trials: int,
        optuna_timeout: int,
        use_test_metric_for_optuna: bool = False
    ):
        super().__init__()
        self.data_params = data_params
        self.base_train_params = base_train_params
        self.optuna_n_trials = optuna_n_trials
        self.optuna_timeout = optuna_timeout
        self.use_test_metric_for_optuna = use_test_metric_for_optuna

    def run(self):
        try:
            seed_everything(42, workers=True)
            study = optuna.create_study(direction="minimize")

            def objective(trial: optuna.Trial) -> float:
                trial_train_params = dict(self.base_train_params)
                lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
                dropout = trial.suggest_float("dropout_rate", 0.0, 0.7)
                optimizer_name = trial.suggest_categorical("optimizer_name", ["adam", "sgd", "adamw"])
                scheduler_name = trial.suggest_categorical("scheduler_name", ["none", "steplr", "cosineannealing"])

                trial_train_params["lr"] = lr
                trial_train_params["dropout_rate"] = dropout
                trial_train_params["optimizer_name"] = optimizer_name
                trial_train_params["scheduler_name"] = scheduler_name

                msg = (
                    f"Trial {trial.number}: "
                    f"lr={lr}, dropout={dropout}, opt={optimizer_name}, sch={scheduler_name}"
                )
                logger.info(msg)
                self.log_signal.emit(msg + "\n")

                best_ckpt_path, test_metrics, val_loss = run_training_once(
                    DataConfig(**self.data_params),
                    TrainConfig(**trial_train_params)
                )
                if not test_metrics:
                    return 9999.0

                if self.use_test_metric_for_optuna:
                    test_loss = 9999.0
                    if "test_results" in test_metrics:
                        tr_ = test_metrics["test_results"]
                        if len(tr_) > 0 and "test_loss" in tr_[0]:
                            test_loss = tr_[0]["test_loss"]
                    return test_loss
                else:
                    return val_loss if val_loss is not None else 9999.0

            study.optimize(
                objective,
                n_trials=self.optuna_n_trials,
                timeout=self.optuna_timeout if self.optuna_timeout > 0 else None
            )

            best_trial = study.best_trial
            self.log_signal.emit(
                f"Optuna best trial: {best_trial.number}, value={best_trial.value}\n"
            )
            self.log_signal.emit(f"Best params: {best_trial.params}\n")

            # Re-train final model
            best_params = dict(self.base_train_params)
            best_params["lr"] = best_trial.params["lr"]
            best_params["dropout_rate"] = best_trial.params["dropout_rate"]
            best_params["optimizer_name"] = best_trial.params["optimizer_name"]
            best_params["scheduler_name"] = best_trial.params["scheduler_name"]

            self.log_signal.emit("Re-training final model with best hyperparams...\n")
            best_ckpt_path, metrics_dict, _ = run_training_once(
                DataConfig(**self.data_params),
                TrainConfig(**best_params)
            )
            self.done_signal.emit(best_ckpt_path, metrics_dict)
        except (ValueError, ImportError, RuntimeError) as e:
            err_msg = f"ERROR during Optuna tuning: {e}\n{traceback.format_exc()}"
            self.log_signal.emit(err_msg)
            self.done_signal.emit("ERROR", {})


# =========================
# 8) MAIN PLUGIN CLASS (GUI)
# =========================
class Plugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.plugin_name = "Training"

        def tr(text):
            return text
        self.tr = tr

        self.widget_main: Optional[QWidget] = None
        self.text_log: Optional[QTextEdit] = None
        self.progress_bar: Optional[QProgressBar] = None

        # Data / split inputs
        self.train_data_dir_edit: Optional[QLineEdit] = None
        self.val_split_spin: Optional[QDoubleSpinBox] = None
        self.test_split_spin: Optional[QDoubleSpinBox] = None
        self.cb_allow_grayscale: Optional[QCheckBox] = None

        self.arch_combo: Optional[QComboBox] = None
        self.weighted_loss_cb: Optional[QCheckBox] = None
        self.cb_normalize_pixel_intensity: Optional[QCheckBox] = None
        self.cb_inception_299: Optional[QCheckBox] = None
        self.loss_combo: Optional[QComboBox] = None

        self.cb_bright_contrast: Optional[QCheckBox] = None
        self.cb_hue_sat: Optional[QCheckBox] = None
        self.cb_gauss_noise: Optional[QCheckBox] = None
        self.cb_rotation: Optional[QCheckBox] = None
        self.cb_flip: Optional[QCheckBox] = None
        self.flip_mode_combo: Optional[QComboBox] = None
        self.flip_prob_spin: Optional[QDoubleSpinBox] = None
        self.cb_crop: Optional[QCheckBox] = None
        self.cb_elastic: Optional[QCheckBox] = None
        self.cb_grid_distortion: Optional[QCheckBox] = None
        self.cb_optical_distortion: Optional[QCheckBox] = None

        self.cb_mixup: Optional[QCheckBox] = None
        self.cb_cutmix: Optional[QCheckBox] = None
        self.mix_alpha_spin: Optional[QDoubleSpinBox] = None

        self.random_crop_prob_spin: Optional[QDoubleSpinBox] = None
        self.random_crop_scale_min_spin: Optional[QDoubleSpinBox] = None
        self.random_crop_scale_max_spin: Optional[QDoubleSpinBox] = None

        self.lr_spin: Optional[QDoubleSpinBox] = None
        self.momentum_spin: Optional[QDoubleSpinBox] = None
        self.wd_spin: Optional[QDoubleSpinBox] = None
        self.optimizer_combo: Optional[QComboBox] = None
        self.scheduler_combo: Optional[QComboBox] = None
        self.epochs_spin: Optional[QSpinBox] = None
        self.batch_spin: Optional[QSpinBox] = None
        self.cb_early_stopping: Optional[QCheckBox] = None
        self.scheduler_params_edit: Optional[QLineEdit] = None
        self.dropout_spin: Optional[QDoubleSpinBox] = None
        self.label_smoothing_spin: Optional[QDoubleSpinBox] = None
        self.cb_freeze_backbone: Optional[QCheckBox] = None
        self.clip_val_spin: Optional[QDoubleSpinBox] = None
        self.cb_lr_finder: Optional[QCheckBox] = None
        self.cb_accept_lr_suggestion: Optional[QCheckBox] = None
        self.cb_tensorboard: Optional[QCheckBox] = None
        self.cb_mixed_precision: Optional[QCheckBox] = None
        self.warmup_epochs_spin: Optional[QSpinBox] = None

        self.num_workers_spin: Optional[QSpinBox] = None
        self.cb_grad_checkpoint: Optional[QCheckBox] = None
        self.cb_grad_accum: Optional[QCheckBox] = None
        self.accum_batches_spin: Optional[QSpinBox] = None
        self.check_val_every_n_epoch: Optional[QSpinBox] = None

        # The following checkboxes apply to both ResNet & ConvNeXt
        # We'll just unify them in the code
        self.cb_freeze_conv1_bn1: Optional[QCheckBox] = None
        self.cb_freeze_layer1: Optional[QCheckBox] = None
        self.cb_freeze_layer2: Optional[QCheckBox] = None
        self.cb_freeze_layer3: Optional[QCheckBox] = None
        self.cb_freeze_layer4: Optional[QCheckBox] = None
        self.cb_freeze_convnext_block0: Optional[QCheckBox] = None
        self.cb_freeze_convnext_block1: Optional[QCheckBox] = None
        self.cb_freeze_convnext_block2: Optional[QCheckBox] = None
        self.cb_freeze_convnext_block3: Optional[QCheckBox] = None
        self.cb_val_center_crop: Optional[QCheckBox] = None

        self.es_monitor_combo: Optional[QComboBox] = None
        self.es_patience_spin: Optional[QSpinBox] = None
        self.es_min_delta_spin: Optional[QDoubleSpinBox] = None
        self.es_mode_combo: Optional[QComboBox] = None

        self.cb_pretrained_weights: Optional[QCheckBox] = None
        self.cb_run_gc: Optional[QCheckBox] = None
        self.cb_enable_tta: Optional[QCheckBox] = None
        self.cb_profile_memory: Optional[QCheckBox] = None

        # Custom model
        self.cb_load_custom_model: Optional[QCheckBox] = None
        self.custom_model_path_edit: Optional[QLineEdit] = None
        self.custom_arch_file_edit: Optional[QLineEdit] = None

        self.btn_train: Optional[QPushButton] = None
        self.btn_export_results: Optional[QPushButton] = None
        self.btn_tune_optuna: Optional[QPushButton] = None
        self.cb_optuna_use_test_metric: Optional[QCheckBox] = None
        self.optuna_trials_spin: Optional[QSpinBox] = None
        self.optuna_timeout_spin: Optional[QSpinBox] = None

        self.train_thread: Optional[TrainThread] = None
        self.optuna_thread: Optional[OptunaTrainThread] = None

        self.best_ckpt_path: Optional[str] = None
        self.last_test_metrics: Optional[Dict[str, Any]] = None

    def create_tab(self) -> QWidget:
        """
        Build the entire UI layout for this plugin tab.
        """
        self.widget_main = QWidget()
        layout = QVBoxLayout(self.widget_main)

        # 1) DATA PATHS AND SPLITS
        h_data = QHBoxLayout()
        self.train_data_dir_edit = QLineEdit()
        btn_browse_data = QPushButton(self.tr("Browse Data Folder..."))
        btn_browse_data.clicked.connect(self.browse_dataset_folder)
        h_data.addWidget(QLabel(self.tr("Dataset Folder:")))
        h_data.addWidget(self.train_data_dir_edit)
        h_data.addWidget(btn_browse_data)
        layout.addLayout(h_data)

        h_splits = QHBoxLayout()
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0, 100)
        self.val_split_spin.setValue(15.0)
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0, 100)
        self.test_split_spin.setValue(15.0)

        h_splits.addWidget(QLabel(self.tr("Val Split (%)")))
        h_splits.addWidget(self.val_split_spin)
        h_splits.addWidget(QLabel(self.tr("Test Split (%)")))
        h_splits.addWidget(self.test_split_spin)
        layout.addLayout(h_splits)

        self.cb_allow_grayscale = QCheckBox(self.tr("Allow Grayscale Images"))
        layout.addWidget(self.cb_allow_grayscale)

        # 2) ARCH + BASIC OPTIONS
        h_arch = QHBoxLayout()
        self.arch_combo = QComboBox()
        self.arch_combo.addItems([
            "resnet18", "resnet50", "resnet101",
            "densenet", "vgg", "inception", "mobilenet",
            "efficientnet_b0", "convnext_tiny", "convnext_large"
        ])
        self.weighted_loss_cb = QCheckBox(self.tr("Weighted Loss"))
        self.cb_normalize_pixel_intensity = QCheckBox(self.tr("Normalize Pixel"))
        self.cb_inception_299 = QCheckBox(self.tr("Use 299 for Inception"))

        h_arch.addWidget(QLabel(self.tr("Architecture:")))
        h_arch.addWidget(self.arch_combo)
        h_arch.addWidget(self.weighted_loss_cb)
        h_arch.addWidget(self.cb_normalize_pixel_intensity)
        h_arch.addWidget(self.cb_inception_299)
        layout.addLayout(h_arch)

        # LOSS
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["cross_entropy", "focal", "bce", "bce_single_logit"])

        # 3) AUGMENTATIONS
        group_aug = QHBoxLayout()
        self.cb_bright_contrast = QCheckBox(self.tr("Brightness/Contrast"))
        self.cb_hue_sat = QCheckBox(self.tr("Hue/Sat"))
        self.cb_gauss_noise = QCheckBox(self.tr("GaussNoise"))
        self.cb_rotation = QCheckBox(self.tr("Rotation"))
        group_aug.addWidget(self.cb_bright_contrast)
        group_aug.addWidget(self.cb_hue_sat)
        group_aug.addWidget(self.cb_gauss_noise)
        group_aug.addWidget(self.cb_rotation)
        layout.addLayout(group_aug)

        group_aug2 = QHBoxLayout()
        self.cb_flip = QCheckBox(self.tr("Flipping"))
        self.flip_mode_combo = QComboBox()
        self.flip_mode_combo.addItems(["horizontal", "vertical", "both"])
        self.flip_prob_spin = QDoubleSpinBox()
        self.flip_prob_spin.setRange(0.0, 1.0)
        self.flip_prob_spin.setValue(0.5)
        self.cb_crop = QCheckBox(self.tr("Random Crop"))
        self.cb_elastic = QCheckBox(self.tr("Elastic"))

        group_aug2.addWidget(self.cb_flip)
        group_aug2.addWidget(QLabel(self.tr("Mode:")))
        group_aug2.addWidget(self.flip_mode_combo)
        group_aug2.addWidget(QLabel(self.tr("Prob:")))
        group_aug2.addWidget(self.flip_prob_spin)
        group_aug2.addWidget(self.cb_crop)
        group_aug2.addWidget(self.cb_elastic)
        layout.addLayout(group_aug2)

        group_aug3 = QHBoxLayout()
        group_aug3.addWidget(QLabel(self.tr("Random Crop p:")))
        self.random_crop_prob_spin = QDoubleSpinBox()
        self.random_crop_prob_spin.setRange(0.0, 1.0)
        self.random_crop_prob_spin.setValue(1.0)
        group_aug3.addWidget(self.random_crop_prob_spin)
        group_aug3.addWidget(QLabel(self.tr("Scale Min:")))
        self.random_crop_scale_min_spin = QDoubleSpinBox()
        self.random_crop_scale_min_spin.setRange(0.0, 1.0)
        self.random_crop_scale_min_spin.setValue(0.8)
        group_aug3.addWidget(self.random_crop_scale_min_spin)
        group_aug3.addWidget(QLabel(self.tr("Scale Max:")))
        self.random_crop_scale_max_spin = QDoubleSpinBox()
        self.random_crop_scale_max_spin.setRange(0.0, 1.0)
        self.random_crop_scale_max_spin.setValue(1.0)
        group_aug3.addWidget(self.random_crop_scale_max_spin)
        layout.addLayout(group_aug3)

        group_aug4 = QHBoxLayout()
        self.cb_grid_distortion = QCheckBox(self.tr("GridDistortion"))
        self.cb_optical_distortion = QCheckBox(self.tr("OpticalDistortion"))
        self.cb_mixup = QCheckBox(self.tr("MixUp"))
        self.cb_cutmix = QCheckBox(self.tr("CutMix"))
        self.mix_alpha_spin = QDoubleSpinBox()
        self.mix_alpha_spin.setRange(0.0, 5.0)
        self.mix_alpha_spin.setValue(1.0)
        group_aug4.addWidget(self.cb_grid_distortion)
        group_aug4.addWidget(self.cb_optical_distortion)
        group_aug4.addWidget(self.cb_mixup)
        group_aug4.addWidget(self.cb_cutmix)
        group_aug4.addWidget(QLabel(self.tr("alpha:")))
        group_aug4.addWidget(self.mix_alpha_spin)
        layout.addLayout(group_aug4)

        # 4) HYPERPARAMS
        h_params = QHBoxLayout()
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setValue(1e-4)
        self.lr_spin.setToolTip("Learning rate")
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setValue(0.9)
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0, 1.0)
        self.wd_spin.setDecimals(6)
        self.wd_spin.setValue(1e-4)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd", "adamw", "lamb"])
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["none", "steplr", "reducelronplateau", "cosineannealing", "cycliclr"])
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(5)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(8)
        self.cb_early_stopping = QCheckBox(self.tr("Early Stopping"))

        h_params.addWidget(QLabel(self.tr("LR:")))
        h_params.addWidget(self.lr_spin)
        h_params.addWidget(QLabel(self.tr("Momentum:")))
        h_params.addWidget(self.momentum_spin)
        h_params.addWidget(QLabel(self.tr("WD:")))
        h_params.addWidget(self.wd_spin)
        h_params.addWidget(QLabel(self.tr("Opt:")))
        h_params.addWidget(self.optimizer_combo)
        h_params.addWidget(QLabel(self.tr("Sched:")))
        h_params.addWidget(self.scheduler_combo)
        h_params.addWidget(QLabel(self.tr("Epochs:")))
        h_params.addWidget(self.epochs_spin)
        h_params.addWidget(QLabel(self.tr("Batch:")))
        h_params.addWidget(self.batch_spin)
        h_params.addWidget(self.cb_early_stopping)
        layout.addLayout(h_params)

        # Scheduler params
        self.scheduler_params_edit = QLineEdit("step_size=10,gamma=0.1")
        self.scheduler_params_edit.setToolTip("Scheduler parameters (key=val, comma-separated).")
        layout.addWidget(QLabel(self.tr("Scheduler Params:")))
        layout.addWidget(self.scheduler_params_edit)

        h_reg = QHBoxLayout()
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setValue(0.0)
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.9)
        self.label_smoothing_spin.setValue(0.0)
        self.cb_freeze_backbone = QCheckBox(self.tr("Freeze Entire Backbone"))
        self.clip_val_spin = QDoubleSpinBox()
        self.clip_val_spin.setRange(0.0, 10.0)
        self.clip_val_spin.setValue(0.0)

        h_reg.addWidget(QLabel(self.tr("Dropout:")))
        h_reg.addWidget(self.dropout_spin)
        h_reg.addWidget(QLabel(self.tr("LabelSmooth:")))
        h_reg.addWidget(self.label_smoothing_spin)
        h_reg.addWidget(self.cb_freeze_backbone)
        h_reg.addWidget(QLabel(self.tr("Loss:")))
        h_reg.addWidget(self.loss_combo)
        h_reg.addWidget(QLabel(self.tr("ClipVal:")))
        h_reg.addWidget(self.clip_val_spin)
        layout.addLayout(h_reg)

        # 5) ADVANCED
        h_monitor = QHBoxLayout()
        self.cb_lr_finder = QCheckBox(self.tr("LR Finder"))
        self.cb_accept_lr_suggestion = QCheckBox(self.tr("Accept LR Suggestion?"))
        self.cb_tensorboard = QCheckBox(self.tr("TensorBoard Logger"))
        self.cb_mixed_precision = QCheckBox(self.tr("Mixed Precision"))
        self.warmup_epochs_spin = QSpinBox()
        self.warmup_epochs_spin.setRange(0, 100)
        self.warmup_epochs_spin.setValue(0)

        h_monitor.addWidget(self.cb_lr_finder)
        h_monitor.addWidget(self.cb_accept_lr_suggestion)
        h_monitor.addWidget(self.cb_tensorboard)
        h_monitor.addWidget(self.cb_mixed_precision)
        h_monitor.addWidget(QLabel(self.tr("Warmup:")))
        h_monitor.addWidget(self.warmup_epochs_spin)
        layout.addLayout(h_monitor)

        h_new1 = QHBoxLayout()
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 32)
        self.num_workers_spin.setValue(8)
        self.cb_grad_checkpoint = QCheckBox(self.tr("Gradient Checkpointing"))
        self.cb_grad_accum = QCheckBox(self.tr("Gradient Accumulation"))
        self.accum_batches_spin = QSpinBox()
        self.accum_batches_spin.setRange(1, 64)
        self.accum_batches_spin.setValue(2)
        self.accum_batches_spin.setEnabled(False)

        def toggle_accum_spin():
            self.accum_batches_spin.setEnabled(self.cb_grad_accum.isChecked())
        self.cb_grad_accum.stateChanged.connect(toggle_accum_spin)

        h_new1.addWidget(QLabel(self.tr("Workers:")))
        h_new1.addWidget(self.num_workers_spin)
        h_new1.addWidget(self.cb_grad_checkpoint)
        h_new1.addWidget(self.cb_grad_accum)
        h_new1.addWidget(QLabel(self.tr("Accumulate Batches:")))
        h_new1.addWidget(self.accum_batches_spin)
        layout.addLayout(h_new1)

        h_new2 = QHBoxLayout()
        self.check_val_every_n_epoch = QSpinBox()
        self.check_val_every_n_epoch.setRange(1, 50)
        self.check_val_every_n_epoch.setValue(1)

        h_new2.addWidget(QLabel(self.tr("Val every N epoch:")))
        h_new2.addWidget(self.check_val_every_n_epoch)

        self.btn_train = QPushButton(self.tr("Start Training"))
        self.btn_train.clicked.connect(self.start_training)
        h_new2.addWidget(self.btn_train)
        layout.addLayout(h_new2)

        # 6) PARTIAL FREEZE CONTROLS (For ResNet & ConvNeXt)
        group_resnet_freeze = QHBoxLayout()
        group_resnet_freeze.addWidget(QLabel(self.tr("ResNet / ConvNeXt Freeze:")))
        self.cb_freeze_conv1_bn1 = QCheckBox(self.tr("conv1+bn1"))
        self.cb_freeze_layer1 = QCheckBox(self.tr("layer1"))
        self.cb_freeze_layer2 = QCheckBox(self.tr("layer2"))
        self.cb_freeze_layer3 = QCheckBox(self.tr("layer3"))
        self.cb_freeze_layer4 = QCheckBox(self.tr("layer4"))
        self.cb_freeze_convnext_block0 = QCheckBox(self.tr("block0"))
        self.cb_freeze_convnext_block1 = QCheckBox(self.tr("block1"))
        self.cb_freeze_convnext_block2 = QCheckBox(self.tr("block2"))
        self.cb_freeze_convnext_block3 = QCheckBox(self.tr("block3"))

        group_resnet_freeze.addWidget(self.cb_freeze_conv1_bn1)
        group_resnet_freeze.addWidget(self.cb_freeze_layer1)
        group_resnet_freeze.addWidget(self.cb_freeze_layer2)
        group_resnet_freeze.addWidget(self.cb_freeze_layer3)
        group_resnet_freeze.addWidget(self.cb_freeze_layer4)
        group_resnet_freeze.addWidget(self.cb_freeze_convnext_block0)
        group_resnet_freeze.addWidget(self.cb_freeze_convnext_block1)
        group_resnet_freeze.addWidget(self.cb_freeze_convnext_block2)
        group_resnet_freeze.addWidget(self.cb_freeze_convnext_block3)
        layout.addLayout(group_resnet_freeze)

        self.cb_val_center_crop = QCheckBox(self.tr("Center Crop for Validation/Test"))
        layout.addWidget(self.cb_val_center_crop)

        # 7) EARLY STOPPING
        es_layout = QHBoxLayout()
        es_layout.addWidget(QLabel(self.tr("ES Monitor:")))
        self.es_monitor_combo = QComboBox()
        self.es_monitor_combo.addItems(["val_loss", "val_acc"])
        es_layout.addWidget(self.es_monitor_combo)

        es_layout.addWidget(QLabel(self.tr("Patience:")))
        self.es_patience_spin = QSpinBox()
        self.es_patience_spin.setRange(1, 20)
        self.es_patience_spin.setValue(5)
        es_layout.addWidget(self.es_patience_spin)

        es_layout.addWidget(QLabel(self.tr("Min Delta:")))
        self.es_min_delta_spin = QDoubleSpinBox()
        self.es_min_delta_spin.setRange(0.0, 1.0)
        self.es_min_delta_spin.setDecimals(4)
        self.es_min_delta_spin.setSingleStep(0.0001)
        self.es_min_delta_spin.setValue(0.0)
        es_layout.addWidget(self.es_min_delta_spin)

        es_layout.addWidget(QLabel(self.tr("Mode:")))
        self.es_mode_combo = QComboBox()
        self.es_mode_combo.addItems(["min", "max"])
        es_layout.addWidget(self.es_mode_combo)
        layout.addLayout(es_layout)

        # 8) OPTUNA
        self.cb_optuna_use_test_metric = QCheckBox(self.tr("Use Test Loss as Optuna Objective?"))
        self.cb_optuna_use_test_metric.setChecked(False)
        layout.addWidget(self.cb_optuna_use_test_metric)

        h_optuna = QHBoxLayout()
        self.btn_tune_optuna = QPushButton(self.tr("Tune with Optuna"))
        self.btn_tune_optuna.clicked.connect(self.start_optuna_tuning)
        self.optuna_trials_spin = QSpinBox()
        self.optuna_trials_spin.setRange(1, 100)
        self.optuna_trials_spin.setValue(5)
        self.optuna_timeout_spin = QSpinBox()
        self.optuna_timeout_spin.setRange(0, 100000)
        self.optuna_timeout_spin.setValue(0)
        h_optuna.addWidget(self.btn_tune_optuna)
        h_optuna.addWidget(QLabel(self.tr("Trials:")))
        h_optuna.addWidget(self.optuna_trials_spin)
        h_optuna.addWidget(QLabel(self.tr("Timeout (sec):")))
        h_optuna.addWidget(self.optuna_timeout_spin)
        layout.addLayout(h_optuna)

        # 9) EXTRA OPTIONS
        extra_layout = QHBoxLayout()
        self.cb_pretrained_weights = QCheckBox(self.tr("Use Pretrained Weights"))
        self.cb_pretrained_weights.setChecked(True)
        self.cb_run_gc = QCheckBox(self.tr("Run GC Each Epoch"))
        self.cb_enable_tta = QCheckBox(self.tr("Enable TTA"))
        self.cb_profile_memory = QCheckBox(self.tr("Profile Memory"))
        extra_layout.addWidget(self.cb_pretrained_weights)
        extra_layout.addWidget(self.cb_run_gc)
        extra_layout.addWidget(self.cb_enable_tta)
        extra_layout.addWidget(self.cb_profile_memory)
        layout.addLayout(extra_layout)

        # 10) CUSTOM MODEL
        custom_layout = QHBoxLayout()
        self.cb_load_custom_model = QCheckBox(self.tr("Load Custom Model"))
        self.custom_model_path_edit = QLineEdit()
        btn_browse_custom_model = QPushButton(self.tr("Browse Model..."))
        btn_browse_custom_model.clicked.connect(self.browse_custom_model)
        self.custom_arch_file_edit = QLineEdit()
        browse_arch_btn = QPushButton(self.tr("Browse Arch File..."))
        browse_arch_btn.clicked.connect(self.browse_arch_file)

        custom_layout.addWidget(self.cb_load_custom_model)
        custom_layout.addWidget(QLabel(self.tr("Weights Path:")))
        custom_layout.addWidget(self.custom_model_path_edit)
        custom_layout.addWidget(btn_browse_custom_model)
        custom_layout.addWidget(QLabel(self.tr("Arch File:")))
        custom_layout.addWidget(self.custom_arch_file_edit)
        custom_layout.addWidget(browse_arch_btn)
        layout.addLayout(custom_layout)

        # 11) EXPORT + LOG
        self.btn_export_results = QPushButton(self.tr("Export Results"))
        self.btn_export_results.setEnabled(False)
        self.btn_export_results.clicked.connect(self.export_all_results)
        layout.addWidget(self.btn_export_results)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        layout.addWidget(self.text_log)

        return self.widget_main

    def browse_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, self.tr("Select Dataset Folder"))
        if folder:
            self.train_data_dir_edit.setText(folder)

    def browse_custom_model(self):
        fpath, _ = QFileDialog.getOpenFileName(self.widget_main, self.tr("Select Custom Model File"), filter="*.pth *.pt")
        if fpath:
            self.custom_model_path_edit.setText(fpath)

    def browse_arch_file(self):
        fpath, _ = QFileDialog.getOpenFileName(self.widget_main, self.tr("Select Architecture Python File"), filter="*.py")
        if fpath:
            self.custom_arch_file_edit.setText(fpath)

    def _parse_common_params(self) -> Tuple[DataConfig, TrainConfig]:
        dataset_dir = self.train_data_dir_edit.text().strip()
        if not os.path.isdir(dataset_dir):
            raise ValueError(self.tr("Invalid dataset folder."))

        val_split = self.val_split_spin.value() / 100.0
        test_split = self.test_split_spin.value() / 100.0
        if val_split + test_split > 1.0:
            raise ValueError(self.tr("val + test split cannot exceed 100%."))

        scheduler_params_str = self.scheduler_params_edit.text().strip()
        scheduler_params = {}
        if scheduler_params_str:
            parts = scheduler_params_str.split(",")
            for part in parts:
                if "=" not in part:
                    raise ValueError(self.tr(f"Invalid scheduler param format: {part}"))
                k, v = part.split("=")
                k, v = k.strip(), v.strip()
                float_val = float(v)
                scheduler_params[k] = float_val

        data_config = DataConfig(
            root_dir=dataset_dir,
            val_split=val_split,
            test_split=test_split,
            batch_size=self.batch_spin.value(),
            allow_grayscale=self.cb_allow_grayscale.isChecked()
        )

        freeze_config = {
            "freeze_entire_backbone": self.cb_freeze_backbone.isChecked(),
            "conv1_bn1": self.cb_freeze_conv1_bn1.isChecked(),
            "layer1": self.cb_freeze_layer1.isChecked(),
            "layer2": self.cb_freeze_layer2.isChecked(),
            "layer3": self.cb_freeze_layer3.isChecked(),
            "layer4": self.cb_freeze_layer4.isChecked(),
            "block0": self.cb_freeze_convnext_block0.isChecked(),
            "block1": self.cb_freeze_convnext_block1.isChecked(),
            "block2": self.cb_freeze_convnext_block2.isChecked(),
            "block3": self.cb_freeze_convnext_block3.isChecked(),
        }

        train_config = TrainConfig(
            max_epochs=self.epochs_spin.value(),
            architecture=self.arch_combo.currentText(),
            lr=self.lr_spin.value(),
            momentum=self.momentum_spin.value(),
            weight_decay=self.wd_spin.value(),
            use_weighted_loss=self.weighted_loss_cb.isChecked(),
            optimizer_name=self.optimizer_combo.currentText(),
            scheduler_name=self.scheduler_combo.currentText(),
            scheduler_params=scheduler_params,
            do_early_stopping=self.cb_early_stopping.isChecked(),

            early_stopping_monitor=self.es_monitor_combo.currentText(),
            early_stopping_patience=self.es_patience_spin.value(),
            early_stopping_min_delta=self.es_min_delta_spin.value(),
            early_stopping_mode=self.es_mode_combo.currentText(),

            brightness_contrast=self.cb_bright_contrast.isChecked(),
            hue_saturation=self.cb_hue_sat.isChecked(),
            gaussian_noise=self.cb_gauss_noise.isChecked(),
            use_rotation=self.cb_rotation.isChecked(),
            use_flip=self.cb_flip.isChecked(),
            flip_mode=self.flip_mode_combo.currentText(),
            flip_prob=self.flip_prob_spin.value(),
            use_crop=self.cb_crop.isChecked(),
            use_elastic=self.cb_elastic.isChecked(),
            normalize_pixel_intensity=self.cb_normalize_pixel_intensity.isChecked(),

            use_grid_distortion=self.cb_grid_distortion.isChecked(),
            use_optical_distortion=self.cb_optical_distortion.isChecked(),
            use_mixup=self.cb_mixup.isChecked(),
            use_cutmix=self.cb_cutmix.isChecked(),
            mix_alpha=self.mix_alpha_spin.value(),

            dropout_rate=self.dropout_spin.value(),
            label_smoothing=self.label_smoothing_spin.value(),
            freeze_backbone=self.cb_freeze_backbone.isChecked(),
            loss_function=self.loss_combo.currentText(),

            gradient_clip_val=self.clip_val_spin.value(),
            use_lr_finder=self.cb_lr_finder.isChecked(),
            accept_lr_suggestion=self.cb_accept_lr_suggestion.isChecked(),
            use_tensorboard=self.cb_tensorboard.isChecked(),
            use_mixed_precision=self.cb_mixed_precision.isChecked(),
            warmup_epochs=self.warmup_epochs_spin.value(),
            use_inception_299=self.cb_inception_299.isChecked(),

            enable_gradient_checkpointing=self.cb_grad_checkpoint.isChecked(),
            enable_grad_accum=self.cb_grad_accum.isChecked(),
            accumulate_grad_batches=self.accum_batches_spin.value(),
            check_val_every_n_epoch=self.check_val_every_n_epoch.value(),

            freeze_config=freeze_config,
            num_workers=self.num_workers_spin.value(),
            val_center_crop=self.cb_val_center_crop.isChecked(),

            random_crop_prob=self.random_crop_prob_spin.value(),
            random_crop_scale_min=self.random_crop_scale_min_spin.value(),
            random_crop_scale_max=self.random_crop_scale_max_spin.value(),

            pretrained_weights=self.cb_pretrained_weights.isChecked(),
            run_gc=self.cb_run_gc.isChecked(),
            enable_tta=self.cb_enable_tta.isChecked(),
            profile_memory=self.cb_profile_memory.isChecked(),

            load_custom_model=self.cb_load_custom_model.isChecked(),
            custom_model_path=self.custom_model_path_edit.text().strip(),
            custom_architecture_file=self.custom_arch_file_edit.text().strip()
        )

        return data_config, train_config

    def start_training(self):
        try:
            data_config, train_config = self._parse_common_params()
        except ValueError as e:
            QMessageBox.warning(self.widget_main, self.tr("Invalid Input"), str(e))
            return

        self.btn_train.setEnabled(False)
        self.btn_tune_optuna.setEnabled(False)
        self.btn_export_results.setEnabled(False)
        self.progress_bar.setVisible(True)

        data_dict = data_config.__dict__
        train_dict = train_config.__dict__

        self.train_thread = TrainThread(data_dict, train_dict)
        self.train_thread.log_signal.connect(self.append_log)
        self.train_thread.done_signal.connect(self.train_finished)
        self.train_thread.start()

    def start_optuna_tuning(self):
        try:
            data_config, base_train_config = self._parse_common_params()
        except ValueError as e:
            QMessageBox.warning(self.widget_main, self.tr("Invalid Input"), str(e))
            return

        self.btn_train.setEnabled(False)
        self.btn_tune_optuna.setEnabled(False)
        self.btn_export_results.setEnabled(False)
        self.progress_bar.setVisible(True)

        data_dict = data_config.__dict__
        base_train_dict = base_train_config.__dict__

        n_trials = self.optuna_trials_spin.value()
        timeout = self.optuna_timeout_spin.value()
        use_test_metric_for_optuna = self.cb_optuna_use_test_metric.isChecked()

        self.optuna_thread = OptunaTrainThread(
            data_dict, base_train_dict,
            n_trials, timeout,
            use_test_metric_for_optuna=use_test_metric_for_optuna
        )
        self.optuna_thread.log_signal.connect(self.append_log)
        self.optuna_thread.done_signal.connect(self.optuna_tuning_finished)
        self.optuna_thread.start()

    def train_finished(self, ckpt_path: str, test_metrics: dict):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_tune_optuna.setEnabled(True)

        if ckpt_path != "ERROR":
            self.best_ckpt_path = ckpt_path
            self.last_test_metrics = test_metrics
            self.btn_export_results.setEnabled(True)
            self.append_log(f"Training finished. Best checkpoint: {ckpt_path}\n")

            cm = test_metrics.get("confusion_matrix")
            if cm:
                self.append_log(f"Confusion Matrix:\n{cm}\n")
            cr = test_metrics.get("class_report")
            if cr:
                self.append_log(f"Classification Report:\n{cr}\n")
            auc_roc = test_metrics.get("auc_roc")
            if auc_roc is not None:
                self.append_log(f"AUC-ROC: {auc_roc:.4f}\n")
        else:
            self.append_log("Training encountered an error.\n")

    def optuna_tuning_finished(self, ckpt_path: str, test_metrics: dict):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_tune_optuna.setEnabled(True)

        if ckpt_path != "ERROR":
            self.best_ckpt_path = ckpt_path
            self.last_test_metrics = test_metrics
            self.btn_export_results.setEnabled(True)
            self.append_log(f"Optuna tuning finished. Best checkpoint: {ckpt_path}\n")

            cm = test_metrics.get("confusion_matrix")
            if cm:
                self.append_log(f"Confusion Matrix:\n{cm}\n")
            cr = test_metrics.get("class_report")
            if cr:
                self.append_log(f"Classification Report:\n{cr}\n")
            auc_roc = test_metrics.get("auc_roc")
            if auc_roc is not None:
                self.append_log(f"AUC-ROC: {auc_roc:.4f}\n")
        else:
            self.append_log("Optuna tuning encountered an error.\n")

    def append_log(self, msg: str):
        if self.text_log:
            self.text_log.append(msg)
            self.text_log.ensureCursorVisible()
        logger.info(msg.strip())

    def export_all_results(self):
        if not self.last_test_metrics:
            QMessageBox.warning(self.widget_main, "No Results", "No test metrics available.")
            return

        fpath, _ = QFileDialog.getSaveFileName(
            self.widget_main, "Export Results", filter="*.txt"
        )
        if not fpath:
            return

        cm = self.last_test_metrics.get("confusion_matrix")
        cr = self.last_test_metrics.get("class_report")
        auc_roc = self.last_test_metrics.get("auc_roc")
        tb_log_dir = self.last_test_metrics.get("tb_log_dir")
        cm_fig_path = self.last_test_metrics.get("cm_fig_path")

        with open(fpath, "w", encoding="utf-8") as f:
            f.write("=== CNN Training Exported Results ===\n\n")
            f.write(f"Best Checkpoint Path: {self.best_ckpt_path}\n\n")
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(f"Classification Report:\n{cr}\n\n")
            f.write(f"AUC-ROC: {auc_roc}\n")
            if cm_fig_path and os.path.isfile(cm_fig_path):
                f.write(f"\nConfusion Matrix Figure: {cm_fig_path}\n")

        if tb_log_dir and os.path.isdir(tb_log_dir):
            zip_base = os.path.splitext(fpath)[0] + "_tb_logs"
            shutil.make_archive(zip_base, 'zip', tb_log_dir)
            self.append_log(f"TensorBoard logs exported to {zip_base}.zip\n")

        QMessageBox.information(
            self.widget_main,
            "Export Results",
            f"Results exported to {fpath}"
        )


# ===============================
# 9) UNIT & INTEGRATION TESTS
# ===============================
class TestCorruptImages(unittest.TestCase):
    def test_gather_samples_corrupt_folder(self):
        with self.assertRaises(ValueError):
            gather_samples_and_classes("/invalid/folder/path")

    def test_loading_corrupt_image(self):
        ds = AlbumentationsDataset([("/fake/path/to/corrupt.jpg", 0)])
        with self.assertRaises(OSError):
            _ = ds[0]


class TestBasicIntegration(unittest.TestCase):
    def test_create_configs(self):
        data_config = DataConfig("some/path", 0.1, 0.1, 8, True)
        self.assertEqual(data_config.val_split, 0.1)
        train_config = TrainConfig(
            max_epochs=1,
            architecture="resnet18",
            lr=1e-4,
            momentum=0.9,
            weight_decay=1e-4,
            use_weighted_loss=False,
            optimizer_name="adam",
            scheduler_name="none",
            scheduler_params={},
            do_early_stopping=False,
            early_stopping_monitor="val_loss",
            early_stopping_patience=5,
            early_stopping_min_delta=0.0,
            early_stopping_mode="min",
            brightness_contrast=False,
            hue_saturation=False,
            gaussian_noise=False,
            use_rotation=False,
            use_flip=False,
            flip_mode="horizontal",
            flip_prob=0.5,
            use_crop=False,
            use_elastic=False,
            normalize_pixel_intensity=False,
            use_grid_distortion=False,
            use_optical_distortion=False,
            use_mixup=False,
            use_cutmix=False,
            mix_alpha=1.0,
            dropout_rate=0.0,
            label_smoothing=0.0,
            freeze_backbone=False,
            loss_function="cross_entropy",
            gradient_clip_val=0.0,
            use_lr_finder=False,
            accept_lr_suggestion=False,
            use_tensorboard=False,
            use_mixed_precision=False,
            warmup_epochs=0,
            use_inception_299=False,
            enable_gradient_checkpointing=False,
            enable_grad_accum=False,
            accumulate_grad_batches=1,
            check_val_every_n_epoch=1,
            freeze_config={},
            num_workers=0,
            val_center_crop=False,
            random_crop_prob=1.0,
            random_crop_scale_min=0.8,
            random_crop_scale_max=1.0
        )
        self.assertEqual(train_config.architecture, "resnet18")


def run_tests():
    unittest.main(argv=[''], exit=False)


if __name__ == "__main__":
    run_tests()
