import os
import glob
import time
import random
import gc
import csv
from typing import Tuple, Dict, Any, List, Optional
import unittest
import numpy as np
import cv2
import torch

torch.set_float32_matmul_precision('medium')
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import (
    SequentialLR, LambdaLR, StepLR, ReduceLROnPlateau,
    CosineAnnealingLR, CyclicLR
)

# Additional imports for plotting, metrics, and UI
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, log_loss, average_precision_score,
    precision_recall_curve, roc_curve, auc, roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize

from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QProgressBar, QMessageBox, QGroupBox, QScrollArea, QFormLayout, QGridLayout,
    QDialog, QApplication
)

import multiprocessing
import argparse
import sys
from PIL import Image
import torchvision.transforms as T

# -------------------------------------------------------------------------
# 1) TorchRandAugment & TorchAutoAugment
# -------------------------------------------------------------------------
class TorchRandAugment:
    def __init__(self, num_ops: int = 2, magnitude: int = 9):
        self.randaug = T.RandAugment(num_ops=num_ops, magnitude=magnitude)
        self.available_keys = ("image",)

    def __call__(self, image, **kwargs):
        pil_image = Image.fromarray(image)
        pil_image = self.randaug(pil_image)
        return {"image": np.array(pil_image)}

class TorchAutoAugment:
    def __init__(self, policy: str = "IMAGENET"):
        if policy.upper() == "IMAGENET":
            self.autoaug = T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET)
        elif policy.upper() == "CIFAR10":
            self.autoaug = T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10)
        else:
            self.autoaug = T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET)
        self.available_keys = ("image",)

    def __call__(self, image, **kwargs):
        pil_image = Image.fromarray(image)
        pil_image = self.autoaug(pil_image)
        return {"image": np.array(pil_image)}

# -------------------------------------------------------------------------
# 2) ClickableLabel
# -------------------------------------------------------------------------
class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

# -------------------------------------------------------------------------
# 3) BasePlugin stub if missing
# -------------------------------------------------------------------------
try:
    from base_plugin import BasePlugin
except ImportError:
    class BasePlugin:
        def __init__(self):
            pass

torch.backends.cudnn.benchmark = True

# -------------------------------------------------------------------------
# 4) CONFIG CLASSES
# -------------------------------------------------------------------------
class DataConfig:
    def __init__(
        self,
        root_dir: str,
        val_split: float,
        test_split: float,
        batch_size: int,
        allow_grayscale: bool,
        preload_images: bool = True,
        use_memmap_images: bool = False
    ):
        self.root_dir = root_dir
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.allow_grayscale = allow_grayscale
        self.preload_images = preload_images
        self.use_memmap_images = use_memmap_images

# NEW CODE: Add a 'non_blocking' flag to TrainConfig for GPU transfers
class TrainConfig:
    def __init__(
        self,
        max_epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        use_weighted_loss: bool,
        optimizer_name: str,
        scheduler_name: str,
        scheduler_params: Dict[str, Any],
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
        normalize_pixel_intensity: bool,
        use_mixup: bool,
        use_cutmix: bool,
        mix_alpha: float,
        dropout_rate: float,
        label_smoothing: float,
        loss_function: str,
        gradient_clip_val: float,
        use_mixed_precision: bool,
        enable_grad_accum: bool,
        accumulate_grad_batches: int,
        check_val_every_n_epoch: int,
        num_workers: int,
        val_center_crop: bool,
        random_crop_prob: float,
        random_crop_scale_min: float,
        random_crop_scale_max: float,
        pretrained_weights: bool,
        run_gc: bool,
        enable_tta: bool,
        profile_memory: bool,
        persistent_workers: bool,
        crop_size: int,
        model_variant: str = "convnext_base",
        drop_path_rate: float = 0.0,
        use_randaug: bool = False,
        use_autoaug: bool = False,
        randaug_num_ops: int = 2,
        randaug_magnitude: int = 9,
        autoaug_policy: str = "IMAGENET",

        # NEW CODE: Non-blocking GPU transfers
        non_blocking: bool = False
    ):
        self.max_epochs = max_epochs
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
        self.normalize_pixel_intensity = normalize_pixel_intensity
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mix_alpha = mix_alpha
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.loss_function = loss_function
        self.gradient_clip_val = gradient_clip_val
        self.use_mixed_precision = use_mixed_precision
        self.enable_grad_accum = enable_grad_accum
        self.accumulate_grad_batches = accumulate_grad_batches
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.num_workers = num_workers
        self.val_center_crop = val_center_crop
        self.random_crop_prob = random_crop_prob
        self.random_crop_scale_min = random_crop_scale_min
        self.random_crop_scale_max = random_crop_scale_max
        self.pretrained_weights = pretrained_weights
        self.run_gc = run_gc
        self.enable_tta = enable_tta
        self.profile_memory = profile_memory
        self.persistent_workers = persistent_workers
        self.crop_size = crop_size
        self.model_variant = model_variant
        self.drop_path_rate = drop_path_rate
        self.use_randaug = use_randaug
        self.use_autoaug = use_autoaug
        self.randaug_num_ops = randaug_num_ops
        self.randaug_magnitude = randaug_magnitude
        self.autoaug_policy = autoaug_policy

        # NEW CODE: store the non_blocking param
        self.non_blocking = non_blocking

# -------------------------------------------------------------------------
# 5) DATASET & SPLITS
# -------------------------------------------------------------------------
def gather_samples_and_classes(root_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    if not os.path.isdir(root_dir):
        raise ValueError(f"Invalid root_dir: {root_dir}")
    classes = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    if not classes:
        raise ValueError(f"No class subfolders found in {root_dir}.")

    valid_extensions = (".tiff", ".tif", ".png", ".jpg", ".jpeg")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    samples = []

    for cls_name in classes:
        sub_dir = os.path.join(root_dir, cls_name)
        for ext in valid_extensions:
            pattern = os.path.join(sub_dir, "**", f"*{ext}")
            for fpath in glob.glob(pattern, recursive=True):
                if os.path.isfile(fpath):
                    try:
                        with open(fpath, "rb") as fp:
                            fp.read(20)
                    except OSError:
                        continue
                    samples.append((fpath, class_to_idx[cls_name]))
    return samples, classes

# -------------------------------------------------------------------------
# 6) AlbumentationsDatasetPreloaded
# -------------------------------------------------------------------------
class AlbumentationsDatasetPreloaded(torch.utils.data.Dataset):
    def __init__(self, samples: List[Tuple[str, int]],
                 transform: Optional[A.Compose] = None,
                 classes: Optional[List[str]] = None,
                 allow_grayscale: bool = False):
        self.transform = transform
        self.classes = classes if classes is not None else []
        self.allow_grayscale = allow_grayscale
        self.data = []

        for fpath, label in samples:
            image_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise OSError(f"Failed to load or corrupt image: {fpath}")
            if len(image_bgr.shape) == 2:
                if not self.allow_grayscale:
                    raise ValueError(
                        f"Single-channel image but allow_grayscale=False: {fpath}"
                    )
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.data.append((image_rgb, label))

        print(f"[INFO] Preloaded {len(self.data)} images into memory.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        image_rgb, label = self.data[idx]
        if self.transform:
            augmented = self.transform(image=image_rgb)
            image_tensor = augmented["image"].float()
        else:
            image_tensor = torch.from_numpy(
                np.transpose(image_rgb, (2, 0, 1))
            ).float() / 255.0
        return image_tensor, label

# -------------------------------------------------------------------------
# 7) AlbumentationsDatasetLazy
# -------------------------------------------------------------------------
class AlbumentationsDatasetLazy(torch.utils.data.Dataset):
    def __init__(self, samples: List[Tuple[str, int]],
                 transform: Optional[A.Compose] = None,
                 classes: Optional[List[str]] = None,
                 allow_grayscale: bool = False):
        self.transform = transform
        self.classes = classes if classes is not None else []
        self.allow_grayscale = allow_grayscale
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        image_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"Failed to load or corrupt image: {fpath}")
        if len(image_bgr.shape) == 2:
            if not self.allow_grayscale:
                raise ValueError(
                    f"Single-channel image but allow_grayscale=False: {fpath}"
                )
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image_bgr)
            image_tensor = augmented["image"].float()
        else:
            image_tensor = torch.from_numpy(
                np.transpose(image_bgr, (2, 0, 1))
            ).float() / 255.0
        return image_tensor, label

# -------------------------------------------------------------------------
# 8) AlbumentationsDatasetMemmap
# -------------------------------------------------------------------------
class AlbumentationsDatasetMemmap(torch.utils.data.Dataset):
    def __init__(self, samples: List[Tuple[str, int]],
                 transform: Optional[A.Compose] = None,
                 classes: Optional[List[str]] = None,
                 allow_grayscale: bool = False):
        self.transform = transform
        self.classes = classes if classes is not None else []
        self.allow_grayscale = allow_grayscale
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        mm = np.memmap(fpath, dtype=np.uint8, mode='r')
        data_array = np.frombuffer(mm, dtype=np.uint8)
        image_bgr = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"Failed to load or corrupt image via memmap: {fpath}")
        if len(image_bgr.shape) == 2:
            if not self.allow_grayscale:
                raise ValueError(
                    f"Single-channel image but allow_grayscale=False: {fpath}"
                )
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image_bgr)
            image_tensor = augmented["image"].float()
        else:
            image_tensor = torch.from_numpy(
                np.transpose(image_bgr, (2, 0, 1))
            ).float() / 255.0

        return image_tensor, label

# -------------------------------------------------------------------------
# 9) CUSTOM LOSS (FocalLoss)
# -------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
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

# -------------------------------------------------------------------------
# 10) MODEL CLASS
# -------------------------------------------------------------------------
class MaidClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        use_weighted_loss: bool = False,
        optimizer_name: str = "adam",
        scheduler_name: str = "none",
        scheduler_params: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.0,
        label_smoothing: float = 0.0,
        loss_function: str = "cross_entropy",
        pretrained: bool = True,
        enable_tta: bool = False,
        use_mixup: bool = False,
        use_cutmix: bool = False,
        mix_alpha: float = 1.0,
        model_variant: str = "convnext_base",
        drop_path_rate: float = 0.0,

        # NEW CODE: We'll store the non_blocking in the constructor too
        non_blocking: bool = False
    ):
        super().__init__()
        if scheduler_params is None:
            scheduler_params = {}
        self.save_hyperparameters(ignore=["scheduler_params"])

        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_weighted_loss = use_weighted_loss
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.loss_function = loss_function
        self.enable_tta = enable_tta
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mix_alpha = mix_alpha

        # NEW CODE: store it as an attribute for easy usage
        self.non_blocking = non_blocking

        self.class_names: Optional[List[str]] = None

        # NOTE: Removed the old code that appended ".fcmae_ft_in22k_in1k" for convnextv2
        try:
            import timm
            local_variant = model_variant
            self.backbone = timm.create_model(
                local_variant,
                pretrained=pretrained,
                num_classes=0,
                drop_path_rate=drop_path_rate
            )
        except ImportError:
            raise ImportError("timm is required for ConvNeXt. Please install timm.")

        in_feats = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_feats, num_classes)
        )

        # Choose the appropriate loss function
        if self.loss_function == "focal":
            self.loss_fn = FocalLoss()
        elif self.loss_function == "bce_single_logit" and self.num_classes == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = None

    # NEW CODE: Override transfer_batch_to_device to enable non-blocking
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        x, y = batch
        x = x.to(device, non_blocking=self.non_blocking)
        y = y.to(device, non_blocking=self.non_blocking)
        return x, y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        if self.loss_function == "bce_single_logit":
            y = y.unsqueeze(1).float()

        # Mixup or Cutmix if enabled
        if self.use_mixup or self.use_cutmix:
            if self.use_mixup and self.use_cutmix:
                if random.random() < 0.5:
                    x, y_a, y_b, lam = mixup_data(x, y, alpha=self.mix_alpha)
                else:
                    x, y_a, y_b, lam = cutmix_data(x, y, alpha=self.mix_alpha)
            elif self.use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=self.mix_alpha)
            else:
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=self.mix_alpha)

            logits = self(x)
            if self.loss_fn is None:
                loss_a = F.cross_entropy(logits, y_a, label_smoothing=self.label_smoothing)
                loss_b = F.cross_entropy(logits, y_b, label_smoothing=self.label_smoothing)
            else:
                if self.loss_function == "bce_single_logit":
                    y_a = y_a.unsqueeze(1).float()
                    y_b = y_b.unsqueeze(1).float()
                loss_a = self.loss_fn(logits, y_a)
                loss_b = self.loss_fn(logits, y_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            logits = self(x)
            if self.loss_fn is None:
                loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
            else:
                loss = self.loss_fn(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.loss_function == "bce_single_logit":
            y = y.unsqueeze(1).float()
        logits = self(x)
        if self.loss_fn is None:
            loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        else:
            loss = self.loss_fn(logits, y)

        if self.loss_function == "bce_single_logit":
            preds = (torch.sigmoid(logits) > 0.5).long().view(-1)
            acc = (preds == y.view(-1)).float().mean()
        else:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.loss_function == "bce_single_logit":
            y = y.unsqueeze(1).float()
        logits = self(x)
        if self.loss_fn is None:
            loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        else:
            loss = self.loss_fn(logits, y)

        if self.loss_function == "bce_single_logit":
            preds = (torch.sigmoid(logits) > 0.5).long().view(-1)
            acc = (preds == y.view(-1)).float().mean()
        else:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        opt_name = self.optimizer_name.lower()
        scheduler_params = self.scheduler_params.copy()
        warmup_epochs = int(scheduler_params.pop("warmup_epochs", 0))
        monitor_metric = scheduler_params.pop("monitor", "val_loss")

        # Choose optimizer
        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}.")

        sch_name = self.scheduler_name.lower()
        main_scheduler = None
        if sch_name == "steplr":
            step_size = int(scheduler_params.get("step_size", 10))
            gamma = scheduler_params.get("gamma", 0.1)
            main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif sch_name == "reducelronplateau":
            factor = scheduler_params.get("factor", 0.1)
            patience = int(scheduler_params.get("patience", 5))
            main_scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
        elif sch_name == "cosineannealing":
            T_max = int(scheduler_params.get("t_max", 10))
            main_scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        elif sch_name == "cycliclr":
            base_lr = scheduler_params.get("base_lr", 1e-4)
            max_lr = scheduler_params.get("max_lr", 1e-2)
            step_size_up = int(scheduler_params.get("step_size_up", 2000))
            main_scheduler = CyclicLR(
                optimizer, base_lr, max_lr,
                step_size_up=step_size_up, mode="triangular2"
            )
        elif sch_name == "cosinedecay":
            if "decay_steps" in scheduler_params:
                decay_steps = int(scheduler_params.get("decay_steps", 10))
            else:
                decay_steps = 10
            main_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 0.5 * (1 + np.cos(np.pi * epoch / decay_steps))
            )
        elif sch_name == "cosineannealingwarmrestarts":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            T_0 = scheduler_params.get("T_0", 10)
            T_mult = scheduler_params.get("T_mult", 1)
            eta_min = scheduler_params.get("eta_min", 1e-6)
            main_scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
        # 'none' => no LR scheduler

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
                        "lr_scheduler": {"scheduler": scheduler, "monitor": monitor_metric}
                    }
                return [optimizer], [scheduler]
            else:
                return [optimizer], [warmup_scheduler]
        else:
            if main_scheduler is not None:
                if isinstance(main_scheduler, ReduceLROnPlateau):
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {"scheduler": main_scheduler, "monitor": monitor_metric}
                    }
                return [optimizer], [main_scheduler]
            return optimizer

# -------------------------------------------------------------------------
# 11) MIXUP & CUTMIX
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 12) Utility to collect predictions for confusion matrix
# -------------------------------------------------------------------------
def gather_predictions(loader, model) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            # NEW CODE: use the same non_blocking approach here
            xb = xb.to(model.device, non_blocking=model.non_blocking)
            yb = yb.to(model.device, non_blocking=model.non_blocking)

            logits = model(xb)
            if model.loss_function == "bce_single_logit":
                preds = (torch.sigmoid(logits) > 0.5).long().view(-1)
            else:
                preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    return np.array(all_preds), np.array(all_targets)

# -------------------------------------------------------------------------
# 13) TRAINING FUNCTION
# -------------------------------------------------------------------------
def run_training_once(data_params: DataConfig, train_params: TrainConfig) -> Tuple[str, Dict[str, Any], Optional[float]]:
    seed_everything(42, workers=True)
    print("[INFO] Gathering samples...")
    samples, class_names = gather_samples_and_classes(data_params.root_dir)
    n_total = len(samples)
    if n_total < 2:
        raise ValueError("Dataset has insufficient images.")
    print(f"[INFO] Found {n_total} images among {len(class_names)} classes: {class_names}")
    targets = [lbl for _, lbl in samples]

    # Splitting into train/val/test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=data_params.test_split, random_state=42)
    trainval_index, test_index = next(sss_test.split(np.arange(n_total), targets))
    trainval_samples = [samples[i] for i in trainval_index]
    trainval_targets = [targets[i] for i in trainval_index]

    if data_params.val_split > 0:
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=data_params.val_split / (1.0 - data_params.test_split),
            random_state=42
        )
        train_index, val_index = next(sss_val.split(trainval_samples, trainval_targets))
    else:
        train_index = range(len(trainval_samples))
        val_index = []

    train_samples = [trainval_samples[i] for i in train_index]
    val_samples = [trainval_samples[i] for i in val_index]
    test_samples = [samples[i] for i in test_index]

    print(f"[INFO] Splits => Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Decide final layer size
    if train_params.loss_function == "bce_single_logit":
        model_num_classes = 1
    else:
        model_num_classes = len(class_names)

    # Build the model
    model = MaidClassifier(
        num_classes=model_num_classes,
        lr=train_params.lr,
        momentum=train_params.momentum,
        weight_decay=train_params.weight_decay,
        use_weighted_loss=train_params.use_weighted_loss,
        optimizer_name=train_params.optimizer_name,
        scheduler_name=train_params.scheduler_name,
        scheduler_params=train_params.scheduler_params,
        dropout_rate=train_params.dropout_rate,
        label_smoothing=train_params.label_smoothing,
        loss_function=train_params.loss_function,
        pretrained=train_params.pretrained_weights,
        enable_tta=train_params.enable_tta,
        use_mixup=train_params.use_mixup,
        use_cutmix=train_params.use_cutmix,
        mix_alpha=train_params.mix_alpha,
        model_variant=train_params.model_variant,
        drop_path_rate=train_params.drop_path_rate,

        # NEW CODE: pass the non_blocking option
        non_blocking=train_params.non_blocking
    )

    model.class_names = class_names

    wandb_logger = WandbLogger(project="MAID", log_model=True)
    if hasattr(wandb_logger, "experiment"):
        wandb_logger.experiment.watch(model, log="all")

    ckpt_callback = ModelCheckpoint(
        monitor=train_params.early_stopping_monitor,
        save_top_k=1,
        mode=train_params.early_stopping_mode,
        filename="best-checkpoint"
    )
    callbacks = [ckpt_callback]

    if train_params.do_early_stopping:
        early_stop = EarlyStopping(
            monitor=train_params.early_stopping_monitor,
            patience=train_params.early_stopping_patience,
            min_delta=train_params.early_stopping_min_delta,
            mode=train_params.early_stopping_mode
        )
        callbacks.append(early_stop)

    progress_cb = ProgressCallback(
        total_epochs=train_params.max_epochs,
        run_gc=train_params.run_gc,
        profile_memory=train_params.profile_memory
    )
    callbacks.append(progress_cb)
    callbacks.append(GradNormCallback())

    trainer_device = "gpu" if torch.cuda.is_available() else "cpu"
    devices_to_use = torch.cuda.device_count() if torch.cuda.is_available() else 1

    trainer = Trainer(
        max_epochs=train_params.max_epochs,
        accelerator=trainer_device,
        devices=devices_to_use,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=train_params.gradient_clip_val,
        precision=16 if (train_params.use_mixed_precision and torch.cuda.is_available()) else 32,
        accumulate_grad_batches=train_params.accumulate_grad_batches if train_params.enable_grad_accum else 1,
        check_val_every_n_epoch=train_params.check_val_every_n_epoch,
        enable_progress_bar=True
    )

    final_crop_dim = int(train_params.crop_size)
    bigger_resize = int(final_crop_dim * 1.14)

    aug_list = []
    if train_params.use_randaug:
        aug_list.insert(0, TorchRandAugment(
            num_ops=train_params.randaug_num_ops,
            magnitude=train_params.randaug_magnitude
        ))
    elif train_params.use_autoaug:
        aug_list.insert(0, TorchAutoAugment(
            policy=train_params.autoaug_policy
        ))

    if train_params.use_crop:
        aug_list.append(A.RandomResizedCrop(
            size=(final_crop_dim, final_crop_dim),
            scale=(train_params.random_crop_scale_min, train_params.random_crop_scale_max),
            ratio=(0.75, 1.3333),
            interpolation=cv2.INTER_LINEAR,
            p=train_params.random_crop_prob
        ))
    else:
        aug_list.append(A.Resize(
            height=final_crop_dim, width=final_crop_dim,
            interpolation=cv2.INTER_LINEAR, p=1.0
        ))

    if train_params.brightness_contrast:
        aug_list.append(A.RandomBrightnessContrast(p=0.5))
    if train_params.hue_saturation:
        aug_list.append(A.HueSaturationValue(p=0.5))
    if train_params.gaussian_noise:
        aug_list.append(A.GaussNoise(p=0.5))
    if train_params.use_rotation:
        aug_list.append(A.Rotate(limit=30, p=0.5))
    if train_params.use_flip:
        if train_params.flip_mode == "horizontal":
            aug_list.append(A.HorizontalFlip(p=train_params.flip_prob))
        elif train_params.flip_mode == "vertical":
            aug_list.append(A.VerticalFlip(p=train_params.flip_prob))
        else:
            aug_list.append(
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0)
                ], p=train_params.flip_prob)
            )

    if train_params.normalize_pixel_intensity:
        aug_list.append(A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            p=1.0
        ))

    aug_list.append(ToTensorV2())
    train_transform = A.Compose(aug_list)

    val_augs = [
        A.Resize(height=bigger_resize, width=bigger_resize,
                 interpolation=cv2.INTER_LINEAR, p=1.0)
    ]
    if train_params.val_center_crop:
        val_augs.append(A.CenterCrop(
            height=final_crop_dim, width=final_crop_dim
        ))
    if train_params.normalize_pixel_intensity:
        val_augs.append(A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ))
    val_augs.append(ToTensorV2())
    val_transform = A.Compose(val_augs)

    if data_params.use_memmap_images:
        TrainDatasetClass = AlbumentationsDatasetMemmap
        ValTestDatasetClass = AlbumentationsDatasetMemmap
    elif data_params.preload_images:
        TrainDatasetClass = AlbumentationsDatasetPreloaded
        ValTestDatasetClass = AlbumentationsDatasetPreloaded
    else:
        TrainDatasetClass = AlbumentationsDatasetLazy
        ValTestDatasetClass = AlbumentationsDatasetLazy

    train_ds = TrainDatasetClass(
        train_samples, transform=train_transform,
        classes=class_names, allow_grayscale=data_params.allow_grayscale
    )
    val_ds = ValTestDatasetClass(
        val_samples, transform=val_transform,
        classes=class_names, allow_grayscale=data_params.allow_grayscale
    )
    test_ds = ValTestDatasetClass(
        test_samples, transform=val_transform,
        classes=class_names, allow_grayscale=data_params.allow_grayscale
    )

    persistent_workers = train_params.persistent_workers and (train_params.num_workers > 0)

    # pin_memory = True => helps with async data transfer
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=data_params.batch_size,
        shuffle=True,
        num_workers=train_params.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=data_params.batch_size,
        shuffle=False,
        num_workers=train_params.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=data_params.batch_size,
        shuffle=False,
        num_workers=train_params.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    trainer.fit(model, train_loader, val_loader)
    best_checkpoint_path = ckpt_callback.best_model_path
    print(f"[INFO] Validating & testing best checkpoint from: {best_checkpoint_path}")
    best_model = MaidClassifier.load_from_checkpoint(best_checkpoint_path)
    best_model.class_names = model.class_names
    best_model.hparams.enable_tta = model.hparams.enable_tta
    # also apply the non_blocking param to best_model
    best_model.non_blocking = model.non_blocking

    val_results = trainer.validate(best_model, val_loader, verbose=False)
    final_val_loss = float(val_results[0]["val_loss"]) if val_results and "val_loss" in val_results[0] else None
    trainer.test(best_model, test_loader, verbose=False)

    # Gather predictions for confusion matrix logging
    train_preds, train_targets = gather_predictions(train_loader, best_model)
    val_preds, val_targets = gather_predictions(val_loader, best_model)
    test_preds, test_targets = gather_predictions(test_loader, best_model)

    if best_model.loss_function == "bce_single_logit":
        prec_val = precision_score(test_targets, test_preds, average='weighted', zero_division=0)
        recall_val = recall_score(test_targets, test_preds, average='weighted', zero_division=0)
        f1_val = f1_score(test_targets, test_preds, average='weighted', zero_division=0)
        ll_val = None
    else:
        prec_val = precision_score(test_targets, test_preds, average='weighted', zero_division=0)
        recall_val = recall_score(test_targets, test_preds, average='weighted', zero_division=0)
        f1_val = f1_score(test_targets, test_preds, average='weighted', zero_division=0)
        ll_val = None

    cr = classification_report(test_targets, test_preds, target_names=class_names, zero_division=0)

    wandb_logger.experiment.log({
        "final_val_loss": final_val_loss,
        "precision": prec_val,
        "recall": recall_val,
        "f1_score": f1_val,
        "classification_report": cr,
    })

    # Quick static confusion matrices
    def log_confusion_matrix_image(title: str, preds: np.ndarray, targets: np.ndarray):
        cm = confusion_matrix(targets, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        wandb_logger.experiment.log({title: wandb.Image(fig)})
        plt.close(fig)

    log_confusion_matrix_image("train_confusion_matrix", train_preds, train_targets)
    log_confusion_matrix_image("val_confusion_matrix", val_preds, val_targets)
    log_confusion_matrix_image("test_confusion_matrix", test_preds, test_targets)

    print(f"[INFO] Finished training. Best checkpoint: {best_checkpoint_path}")

    test_metrics_dict = {
        "precision": prec_val,
        "recall": recall_val,
        "f1_score": f1_val,
        "classification_report": cr
    }
    return best_checkpoint_path, test_metrics_dict, final_val_loss

# -------------------------------------------------------------------------
# 14) TRAINING WORKER FUNCTION
# -------------------------------------------------------------------------
def _try_run_training(data_dict, train_dict):
    data_config = DataConfig(**data_dict)
    train_config = TrainConfig(**train_dict)
    best_ckpt_path, test_metrics, val_loss = run_training_once(data_config, train_config)
    return best_ckpt_path, test_metrics

def _auto_adjust_batch_size(data_dict, train_dict, original_bs) -> (Optional[str], Optional[dict]):
    low, high = 1, original_bs
    feasible_bs = 1
    best_ckpt_path = None
    best_test_metrics = None
    while low <= high:
        mid = (low + high) // 2
        data_dict["batch_size"] = mid
        print(f"[INFO] Trying batch_size={mid} in fallback.")
        try:
            bcp, tmetrics = _try_run_training(data_dict, train_dict)
            feasible_bs = mid
            best_ckpt_path = bcp
            best_test_metrics = tmetrics
            low = mid + 1
        except RuntimeError as re:
            if "CUDA out of memory" in str(re):
                high = mid - 1
            else:
                print(f"[ERROR] Non-OOM error: {re}")
                return None, None
    if feasible_bs == 1 and low == 1 and best_ckpt_path is None:
        print("[ERROR] Even batch_size=1 failed. Cannot train.")
        return None, None
    return best_ckpt_path, best_test_metrics

def train_worker(data_params, train_params, return_dict):
    try:
        original_bs = data_params["batch_size"]
        try:
            bcp, tmetrics = _try_run_training(data_params, train_params)
            return_dict["status"] = "OK"
            return_dict["ckpt_path"] = bcp
            return_dict["test_metrics"] = tmetrics
        except RuntimeError as re:
            if "CUDA out of memory" in str(re):
                print("[WARN] GPU OOM encountered. Attempting batch-size fallback.")
                bcp2, tmetrics2 = _auto_adjust_batch_size(data_params, train_params, original_bs)
                if bcp2 is None:
                    return_dict["status"] = "ERROR"
                    return_dict["ckpt_path"] = ""
                    return_dict["test_metrics"] = {}
                else:
                    return_dict["status"] = "OK"
                    return_dict["ckpt_path"] = bcp2
                    return_dict["test_metrics"] = tmetrics2
            else:
                print(f"[ERROR] Training crashed: {re}")
                return_dict["status"] = "ERROR"
                return_dict["ckpt_path"] = ""
                return_dict["test_metrics"] = {}
    except Exception as e:
        print(f"[ERROR] Training crashed: {e}")
        return_dict["status"] = "ERROR"
        return_dict["ckpt_path"] = ""
        return_dict["test_metrics"] = {}

# -------------------------------------------------------------------------
# 15) PROGRESS CALLBACK
# -------------------------------------------------------------------------
class ProgressCallback(Callback):
    def __init__(self, total_epochs: int, run_gc: bool = False, profile_memory: bool = False):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        self.run_gc = run_gc
        self.profile_memory = profile_memory

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()
        print("[INFO] Training started...")

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        elapsed = time.time() - self.start_time
        epochs_left = self.total_epochs - current_epoch
        time_per_epoch = elapsed / current_epoch if current_epoch > 0 else 0.0
        eta = time_per_epoch * epochs_left
        train_loss = trainer.callback_metrics.get("train_loss", "N/A")
        val_loss = trainer.callback_metrics.get("val_loss", "N/A")
        val_acc = trainer.callback_metrics.get("val_acc", "N/A")

        grad_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        batch_size = 1
        if hasattr(trainer, "train_dataloader") and trainer.train_dataloader is not None:
            try:
                batch_size = trainer.train_dataloader.batch_size
            except AttributeError:
                batch_size = 1

        throughput = batch_size / time_per_epoch if time_per_epoch > 0 else 0

        print(
            f"[EPOCH {current_epoch}/{self.total_epochs}] "
            f"ETA: {eta:.1f}s | train_loss={train_loss}, val_loss={val_loss}, val_acc={val_acc}"
        )
        print(
            f"[METRICS] LR: {trainer.optimizers[0].param_groups[0]['lr']:.6f}, "
            f"GradNorm: {grad_norm:.4f}, Throughput: {throughput:.2f} img/s"
        )

        if hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({
                "epoch": current_epoch,
                "current_lr": trainer.optimizers[0].param_groups[0]['lr'],
                "grad_norm": grad_norm,
                "epoch_time": time_per_epoch,
                "throughput": throughput
            })

        if self.run_gc:
            gc.collect()
            torch.cuda.empty_cache()

        if self.profile_memory and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"[MEM] GPU allocated: {allocated:.2f} MB")

# -------------------------------------------------------------------------
# 16) GRADIENT NORM CALLBACK
# -------------------------------------------------------------------------
class GradNormCallback(Callback):
    def on_after_backward(self, trainer, pl_module):
        grad_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        print(f"[DEBUG] Grad norm after backward: {grad_norm:.4f}")
        if hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({"grad_norm": grad_norm})

# -------------------------------------------------------------------------
# 17) GRADCAM HELPER
# -------------------------------------------------------------------------
def get_module_by_name(model, module_name):
    modules = dict(model.named_modules())
    return modules.get(module_name, None)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class, gradcam_type="Gradcam"):
        self.model.zero_grad()
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        if gradcam_type.lower() == "gradcam":
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        elif gradcam_type.lower() == "gradcam++":
            grads = self.gradients
            activations = self.activations
            grads_squared = grads ** 2
            grads_cubed = grads ** 3
            sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
            alpha = grads_squared / (2 * grads_squared + sum_activations * grads_cubed + 1e-8)
            weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
        elif gradcam_type.lower() == "xgradcam":
            weights = torch.mean(self.gradients * self.activations, dim=(2, 3), keepdim=True)
        else:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam

def overlay_heatmap_on_image(image, heatmap, colormap_name="Jet", alpha=0.5):
    colormap_dict = {
        "Jet": cv2.COLORMAP_JET, "hot": cv2.COLORMAP_HOT,
        "Bone": cv2.COLORMAP_BONE, "HSV": cv2.COLORMAP_HSV
    }
    colormap = colormap_dict.get(colormap_name, cv2.COLORMAP_JET)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    if image.shape[:2] != heatmap_color.shape[:2]:
        image = cv2.resize(image, (heatmap_color.shape[1], heatmap_color.shape[0]))
    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        1 - alpha, heatmap_color, alpha, 0
    )
    return overlay

# -------------------------------------------------------------------------
# 18) IMAGE VIEWER
# -------------------------------------------------------------------------
class ImageViewerDialog(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Viewer")
        self.original_pixmap = pixmap
        self.label = QLabel()
        self.label.setPixmap(pixmap)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)
        self.resize(pixmap.size())
        self.scale_factor = 1.0

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.scaleImage(1.25)
        else:
            self.scaleImage(0.8)

    def scaleImage(self, factor):
        self.scale_factor *= factor
        new_size = self.original_pixmap.size() * self.scale_factor
        scaled_pixmap = self.original_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)

# -------------------------------------------------------------------------
# 19) Bulk Inference
# -------------------------------------------------------------------------
class BulkInferenceWorker(QThread):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(self, checkpoint_path, bulk_folder, crop_size, topk, gt_csv):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.bulk_folder = bulk_folder
        self.crop_size = crop_size
        self.topk = topk
        self.gt_csv = gt_csv

    def run(self):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = MaidClassifier.load_from_checkpoint(self.checkpoint_path)
            model.eval()
            model.to(device)

            inference_transform = A.Compose([
                A.Resize(height=self.crop_size, width=self.crop_size, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225), p=1.0),
                ToTensorV2()
            ])

            gt_mapping = {}
            if self.gt_csv and os.path.isfile(self.gt_csv):
                with open(self.gt_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        gt_mapping[os.path.basename(row["image_path"])] = row["true_label"]

            valid_extensions = (".tiff", ".tif", ".png", ".jpg", ".jpeg")
            image_paths = [
                p for p in glob.glob(os.path.join(self.bulk_folder, "**", "*.*"), recursive=True)
                if p.lower().endswith(valid_extensions)
            ]
            total = len(image_paths)

            results = []
            for i, image_path in enumerate(image_paths):
                try:
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    augmented = inference_transform(image=image_rgb)
                    image_tensor = augmented["image"].unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(image_tensor)
                        probabilities = F.softmax(logits, dim=1)
                        topk_probs, topk_indices = torch.topk(probabilities, k=self.topk, dim=1)
                        topk_probs = topk_probs.cpu().numpy().flatten()
                        topk_indices = topk_indices.cpu().numpy().flatten()

                    if model.class_names:
                        topk_classes = [model.class_names[ix] for ix in topk_indices]
                    else:
                        topk_classes = [str(ix) for ix in topk_indices]

                    result = {
                        "image_path": image_path,
                        "topk_classes": topk_classes,
                        "topk_confidences": topk_probs.tolist()
                    }
                    base_name = os.path.basename(image_path)
                    if base_name in gt_mapping:
                        result["true_label"] = gt_mapping[base_name]
                        result["correct"] = int(topk_classes[0] == gt_mapping[base_name])

                    results.append(result)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

                self.progress.emit(i + 1, total)

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

# -------------------------------------------------------------------------
# 20) MAIN PLUGIN CLASS (GUI)
# -------------------------------------------------------------------------
class Plugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.plugin_name = "ConvNeXt Training"
        self.widget_main: Optional[QWidget] = None
        self.progress_bar: Optional[QProgressBar] = None

        # Data/Split Inputs
        self.train_data_dir_edit: Optional[QLineEdit] = None
        self.val_split_spin: Optional[QDoubleSpinBox] = None
        self.test_split_spin: Optional[QDoubleSpinBox] = None
        self.cb_allow_grayscale: Optional[QCheckBox] = None

        # Preload & Memmap
        self.cb_preload_images: Optional[QCheckBox] = None
        self.cb_memmap_images: Optional[QCheckBox] = None

        # Architecture & Model Variant
        self.model_variant_combo: Optional[QComboBox] = None
        self.weighted_loss_cb: Optional[QCheckBox] = None
        self.cb_normalize_pixel_intensity: Optional[QCheckBox] = None

        # Basic Augmentations
        self.cb_bright_contrast: Optional[QCheckBox] = None
        self.cb_hue_sat: Optional[QCheckBox] = None
        self.cb_gauss_noise: Optional[QCheckBox] = None
        self.cb_rotation: Optional[QCheckBox] = None
        self.cb_flip: Optional[QCheckBox] = None
        self.flip_mode_combo: Optional[QComboBox] = None
        self.flip_prob_spin: Optional[QDoubleSpinBox] = None
        self.cb_crop: Optional[QCheckBox] = None

        # Advanced Augmentations
        self.cb_mixup: Optional[QCheckBox] = None
        self.cb_cutmix: Optional[QCheckBox] = None
        self.mix_alpha_spin: Optional[QDoubleSpinBox] = None
        self.crop_size_spin: Optional[QSpinBox] = None
        self.cb_randaug: QCheckBox = QCheckBox("RandAugment")
        self.randaug_num_ops_spin: QSpinBox = QSpinBox()
        self.randaug_magnitude_spin: QSpinBox = QSpinBox()
        self.cb_autoaug: QCheckBox = QCheckBox("AutoAugment")
        self.autoaug_policy_combo: QComboBox = QComboBox()

        # Hyperparameters & Training
        self.lr_spin: Optional[QDoubleSpinBox] = None
        self.momentum_spin: Optional[QDoubleSpinBox] = None
        self.wd_spin: Optional[QDoubleSpinBox] = None
        self.optimizer_combo: Optional[QComboBox] = None
        self.scheduler_combo: Optional[QComboBox] = None
        self.epochs_spin: Optional[QSpinBox] = None
        self.batch_spin: Optional[QSpinBox] = None
        self.cb_early_stopping: Optional[QCheckBox] = None

        # Early Stopping Settings
        self.es_monitor_combo: Optional[QComboBox] = None
        self.es_patience_spin: Optional[QSpinBox] = None
        self.es_min_delta_spin: Optional[QDoubleSpinBox] = None
        self.es_mode_combo: Optional[QComboBox] = None

        # Scheduler Options
        self.cb_steplr = QCheckBox("StepLR")
        self.steplr_step_size = QSpinBox()
        self.steplr_gamma = QDoubleSpinBox()
        self.cb_reduce_lr = QCheckBox("ReduceLROnPlateau")
        self.reduce_lr_patience = QSpinBox()
        self.reduce_lr_factor = QDoubleSpinBox()
        self.cb_cosine = QCheckBox("CosineAnnealing")
        self.cosine_T_max = QSpinBox()
        self.cb_cosine_warm = QCheckBox("CosineAnnealingWarmRestarts")
        self.cosine_warm_T0 = QSpinBox()
        self.cosine_warm_Tmult = QSpinBox()
        self.cosine_warm_eta_min = QDoubleSpinBox()
        self.cb_cyclic = QCheckBox("CyclicLR")
        self.cyclic_base_lr = QDoubleSpinBox()
        self.cyclic_max_lr = QDoubleSpinBox()
        self.cyclic_step_size_up = QSpinBox()
        self.cb_drop_path: QCheckBox = QCheckBox("Drop Path")
        self.drop_path_spin: QDoubleSpinBox = QDoubleSpinBox()
        self.loss_function_combo: QComboBox = QComboBox()
        self.cb_decay_steps: QCheckBox = QCheckBox("Decay Steps")
        self.decay_steps_spin: QSpinBox = QSpinBox()
        self.cb_linear_warmup: QCheckBox = QCheckBox("Linear Warmup")
        self.warmup_epochs_spin: QSpinBox = QSpinBox()

        # Extra Options
        self.cb_run_gc: Optional[QCheckBox] = None
        self.cb_enable_tta: Optional[QCheckBox] = None
        self.cb_profile_memory: Optional[QCheckBox] = None

        # NEW CODE: Non-blocking GPU Transfers
        self.cb_non_blocking: Optional[QCheckBox] = None

        # Multiprocessing and Export
        self.btn_train: Optional[QPushButton] = None
        self.btn_stop: Optional[QPushButton] = None
        self.train_process: Optional[multiprocessing.Process] = None
        self.train_result_manager = None
        self.train_result_dict = None
        self.training_timer: Optional[QTimer] = None
        self.best_ckpt_path: Optional[str] = None
        self.last_test_metrics: Optional[Dict[str, Any]] = None

        # Inference UI elements
        self.inference_checkpoint_edit: Optional[QLineEdit] = None
        self.inference_image_edit: Optional[QLineEdit] = None
        self.inference_run_button: Optional[QPushButton] = None
        self.inference_result_label: Optional[ClickableLabel] = None
        self.export_gradcam_button: Optional[QPushButton] = None

        # GradCAM
        self.topk_spin = QSpinBox()
        self.gradcam_combo = QComboBox()
        self.colormap_combo = QComboBox()
        self.target_layer_combo = QComboBox()
        self.inference_preview_label = ClickableLabel("GradCAM Preview")
        self.alpha_spin = QDoubleSpinBox()
        self.current_overlay_pixmap = None

        # Bulk Inference
        self.bulk_folder_edit: Optional[QLineEdit] = None
        self.gt_csv_edit: Optional[QLineEdit] = None
        self.bulk_inference_button: Optional[QPushButton] = None
        self.bulk_export_button: Optional[QPushButton] = None
        self.bulk_results = []
        self.bulk_progress_bar = QProgressBar()

    def create_tab(self) -> QWidget:
        self.widget_main = QWidget()
        main_layout = QVBoxLayout(self.widget_main)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

        # ---------- "Dataset & Splits" group ----------
        group_data = QGroupBox("Dataset & Splits")
        form_data = QFormLayout()
        self.train_data_dir_edit = QLineEdit()
        btn_browse_data = QPushButton("Browse...")
        btn_browse_data.clicked.connect(self.browse_dataset_folder)
        h_data = QHBoxLayout()
        h_data.addWidget(self.train_data_dir_edit)
        h_data.addWidget(btn_browse_data)
        form_data.addRow("Dataset Folder:", h_data)

        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0, 100)
        self.val_split_spin.setValue(15.0)
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0, 100)
        self.test_split_spin.setValue(15.0)
        h_splits = QHBoxLayout()
        h_splits.addWidget(self.val_split_spin)
        h_splits.addWidget(self.test_split_spin)
        form_data.addRow("Val/Test Split (%):", h_splits)

        self.cb_allow_grayscale = QCheckBox("Allow Grayscale")
        self.cb_preload_images = QCheckBox("Preload Images")
        self.cb_memmap_images = QCheckBox("Use NumPy's memmap")

        h_preload = QHBoxLayout()
        h_preload.addWidget(self.cb_allow_grayscale)
        h_preload.addWidget(self.cb_preload_images)
        h_preload.addWidget(self.cb_memmap_images)
        form_data.addRow(h_preload)

        group_data.setLayout(form_data)
        container_layout.addWidget(group_data)

        # ---------- Architecture & Basic Options ----------
        group_arch = QGroupBox("Architecture & Basic Options")
        form_arch = QFormLayout()
        self.model_variant_combo = QComboBox()
        # STABLE v1 MODELS
        self.model_variant_combo.addItem("ConvNeXt-Tiny (v1)", "convnext_tiny")
        self.model_variant_combo.addItem("ConvNeXt-Small (v1)", "convnext_small")
        self.model_variant_combo.addItem("ConvNeXt-Base (v1)", "convnext_base")
        self.model_variant_combo.addItem("ConvNeXt-Large (v1)", "convnext_large")
        self.model_variant_combo.addItem("ConvNeXt-XLarge (v1)", "convnext_xlarge")
        # STABLE v2 MODELS
        self.model_variant_combo.addItem("ConvNeXt-V2-Tiny", "convnextv2_tiny")
        self.model_variant_combo.addItem("ConvNeXt-V2-Small", "convnextv2_small")
        self.model_variant_combo.addItem("ConvNeXt-V2-Base", "convnextv2_base")
        self.model_variant_combo.addItem("ConvNeXt-V2-Large", "convnextv2_large")
        self.model_variant_combo.addItem("ConvNeXt-V2-Huge", "convnextv2_huge")

        form_arch.addRow("Model Variant:", self.model_variant_combo)

        self.weighted_loss_cb = QCheckBox("Weighted Loss")
        self.cb_normalize_pixel_intensity = QCheckBox("Normalize Pixels")
        form_arch.addRow(self.weighted_loss_cb, self.cb_normalize_pixel_intensity)
        group_arch.setLayout(form_arch)
        container_layout.addWidget(group_arch)

        # ---------- Basic Augmentations ----------
        group_basic = QGroupBox("Basic Augmentations")
        form_basic = QFormLayout()
        self.cb_bright_contrast = QCheckBox("Brightness/Contrast")
        self.cb_hue_sat = QCheckBox("Hue/Saturation")
        self.cb_gauss_noise = QCheckBox("Gaussian Noise")
        self.cb_rotation = QCheckBox("Rotation")
        form_basic.addRow("Effects:",
                          self._hbox([self.cb_bright_contrast, self.cb_hue_sat,
                                      self.cb_gauss_noise, self.cb_rotation]))

        self.cb_flip = QCheckBox("Flip")
        self.flip_mode_combo = QComboBox()
        self.flip_mode_combo.addItems(["horizontal", "vertical", "both"])
        self.flip_prob_spin = QDoubleSpinBox()
        self.flip_prob_spin.setRange(0.0, 1.0)
        self.flip_prob_spin.setValue(0.5)
        self.cb_crop = QCheckBox("Random Crop")
        form_basic.addRow("Flip Settings:", self._hbox([
            self.cb_flip, QLabel("Mode:"), self.flip_mode_combo,
            QLabel("Prob:"), self.flip_prob_spin, self.cb_crop
        ]))

        group_basic.setLayout(form_basic)
        container_layout.addWidget(group_basic)

        # ---------- Advanced Augmentations ----------
        group_adv = QGroupBox("Advanced Augmentations")
        form_adv = QFormLayout()
        self.cb_mixup = QCheckBox("Mixup")
        self.cb_cutmix = QCheckBox("Cutmix")
        self.mix_alpha_spin = QDoubleSpinBox()
        self.mix_alpha_spin.setRange(0.0, 10.0)
        self.mix_alpha_spin.setValue(1.0)
        form_adv.addRow("Mixup/Cutmix:",
                        self._hbox([self.cb_mixup, self.cb_cutmix,
                                    QLabel("Mix Alpha:"), self.mix_alpha_spin]))

        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setRange(32, 1024)
        self.crop_size_spin.setValue(224)
        form_adv.addRow("Crop Size:", self.crop_size_spin)

        self.randaug_num_ops_spin.setRange(1, 10)
        self.randaug_num_ops_spin.setValue(2)
        self.randaug_magnitude_spin.setRange(1, 10)
        self.randaug_magnitude_spin.setValue(9)
        self.autoaug_policy_combo.addItems(["IMAGENET", "CIFAR10"])

        adv_aug_layout = QHBoxLayout()
        adv_aug_layout.addWidget(self.cb_randaug)
        adv_aug_layout.addWidget(QLabel("Num Ops:"))
        adv_aug_layout.addWidget(self.randaug_num_ops_spin)
        adv_aug_layout.addWidget(QLabel("Magnitude:"))
        adv_aug_layout.addWidget(self.randaug_magnitude_spin)
        adv_aug_layout.addWidget(self.cb_autoaug)
        adv_aug_layout.addWidget(QLabel("Policy:"))
        adv_aug_layout.addWidget(self.autoaug_policy_combo)

        form_adv.addRow("Augment Options:", adv_aug_layout)
        group_adv.setLayout(form_adv)
        container_layout.addWidget(group_adv)

        # ---------- Hyperparameters & Training ----------
        group_hp = QGroupBox("Hyperparameters & Training")
        form_hp = QFormLayout()

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setValue(1e-4)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setValue(0.9)
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0.0, 1.0)
        self.wd_spin.setDecimals(6)
        self.wd_spin.setValue(1e-4)
        form_hp.addRow("LR:", self.lr_spin)
        form_hp.addRow("Momentum:", self.momentum_spin)
        form_hp.addRow("Weight Decay:", self.wd_spin)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd", "adamw"])
        form_hp.addRow("Optimizer:", self.optimizer_combo)

        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems([
            "none", "steplr", "reducelronplateau", "cosineannealing",
            "cosinedecay", "cosineannealingwarmrestarts", "cycliclr"
        ])
        form_hp.addRow("Scheduler:", self.scheduler_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(5)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(8)
        h_epoch = QHBoxLayout()
        h_epoch.addWidget(QLabel("Epochs:"))
        h_epoch.addWidget(self.epochs_spin)
        h_epoch.addWidget(QLabel("Batch:"))
        h_epoch.addWidget(self.batch_spin)
        self.cb_early_stopping = QCheckBox("Early Stopping")
        h_epoch.addWidget(self.cb_early_stopping)
        form_hp.addRow(h_epoch)

        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setValue(0.0)
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.9)
        self.label_smoothing_spin.setValue(0.0)
        self.clip_val_spin = QDoubleSpinBox()
        self.clip_val_spin.setRange(0, 10)
        self.clip_val_spin.setValue(0)
        form_hp.addRow("Dropout:", self.dropout_spin)
        form_hp.addRow("Label Smoothing:", self.label_smoothing_spin)
        form_hp.addRow("Clip Value:", self.clip_val_spin)

        form_hp.addRow("Drop Path:", self._hbox([self.cb_drop_path, QLabel("Rate:"), self.drop_path_spin]))

        self.loss_function_combo.addItems(["cross_entropy", "focal", "bce_single_logit"])
        form_hp.addRow("Loss Function:", self.loss_function_combo)

        self.cb_mixed_precision = QCheckBox("Mixed Precision")
        form_hp.addRow(self.cb_mixed_precision)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 32)
        self.num_workers_spin.setValue(8)
        self.cb_grad_accum = QCheckBox("Grad Accumulation")
        self.accum_batches_spin = QSpinBox()
        self.accum_batches_spin.setRange(1, 64)
        self.accum_batches_spin.setValue(2)
        self.accum_batches_spin.setEnabled(False)
        self.cb_grad_accum.stateChanged.connect(
            lambda: self.accum_batches_spin.setEnabled(self.cb_grad_accum.isChecked()))
        self.cb_persistent_workers = QCheckBox("Persistent Dataloader Workers")

        h_workers = QHBoxLayout()
        h_workers.addWidget(QLabel("Workers:"))
        h_workers.addWidget(self.num_workers_spin)
        h_workers.addWidget(self.cb_grad_accum)
        h_workers.addWidget(QLabel("Accumulate Batches:"))
        h_workers.addWidget(self.accum_batches_spin)
        h_workers.addWidget(self.cb_persistent_workers)
        form_hp.addRow(h_workers)

        self.check_val_every_n_epoch = QSpinBox()
        self.check_val_every_n_epoch.setRange(1, 50)
        self.check_val_every_n_epoch.setValue(1)
        h_val = QHBoxLayout()
        h_val.addWidget(QLabel("Validate Every N Epochs:"))
        h_val.addWidget(self.check_val_every_n_epoch)
        form_hp.addRow(h_val)

        group_hp.setLayout(form_hp)
        container_layout.addWidget(group_hp)

        # ---------- Early Stopping Settings ----------
        group_es = QGroupBox("Early Stopping Settings")
        form_es = QFormLayout()
        self.es_monitor_combo = QComboBox()
        self.es_monitor_combo.addItems(["val_loss", "val_acc"])
        form_es.addRow("Monitor:", self.es_monitor_combo)
        self.es_patience_spin = QSpinBox()
        self.es_patience_spin.setRange(1, 20)
        self.es_patience_spin.setValue(5)
        form_es.addRow("Patience:", self.es_patience_spin)
        self.es_min_delta_spin = QDoubleSpinBox()
        self.es_min_delta_spin.setRange(0.0, 1.0)
        self.es_min_delta_spin.setDecimals(4)
        self.es_min_delta_spin.setValue(0.0)
        form_es.addRow("Min Delta:", self.es_min_delta_spin)
        self.es_mode_combo = QComboBox()
        self.es_mode_combo.addItems(["min", "max"])
        form_es.addRow("Mode:", self.es_mode_combo)
        group_es.setLayout(form_es)
        container_layout.addWidget(group_es)

        # ---------- Scheduler Options ----------
        group_sched = QGroupBox("Scheduler Options")
        grid_sched = QGridLayout()
        grid_sched.addWidget(self.cb_steplr, 0, 0)
        grid_sched.addWidget(QLabel("Step Size:"), 0, 1)
        self.steplr_step_size.setRange(1, 10000)
        self.steplr_step_size.setValue(10)
        grid_sched.addWidget(self.steplr_step_size, 0, 2)
        grid_sched.addWidget(QLabel("Gamma:"), 0, 3)
        self.steplr_gamma.setRange(0.0, 10.0)
        self.steplr_gamma.setDecimals(2)
        self.steplr_gamma.setValue(0.1)
        grid_sched.addWidget(self.steplr_gamma, 0, 4)

        grid_sched.addWidget(self.cb_reduce_lr, 1, 0)
        grid_sched.addWidget(QLabel("Patience:"), 1, 1)
        self.reduce_lr_patience.setRange(1, 100)
        self.reduce_lr_patience.setValue(5)
        grid_sched.addWidget(self.reduce_lr_patience, 1, 2)
        grid_sched.addWidget(QLabel("Factor:"), 1, 3)
        self.reduce_lr_factor.setRange(0.0, 1.0)
        self.reduce_lr_factor.setDecimals(2)
        self.reduce_lr_factor.setValue(0.1)
        grid_sched.addWidget(self.reduce_lr_factor, 1, 4)

        grid_sched.addWidget(self.cb_cosine, 2, 0)
        grid_sched.addWidget(QLabel("T_max:"), 2, 1)
        self.cosine_T_max.setRange(1, 10000)
        self.cosine_T_max.setValue(10)
        grid_sched.addWidget(self.cosine_T_max, 2, 2)

        grid_sched.addWidget(self.cb_cosine_warm, 3, 0)
        grid_sched.addWidget(QLabel("T_0:"), 3, 1)
        self.cosine_warm_T0.setRange(1, 10000)
        self.cosine_warm_T0.setValue(10)
        grid_sched.addWidget(self.cosine_warm_T0, 3, 2)
        grid_sched.addWidget(QLabel("T_mult:"), 3, 3)
        self.cosine_warm_Tmult.setRange(1, 100)
        self.cosine_warm_Tmult.setValue(2)
        grid_sched.addWidget(self.cosine_warm_Tmult, 3, 4)
        grid_sched.addWidget(QLabel("eta_min:"), 3, 5)
        self.cosine_warm_eta_min.setRange(0.0, 1.0)
        self.cosine_warm_eta_min.setDecimals(6)
        self.cosine_warm_eta_min.setValue(1e-6)
        grid_sched.addWidget(self.cosine_warm_eta_min, 3, 6)

        grid_sched.addWidget(self.cb_cyclic, 4, 0)
        grid_sched.addWidget(QLabel("Base LR:"), 4, 1)
        self.cyclic_base_lr.setRange(0.0, 1.0)
        self.cyclic_base_lr.setDecimals(6)
        self.cyclic_base_lr.setValue(1e-4)
        grid_sched.addWidget(self.cyclic_base_lr, 4, 2)
        grid_sched.addWidget(QLabel("Max LR:"), 4, 3)
        self.cyclic_max_lr.setRange(0.0, 10.0)
        self.cyclic_max_lr.setDecimals(6)
        self.cyclic_max_lr.setValue(1e-2)
        grid_sched.addWidget(self.cyclic_max_lr, 4, 4)
        grid_sched.addWidget(QLabel("Step Size Up:"), 4, 5)
        self.cyclic_step_size_up.setRange(1, 10000)
        self.cyclic_step_size_up.setValue(2000)
        grid_sched.addWidget(self.cyclic_step_size_up, 4, 6)

        grid_sched.addWidget(self.cb_decay_steps, 5, 0)
        grid_sched.addWidget(QLabel("Decay Steps:"), 5, 1)
        self.decay_steps_spin.setRange(1, 10000)
        self.decay_steps_spin.setValue(10)
        grid_sched.addWidget(self.decay_steps_spin, 5, 2)

        grid_sched.addWidget(self.cb_linear_warmup, 6, 1)
        grid_sched.addWidget(QLabel("Warmup Epochs:"), 6, 2)
        self.warmup_epochs_spin.setRange(1, 100)
        self.warmup_epochs_spin.setValue(5)
        grid_sched.addWidget(self.warmup_epochs_spin, 6, 3)

        group_sched.setLayout(grid_sched)
        container_layout.addWidget(group_sched)

        # ---------- Extra Options ----------
        group_extra = QGroupBox("Extra Options")
        form_extra = QFormLayout()
        self.cb_run_gc = QCheckBox("Run GC Each Epoch")
        self.cb_enable_tta = QCheckBox("Enable TTA")
        self.cb_profile_memory = QCheckBox("Profile Memory")

        # NEW CODE: Add a Non-blocking GPU Transfers checkbox
        self.cb_non_blocking = QCheckBox("Non-Blocking GPU Transfers")

        h_extra = QHBoxLayout()
        h_extra.addWidget(self.cb_run_gc)
        h_extra.addWidget(self.cb_enable_tta)
        h_extra.addWidget(self.cb_profile_memory)
        h_extra.addWidget(self.cb_non_blocking)
        form_extra.addRow(h_extra)
        group_extra.setLayout(form_extra)
        container_layout.addWidget(group_extra)

        # ---------- Export & Progress ----------
        group_export = QGroupBox("Export & Progress")
        h_export = QHBoxLayout()
        self.btn_export_results = QPushButton("Export Results")
        self.btn_export_results.setEnabled(False)
        self.btn_export_results.clicked.connect(self.export_all_results)
        h_export.addWidget(self.btn_export_results)
        group_export.setLayout(h_export)
        container_layout.addWidget(group_export)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        container_layout.addWidget(self.progress_bar)

        # ---------- Inference Section ----------
        group_inference = QGroupBox("Inference")
        form_inference = QFormLayout()
        self.inference_checkpoint_edit = QLineEdit()
        btn_browse_checkpoint = QPushButton("Browse Checkpoint...")
        btn_browse_checkpoint.clicked.connect(self.browse_checkpoint_file)
        h_inference_checkpoint = QHBoxLayout()
        h_inference_checkpoint.addWidget(self.inference_checkpoint_edit)
        h_inference_checkpoint.addWidget(btn_browse_checkpoint)
        form_inference.addRow("Checkpoint File:", h_inference_checkpoint)

        self.inference_image_edit = QLineEdit()
        btn_browse_image = QPushButton("Browse Image...")
        btn_browse_image.clicked.connect(self.browse_image_file)
        h_inference_image = QHBoxLayout()
        h_inference_image.addWidget(self.inference_image_edit)
        h_inference_image.addWidget(btn_browse_image)
        form_inference.addRow("Image File:", h_inference_image)

        self.topk_spin.setRange(1, 100)
        self.topk_spin.setValue(1)
        form_inference.addRow("Top-K Predictions:", self.topk_spin)
        self.gradcam_combo.addItems(["None", "Gradcam", "Gradcam++", "XGradcam"])
        form_inference.addRow("GradCAM Method:", self.gradcam_combo)
        self.colormap_combo.addItems(["Jet", "hot", "Bone", "HSV"])
        form_inference.addRow("Color Map:", self.colormap_combo)
        form_inference.addRow("Target Layer:", self.target_layer_combo)
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setValue(0.5)
        form_inference.addRow("Overlay Alpha:", self.alpha_spin)

        self.inference_run_button = QPushButton("Run Inference")
        self.inference_run_button.clicked.connect(self.run_inference_button_clicked)
        form_inference.addRow(self.inference_run_button)
        self.inference_result_label = QLabel("Result: N/A")
        form_inference.addRow("Result:", self.inference_result_label)
        form_inference.addRow("Preview:", self.inference_preview_label)

        self.export_gradcam_button = QPushButton("Export GradCAM")
        self.export_gradcam_button.clicked.connect(self.export_gradcam_image)
        form_inference.addRow("Export GradCAM:", self.export_gradcam_button)

        group_inference.setLayout(form_inference)
        container_layout.addWidget(group_inference)

        # ---------- Bulk Inference Section ----------
        group_bulk = QGroupBox("Bulk Inference")
        form_bulk = QFormLayout()
        self.bulk_folder_edit = QLineEdit()
        btn_browse_bulk_folder = QPushButton("Browse Bulk Folder")
        btn_browse_bulk_folder.clicked.connect(self.browse_bulk_folder)
        h_bulk_folder = QHBoxLayout()
        h_bulk_folder.addWidget(self.bulk_folder_edit)
        h_bulk_folder.addWidget(btn_browse_bulk_folder)
        form_bulk.addRow("Bulk Folder:", h_bulk_folder)

        self.gt_csv_edit = QLineEdit()
        btn_browse_gt_csv = QPushButton("Browse GT CSV")
        btn_browse_gt_csv.clicked.connect(self.browse_gt_csv)
        h_gt_csv = QHBoxLayout()
        h_gt_csv.addWidget(self.gt_csv_edit)
        h_gt_csv.addWidget(btn_browse_gt_csv)
        form_bulk.addRow("Ground Truth CSV:", h_gt_csv)

        self.bulk_inference_button = QPushButton("Run Bulk Inference")
        self.bulk_inference_button.clicked.connect(self.run_bulk_inference_clicked)
        form_bulk.addRow(self.bulk_inference_button)

        form_bulk.addRow("Progress:", self.bulk_progress_bar)
        self.bulk_progress_bar.setVisible(False)

        self.bulk_export_button = QPushButton("Export Bulk Results CSV")
        self.bulk_export_button.clicked.connect(self.export_bulk_results)
        form_bulk.addRow(self.bulk_export_button)

        group_bulk.setLayout(form_bulk)
        container_layout.addWidget(group_bulk)

        # ---------- Training Controls ----------
        group_train = QGroupBox("Training Controls")
        h_train = QHBoxLayout()
        self.btn_train = QPushButton("Start Training")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        h_train.addWidget(self.btn_train)
        h_train.addWidget(self.btn_stop)
        group_train.setLayout(h_train)
        container_layout.addWidget(group_train)

        return self.widget_main

    def _hbox(self, widgets: List[QWidget]) -> QHBoxLayout:
        layout = QHBoxLayout()
        for w in widgets:
            layout.addWidget(w)
        return layout

    def browse_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Dataset Folder")
        if folder:
            self.train_data_dir_edit.setText(folder)

    def export_all_results(self):
        if not self.last_test_metrics:
            QMessageBox.warning(self.widget_main, "No Results", "No test metrics available.")
            return
        fpath, _ = QFileDialog.getSaveFileName(self.widget_main, "Export Results", filter="*.txt")
        if not fpath:
            return
        cr = self.last_test_metrics.get("classification_report", "")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("=== ConvNeXt Training Exported Results ===\n\n")
            f.write(f"Best Checkpoint Path: {self.best_ckpt_path}\n\n")
            f.write(f"Classification Report:\n{cr}\n")
        QMessageBox.information(self.widget_main, "Export Results", f"Results exported to {fpath}")

    def browse_checkpoint_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self.widget_main, "Select Checkpoint File", filter="*.ckpt")
        if file_path:
            self.inference_checkpoint_edit.setText(file_path)

    def browse_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.widget_main, "Select Image File",
            filter="Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path:
            self.inference_image_edit.setText(file_path)

    def run_inference_button_clicked(self):
        checkpoint_path = self.inference_checkpoint_edit.text().strip()
        image_path = self.inference_image_edit.text().strip()
        crop_size = self.crop_size_spin.value()
        topk = self.topk_spin.value()

        if not os.path.isfile(checkpoint_path):
            QMessageBox.warning(self.widget_main, "Invalid Checkpoint", "Please select a valid checkpoint file.")
            return
        if not os.path.isfile(image_path):
            QMessageBox.warning(self.widget_main, "Invalid Image", "Please select a valid image file.")
            return

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = MaidClassifier.load_from_checkpoint(checkpoint_path)
            model.eval()
            model.to(device)

            # Populate target_layer_combo if empty
            if self.target_layer_combo.count() == 0:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        self.target_layer_combo.addItem(name)

            inference_transform = A.Compose([
                A.Resize(height=crop_size, width=crop_size, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225), p=1.0),
                ToTensorV2()
            ])

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = inference_transform(image=image_rgb)
            image_tensor = augmented["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(image_tensor)
                probabilities = F.softmax(logits, dim=1)
                topk_probs, topk_indices = torch.topk(probabilities, k=topk, dim=1)
                topk_probs = topk_probs.cpu().numpy().flatten()
                topk_indices = topk_indices.cpu().numpy().flatten()

            if model.class_names:
                topk_classes = [model.class_names[i] for i in topk_indices]
            else:
                topk_classes = [str(i) for i in topk_indices]

            result_text = f"Top {topk} Predictions:\n"
            for cls_, conf in zip(topk_classes, topk_probs):
                result_text += f"{cls_}: {conf * 100:.2f}%\n"
            self.inference_result_label.setText(result_text)

            # GradCAM
            if self.gradcam_combo.currentText() != "None":
                target_layer_name = self.target_layer_combo.currentText()
                target_layer = get_module_by_name(model, target_layer_name)
                if target_layer is None:
                    QMessageBox.warning(self.widget_main, "Invalid Layer", "Selected target layer not found.")
                    return

                gradcam = GradCAM(model, target_layer)
                target_class = topk_indices[0]
                cam = gradcam.generate(
                    image_tensor, target_class,
                    gradcam_type=self.gradcam_combo.currentText()
                )
                alpha_value = self.alpha_spin.value()
                overlay = overlay_heatmap_on_image(
                    image_rgb,
                    cam,
                    colormap_name=self.colormap_combo.currentText(),
                    alpha=alpha_value
                )
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                height, width, channel = overlay_rgb.shape
                bytes_per_line = 3 * width
                qimg = QImage(overlay_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.inference_preview_label.setPixmap(
                    pixmap.scaled(self.inference_preview_label.size(), Qt.KeepAspectRatio)
                )
                self.current_overlay_pixmap = pixmap
                self.inference_preview_label.clicked.connect(self.open_overlay_viewer)
        except Exception as e:
            QMessageBox.critical(self.widget_main, "Inference Error", str(e))

    def open_overlay_viewer(self):
        if self.current_overlay_pixmap is not None:
            viewer = ImageViewerDialog(self.current_overlay_pixmap, self.widget_main)
            viewer.exec_()

    def export_gradcam_image(self):
        if self.current_overlay_pixmap is None:
            QMessageBox.warning(self.widget_main, "No GradCAM", "No GradCAM image available to export.")
            return
        fpath, _ = QFileDialog.getSaveFileName(
            self.widget_main, "Export GradCAM Image",
            filter="Images (*.png *.jpg)"
        )
        if not fpath:
            return
        if self.current_overlay_pixmap.save(fpath):
            QMessageBox.information(self.widget_main, "Export", f"GradCAM image exported to {fpath}")
        else:
            QMessageBox.warning(self.widget_main, "Export Failed", "Failed to export GradCAM image.")

    def browse_bulk_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Bulk Inference Folder")
        if folder:
            self.bulk_folder_edit.setText(folder)

    def browse_gt_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self.widget_main, "Select Ground Truth CSV", filter="*.csv")
        if file_path:
            self.gt_csv_edit.setText(file_path)

    def run_bulk_inference_clicked(self):
        checkpoint_path = self.inference_checkpoint_edit.text().strip()
        bulk_folder = self.bulk_folder_edit.text().strip()
        crop_size = self.crop_size_spin.value()
        topk = self.topk_spin.value()
        gt_csv = self.gt_csv_edit.text().strip()

        if not os.path.isfile(checkpoint_path):
            QMessageBox.warning(self.widget_main, "Invalid Checkpoint",
                                "Please select a valid checkpoint file for inference.")
            return
        if not os.path.isdir(bulk_folder):
            QMessageBox.warning(self.widget_main, "Invalid Folder", "Please select a valid folder for bulk inference.")
            return

        self.bulk_inference_button.setEnabled(False)
        self.bulk_progress_bar.setVisible(True)
        self.bulk_progress_bar.setValue(0)

        self.bulk_worker = BulkInferenceWorker(checkpoint_path, bulk_folder, crop_size, topk, gt_csv)
        self.bulk_worker.progress.connect(self.on_bulk_inference_progress)
        self.bulk_worker.finished.connect(self.on_bulk_inference_finished)
        self.bulk_worker.error.connect(self.on_bulk_inference_error)
        self.bulk_worker.start()

    def on_bulk_inference_progress(self, current, total):
        self.bulk_progress_bar.setMaximum(total)
        self.bulk_progress_bar.setValue(current)

    def on_bulk_inference_finished(self, results):
        self.bulk_results = results
        QMessageBox.information(self.widget_main, "Bulk Inference", f"Processed {len(results)} images.")
        self.bulk_inference_button.setEnabled(True)
        self.bulk_progress_bar.setVisible(False)

    def on_bulk_inference_error(self, error_msg):
        QMessageBox.critical(self.widget_main, "Bulk Inference Error", error_msg)
        self.bulk_inference_button.setEnabled(True)
        self.bulk_progress_bar.setVisible(False)

    def export_bulk_results(self):
        if not self.bulk_results:
            QMessageBox.warning(self.widget_main, "No Results", "No bulk inference results available.")
            return
        fpath, _ = QFileDialog.getSaveFileName(
            self.widget_main,
            "Export Bulk Inference CSV",
            filter="*.csv"
        )
        if not fpath:
            return
        fieldnames = ["image_path", "topk_classes", "topk_confidences"]
        if any("true_label" in r for r in self.bulk_results):
            fieldnames.extend(["true_label", "correct"])

        with open(fpath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for res in self.bulk_results:
                writer.writerow(res)

        QMessageBox.information(self.widget_main, "Export", f"Bulk results exported to {fpath}")

    def start_training(self):
        try:
            data_config, train_config = self._parse_common_params()
        except ValueError as e:
            QMessageBox.warning(self.widget_main, "Invalid Input", str(e))
            return

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export_results.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.train_result_manager = multiprocessing.Manager()
        self.train_result_dict = self.train_result_manager.dict()

        data_dict = data_config.__dict__
        train_dict = train_config.__dict__

        self.train_process = multiprocessing.Process(
            target=train_worker,
            args=(data_dict, train_dict, self.train_result_dict)
        )
        self.train_process.start()

        self.training_timer = QTimer()
        self.training_timer.timeout.connect(self.check_train_process)
        self.training_timer.start(1000)

    def stop_training(self):
        if self.train_process is not None and self.train_process.is_alive():
            self.train_process.terminate()
            self.train_process.join()
            self.train_process = None
            if self.training_timer is not None:
                self.training_timer.stop()
            self.progress_bar.setVisible(False)
            self.btn_train.setEnabled(True)
            self.btn_stop.setEnabled(False)
            QMessageBox.information(self.widget_main, "Training Stopped", "The training process has been stopped.")

    def check_train_process(self):
        if self.train_process is not None and not self.train_process.is_alive():
            self.train_process.join()
            self.train_process = None
            self.training_timer.stop()
            self.train_finished()

    def train_finished(self):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)

        status = self.train_result_dict.get("status", "ERROR")
        if status == "OK":
            self.best_ckpt_path = self.train_result_dict["ckpt_path"]
            self.last_test_metrics = self.train_result_dict["test_metrics"]
            self.btn_export_results.setEnabled(True)
            print(f"[INFO] Training finished. Best checkpoint: {self.best_ckpt_path}")
        else:
            print("[ERROR] Training encountered an error.")

        self.train_result_manager = None
        self.train_result_dict = None

    def _parse_common_params(self) -> Tuple[DataConfig, TrainConfig]:
        dataset_dir = self.train_data_dir_edit.text().strip()
        if not os.path.isdir(dataset_dir):
            raise ValueError("Invalid dataset folder.")
        val_split = self.val_split_spin.value() / 100.0
        test_split = self.test_split_spin.value() / 100.0
        if val_split + test_split > 1.0:
            raise ValueError("Val + Test split cannot exceed 100%.")

        scheduler_params = {}
        scheduler_type = self.scheduler_combo.currentText().lower()

        if scheduler_type == "steplr":
            if self.cb_steplr.isChecked():
                scheduler_params["step_size"] = self.steplr_step_size.value()
                scheduler_params["gamma"] = self.steplr_gamma.value()
        elif scheduler_type == "reducelronplateau":
            if self.cb_reduce_lr.isChecked():
                scheduler_params["patience"] = self.reduce_lr_patience.value()
                scheduler_params["factor"] = self.reduce_lr_factor.value()
        elif scheduler_type == "cosineannealing":
            if self.cb_cosine.isChecked():
                scheduler_params["t_max"] = self.cosine_T_max.value()
        elif scheduler_type == "cosinedecay":
            if self.cb_decay_steps.isChecked():
                scheduler_params["decay_steps"] = self.decay_steps_spin.value()
            if self.cb_linear_warmup.isChecked():
                scheduler_params["warmup_epochs"] = self.warmup_epochs_spin.value()
        elif scheduler_type == "cosineannealingwarmrestarts":
            if self.cb_cosine_warm.isChecked():
                scheduler_params["T_0"] = self.cosine_warm_T0.value()
                scheduler_params["T_mult"] = self.cosine_warm_Tmult.value()
                scheduler_params["eta_min"] = self.cosine_warm_eta_min.value()
        elif scheduler_type == "cycliclr":
            if self.cb_cyclic.isChecked():
                scheduler_params["base_lr"] = self.cyclic_base_lr.value()
                scheduler_params["max_lr"] = self.cyclic_max_lr.value()
                scheduler_params["step_size_up"] = self.cyclic_step_size_up.value()

        data_config = DataConfig(
            root_dir=dataset_dir,
            val_split=val_split,
            test_split=test_split,
            batch_size=self.batch_spin.value(),
            allow_grayscale=self.cb_allow_grayscale.isChecked(),
            preload_images=self.cb_preload_images.isChecked(),
            use_memmap_images=self.cb_memmap_images.isChecked()
        )

        # NEW CODE: read non_blocking from the new QCheckBox
        train_config = TrainConfig(
            max_epochs=self.epochs_spin.value(),
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
            normalize_pixel_intensity=self.cb_normalize_pixel_intensity.isChecked(),
            use_mixup=self.cb_mixup.isChecked(),
            use_cutmix=self.cb_cutmix.isChecked(),
            mix_alpha=self.mix_alpha_spin.value(),
            dropout_rate=self.dropout_spin.value(),
            label_smoothing=self.label_smoothing_spin.value(),
            loss_function=self.loss_function_combo.currentText(),
            gradient_clip_val=self.clip_val_spin.value(),
            use_mixed_precision=self.cb_mixed_precision.isChecked(),
            enable_grad_accum=self.cb_grad_accum.isChecked(),
            accumulate_grad_batches=self.accum_batches_spin.value(),
            check_val_every_n_epoch=self.check_val_every_n_epoch.value(),
            num_workers=self.num_workers_spin.value(),
            val_center_crop=False,
            random_crop_prob=0.0,
            random_crop_scale_min=0.0,
            random_crop_scale_max=0.0,
            pretrained_weights=True,
            run_gc=self.cb_run_gc.isChecked(),
            enable_tta=self.cb_enable_tta.isChecked(),
            profile_memory=self.cb_profile_memory.isChecked(),
            persistent_workers=self.cb_persistent_workers.isChecked(),
            crop_size=self.crop_size_spin.value(),
            model_variant=self.model_variant_combo.currentData(),
            drop_path_rate=self.cb_drop_path.isChecked() and self.drop_path_spin.value() or 0.0,
            use_randaug=self.cb_randaug.isChecked(),
            use_autoaug=self.cb_autoaug.isChecked(),
            randaug_num_ops=self.randaug_num_ops_spin.value(),
            randaug_magnitude=self.randaug_magnitude_spin.value(),
            autoaug_policy=self.autoaug_policy_combo.currentText(),

            # Our new checkbox
            non_blocking=self.cb_non_blocking.isChecked()
        )

        return data_config, train_config

# -------------------------------------------------------------------------
# 21) UNIT TESTS
# -------------------------------------------------------------------------
class DummyTest(unittest.TestCase):
    def test_dummy(self):
        self.assertTrue(True)

def run_tests():
    unittest.main(argv=[''], exit=False)

# -------------------------------------------------------------------------
# 22) INFERENCE FUNCTION (for CLI usage)
# -------------------------------------------------------------------------
def run_inference(checkpoint_path: str, image_path: str, crop_size: int = 224):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MaidClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    inference_transform = A.Compose([
        A.Resize(height=crop_size, width=crop_size, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
        ToTensorV2()
    ])

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = inference_transform(image=image_rgb)
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        top1_prob, top1_idx = torch.max(probabilities, dim=1)

    if model.class_names:
        prediction = model.class_names[top1_idx.item()]
        print(f"Predicted Class: {prediction} (index {top1_idx.item()}) "
              f"with confidence {top1_prob.item() * 100:.2f}%")
    else:
        print(f"Predicted Class Index: {top1_idx.item()} "
              f"with confidence {top1_prob.item() * 100:.2f}%")

# -------------------------------------------------------------------------
# 23) MAIN ENTRY POINT
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train or run inference with the MaidClassifier.")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file (.ckpt) for inference")
    parser.add_argument("--image", type=str, help="Path to the image file for inference")
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size for inference (default: 224)")
    parser.add_argument("--test", action="store_true", help="Run unit tests instead of training/inference")
    args, _ = parser.parse_known_args()

    if args.test:
        run_tests()
        return

    if args.checkpoint and args.image:
        run_inference(args.checkpoint, args.image, args.crop_size)
    else:
        app = QApplication(sys.argv)
        plugin = Plugin()
        main_widget = plugin.create_tab()
        main_widget.setWindowTitle("ConvNeXt Training GUI")
        main_widget.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
