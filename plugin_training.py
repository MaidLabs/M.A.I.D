# plugin_training.py

import os
import glob
import traceback
import shutil
from collections import Counter
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QDoubleSpinBox, QSpinBox,
    QCheckBox, QComboBox, QTextEdit, QProgressBar, QMessageBox
)

from base_plugin import BasePlugin  # Adjust to your project's base plugin path

try:
    from pytorch_optimizer import Lamb as LambClass
    HAVE_LAMB = True
except ImportError:
    LambClass = None
    HAVE_LAMB = False


def gather_samples_and_classes(root_dir):
    """
    Returns:
      samples: list of (image_path, class_idx)
      classes: list of class names, sorted
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"Invalid root_dir: {root_dir}")

    classes = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    samples = []
    for cls_name in classes:
        sub_dir = os.path.join(root_dir, cls_name)
        for ext in ("*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg"):
            pattern = os.path.join(sub_dir, "**", ext)
            file_list = glob.glob(pattern, recursive=True)
            for fpath in file_list:
                samples.append((fpath, class_to_idx[cls_name]))

    return samples, classes


class AlbumentationsDataset(Dataset):
    def __init__(self, samples, transform=None, classes=None, allow_grayscale=False):
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.classes = classes if classes is not None else []
        self.allow_grayscale = allow_grayscale

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        image_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"Failed to load image: {fpath}")

        # Check if single-channel
        if len(image_bgr.shape) == 2:
            if not self.allow_grayscale:
                raise ValueError(
                    f"Encountered a single-channel (grayscale) image but 'allow_grayscale' is False: {fpath}"
                )
            else:
                # Convert to 3-channel BGR
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        # Convert BGR→RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image_rgb)
            image_tensor = augmented["image"].float()
        else:
            # Minimal fallback
            image_tensor = torch.from_numpy(np.transpose(image_rgb, (2, 0, 1))).float() / 255.0

        return image_tensor, label


def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
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


def custom_collate_fn(batch, use_mixup=False, use_cutmix=False, alpha=1.0):
    images, labels = list(zip(*batch))
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    # If both are selected, do a 50/50 random pick for each batch
    if use_mixup and use_cutmix:
        if random.random() < 0.5:
            # MixUp
            mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=alpha)
            return mixed_x, (y_a, y_b, lam)
        else:
            # CutMix
            cut_x, y_a, y_b, lam = cutmix_data(images, labels, alpha=alpha)
            return cut_x, (y_a, y_b, lam)
    elif use_mixup:
        mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=alpha)
        return mixed_x, (y_a, y_b, lam)
    elif use_cutmix:
        cut_x, y_a, y_b, lam = cutmix_data(images, labels, alpha=alpha)
        return cut_x, (y_a, y_b, lam)
    else:
        return images, labels


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MaidClassifier(pl.LightningModule):
    """
    Main classifier.

    - If loss_function == "bce_single_logit", we do a single-logit BCE approach
      (num_classes=1), using sigmoid+threshold for predictions.
    - If loss_function == "bce", we do BCEWithLogitsLoss only if num_classes=1,
      else fallback to cross-entropy.
    """
    def __init__(
        self,
        architecture="resnet18",
        num_classes=2,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        use_weighted_loss=False,
        class_weights=None,
        optimizer_name="adam",
        scheduler_name="none",
        scheduler_params=None,
        dropout_rate=0.0,
        label_smoothing=0.0,
        freeze_backbone=False,
        loss_function="cross_entropy",
        log_fn=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "scheduler_params", "log_fn"])

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
        self.class_names = None

        # We'll store this for warnings:
        self.log_fn = log_fn

        # Only warn if label smoothing + weighted CE are both used
        if self.use_weighted_loss and self.label_smoothing > 0.0 and loss_function == "cross_entropy":
            self._warn(
                "label_smoothing and class_weights both used. "
                "Proceeding with both. Some users prefer to disable smoothing in this case."
            )

        # Architecture selection
        arch = architecture.lower()
        if arch == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_feats = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        elif arch == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            in_feats = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        elif arch == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            in_feats = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        elif arch == "densenet":
            densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            in_feats = densenet.classifier.in_features
            densenet.classifier = nn.Identity()
            self.backbone = densenet
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        elif arch == "vgg":
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            in_feats = vgg.classifier[6].in_features
            vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
            self.backbone = vgg
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        elif arch == "inception":
            inception = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                aux_logits=False
            )
            inception.aux_logits = False
            in_feats = inception.fc.in_features
            inception.fc = nn.Identity()
            self.backbone = inception
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        elif arch == "mobilenet":
            mbnet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_feats = mbnet.classifier[1].in_features
            mbnet.classifier = nn.Identity()
            self.backbone = mbnet
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        elif arch == "efficientnet_b0":
            effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_feats = effnet.classifier[1].in_features
            effnet.classifier = nn.Identity()
            self.backbone = effnet
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        elif arch == "convnext_tiny":
            convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            in_feats = convnext.classifier[2].in_features
            convnext.classifier[2] = nn.Identity()
            self.backbone = convnext
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        elif arch == "convnext_large":
            convnext = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
            in_feats = convnext.classifier[2].in_features
            convnext.classifier[2] = nn.Identity()
            self.backbone = convnext
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Configure the chosen loss
        if loss_function == "cross_entropy":
            self.loss_fn = None  # We'll use the built-in _compute_loss with F.cross_entropy
        elif loss_function == "focal":
            self.loss_fn = FocalLoss()
        elif loss_function == "bce":
            # If user sets bce + num_classes==1 => use BCEWithLogitsLoss
            # else fallback to cross-entropy
            if num_classes > 1:
                self.loss_fn = None
                self._warn("BCE selected but num_classes > 1. Using CrossEntropyLoss instead.")
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_function == "bce_single_logit":
            # Force single logit usage
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            # fallback to cross-entropy
            self.loss_fn = None

    def _warn(self, msg: str):
        """Helper to emit warnings either to log_fn or fallback to print."""
        if self.log_fn is not None:
            self.log_fn(f"WARNING: {msg}\n")
        else:
            print(f"WARNING: {msg}")

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def enable_gradient_checkpointing(self):
        for module in self.backbone.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Handle MixUp/CutMix
        if isinstance(y, tuple):
            y_a, y_b, lam = y
            logits = self(x)
            if (self.loss_function in ("bce_single_logit", "bce")) and self.num_classes == 1:
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
            if (self.loss_function in ("bce_single_logit", "bce")) and self.num_classes == 1:
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
        x, y = batch
        logits = self(x)

        if (self.loss_function in ("bce_single_logit", "bce")) and self.num_classes == 1:
            # single-logit BCE for binary classification
            y_float = y.float().unsqueeze(1)
            loss = self.loss_fn(logits, y_float)
            prob = torch.sigmoid(logits)
            preds = (prob >= 0.5).long()  # shape [N,1]
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
        x, y = batch
        logits = self(x)

        if (self.loss_function in ("bce_single_logit", "bce")) and self.num_classes == 1:
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

    def configure_optimizers(self):
        # Pop warmup and unify monitor for reduce-on-plateau
        warmup_epochs = self.scheduler_params.pop("warmup_epochs", 0)
        monitor_metric = self.scheduler_params.pop("monitor", "val_loss")

        opt_name = self.optimizer_name.lower()
        if opt_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif opt_name == "lamb":
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
                    optimizer, schedulers=[warmup_scheduler, main_scheduler],
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

    def _compute_loss(self, logits, targets):
        # Weighted cross-entropy vs normal CE
        if self.use_weighted_loss and (self.class_weights is not None):
            wt = torch.tensor(self.class_weights, device=logits.device, dtype=torch.float32)
            return F.cross_entropy(logits, targets, weight=wt, label_smoothing=self.label_smoothing)
        else:
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

    def _compute_loss_custom(self, logits, targets):
        if isinstance(self.loss_fn, FocalLoss):
            return self.loss_fn(logits, targets)
        elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            # For single-logit BCE
            return self.loss_fn(logits, targets.float().unsqueeze(1))
        else:
            return self._compute_loss(logits, targets)


# ===============================
# ADVANCED PARTIAL FREEZING
# ===============================
def apply_partial_freeze(model, freeze_config):
    """
    freeze_config is a dict of booleans, for example:
      {
        "freeze_entire_backbone": bool,
        "conv1_bn1": bool,
        "layer1": bool,
        "layer2": bool,
        "layer3": bool,
        "layer4": bool,
        "convnext_block0": bool,
        "convnext_block1": bool,
        "convnext_block2": bool,
        "convnext_block3": bool,
      }
    We freeze those submodules if True.
    """
    backbone = model.backbone
    arch_name = backbone.__class__.__name__.lower()

    # Freeze entire backbone if specified
    if freeze_config.get("freeze_entire_backbone", False):
        for param in backbone.parameters():
            param.requires_grad = False
        return  # if entire backbone is frozen, we can stop here

    # If it's a ResNet-like
    if "resnet" in arch_name:
        # conv1+bn1
        if freeze_config.get("conv1_bn1", False):
            if hasattr(backbone, "conv1"):
                for p in backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(backbone, "bn1"):
                for p in backbone.bn1.parameters():
                    p.requires_grad = False

        # layer1..layer4
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            if freeze_config.get(layer_name, False):
                layer_module = getattr(backbone, layer_name, None)
                if layer_module is not None:
                    for p in layer_module.parameters():
                        p.requires_grad = False

    # If it's a ConvNeXt
    elif "convnext" in arch_name:
        if hasattr(backbone, "features"):
            # Typically we have:
            #   backbone.features[0] = Stem
            #   backbone.features[1..4] = Stages 1..4
            #   backbone.features[5] = Final LayerNorm
            # We'll map convnext_block0..3 => features[1..4].
            for idx in range(4):
                key = f"convnext_block{idx}"
                if freeze_config.get(key, False):
                    block_index = idx + 1
                    if block_index < len(backbone.features):
                        block_module = backbone.features[block_index]
                        for p in block_module.parameters():
                            p.requires_grad = False


class CollateFnWrapper:
    def __init__(self, use_mixup=False, use_cutmix=False, alpha=1.0):
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.alpha = alpha

    def __call__(self, batch):
        return custom_collate_fn(batch, self.use_mixup, self.use_cutmix, self.alpha)


def run_training_once(data_params, train_params, log_fn):
    """
    Performs a single training run using the given data_params and train_params.
    Returns (best_ckpt_path, test_metrics, val_loss).
    """
    root_dir = data_params["root_dir"]
    val_split = data_params["val_split"]
    test_split = data_params["test_split"]
    batch_size = data_params["batch_size"]
    allow_grayscale = data_params.get("allow_grayscale", False)

    max_epochs = train_params["max_epochs"]
    architecture = train_params["architecture"]
    lr = train_params["lr"]
    momentum = train_params["momentum"]
    weight_decay = train_params["weight_decay"]
    use_weighted_loss = train_params["use_weighted_loss"]
    optimizer_name = train_params["optimizer_name"]
    scheduler_name = train_params["scheduler_name"]
    scheduler_params = dict(train_params["scheduler_params"])
    do_early_stopping = train_params["do_early_stopping"]

    # NEW Early Stopping parameters:
    early_stopping_monitor = train_params.get("early_stopping_monitor", "val_loss")
    early_stopping_patience = train_params.get("early_stopping_patience", 5)
    early_stopping_min_delta = train_params.get("early_stopping_min_delta", 0.0)
    early_stopping_mode = train_params.get("early_stopping_mode", "min")

    brightness_contrast = train_params["brightness_contrast"]
    hue_saturation = train_params["hue_saturation"]
    gaussian_noise = train_params["gaussian_noise"]
    rotation = train_params["use_rotation"]
    flip = train_params["use_flip"]
    flip_mode = train_params["flip_mode"]
    flip_prob = train_params["flip_prob"]
    crop = train_params["use_crop"]
    elastic = train_params["use_elastic"]
    normalize_pixel_intensity = train_params["normalize_pixel_intensity"]

    use_grid_distortion = train_params["use_grid_distortion"]
    use_optical_distortion = train_params["use_optical_distortion"]

    use_mixup = train_params["use_mixup"]
    use_cutmix = train_params["use_cutmix"]
    mix_alpha = train_params["mix_alpha"]

    dropout_rate = train_params["dropout_rate"]
    label_smoothing = train_params["label_smoothing"]
    freeze_backbone = train_params["freeze_backbone"]
    loss_function = train_params["loss_function"]

    gradient_clip_val = train_params["gradient_clip_val"]
    use_lr_finder = train_params["use_lr_finder"]
    use_tensorboard = train_params["use_tensorboard"]
    use_mixed_precision = train_params["use_mixed_precision"]
    warmup_epochs = train_params["warmup_epochs"]
    use_inception_299 = train_params.get("use_inception_299", False)

    enable_gradient_checkpointing = train_params["enable_gradient_checkpointing"]
    enable_grad_accum = train_params["enable_grad_accum"]
    accumulate_grad_batches = train_params["accumulate_grad_batches"]
    check_val_every_n_epoch = train_params["check_val_every_n_epoch"]

    freeze_config = train_params["freeze_config"]

    num_workers = train_params["num_workers"]
    val_center_crop = train_params["val_center_crop"]

    accept_lr_suggestion = train_params["accept_lr_suggestion"]

    # Random crop probability and scale range
    random_crop_prob = train_params.get("random_crop_prob", 1.0)
    random_crop_scale_min = train_params.get("random_crop_scale_min", 0.8)
    random_crop_scale_max = train_params.get("random_crop_scale_max", 1.0)

    log_fn("Gathering samples...\n")
    samples, class_names = gather_samples_and_classes(root_dir)

    n_total = len(samples)
    if n_total < 2:
        raise ValueError("Dataset has insufficient images.")

    log_fn(f"Found {n_total} total images.\n")

    val_size = int(val_split * n_total)
    test_size = int(test_split * n_total)
    train_size = n_total - val_size - test_size
    if train_size <= 0:
        raise ValueError("Train set size is zero. Adjust splits.")

    if val_size == 0:
        log_fn("WARNING: Validation set has 0 samples.\n")
    if test_size == 0:
        log_fn("WARNING: Test set has 0 samples.\n")

    indices = np.arange(n_total)
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    # Decide final crop dimension
    if architecture.lower() == "inception" and use_inception_299:
        final_crop_dim = 299
        bigger_resize = 320
    else:
        final_crop_dim = 224
        bigger_resize = 256

    log_fn("Preparing transforms...\n")

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

    # === DO NOT MODIFY RANDOM CROP CODE BELOW ===
    if crop:
        train_augs.append(
            A.RandomResizedCrop(
                height=final_crop_dim,
                width=final_crop_dim,
                scale=(random_crop_scale_min, random_crop_scale_max),
                p=random_crop_prob
            )
        )
    else:
        train_augs.append(A.Resize(final_crop_dim, final_crop_dim))
    # === DO NOT CHANGE THIS RANDOM CROP CODE ===

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

    # Official ImageNet mean/std
    if normalize_pixel_intensity:
        train_augs.append(A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ))

    train_augs.append(ToTensorV2())
    train_transform = A.Compose(train_augs)

    # Validation transforms
    val_augs = [
        A.Resize(bigger_resize, bigger_resize)
    ]
    if val_center_crop:
        val_augs.append(A.CenterCrop(final_crop_dim, final_crop_dim, p=1.0))
    if normalize_pixel_intensity:
        val_augs.append(A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ))
    val_augs.append(ToTensorV2())
    val_transform = A.Compose(val_augs)

    # Test transforms
    test_augs = [
        A.Resize(bigger_resize, bigger_resize)
    ]
    if val_center_crop:
        test_augs.append(A.CenterCrop(final_crop_dim, final_crop_dim, p=1.0))
    if normalize_pixel_intensity:
        test_augs.append(A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ))
    test_augs.append(ToTensorV2())
    test_transform = A.Compose(test_augs)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    log_fn(f"Splits => Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}\n")
    num_classes = len(class_names)
    log_fn(f"Detected {num_classes} classes: {class_names}\n")

    # If user explicitly chose single-logit BCE, force num_classes=1 in the model
    if loss_function == "bce_single_logit":
        model_num_classes = 1
    else:
        model_num_classes = num_classes

    class_weights = None
    if use_weighted_loss and loss_function != "bce_single_logit":
        # Weighted loss for multi-class only
        all_labels = [lbl for _, lbl in samples]
        label_counter = Counter(all_labels)
        freq = [label_counter[i] for i in range(num_classes)]
        if any(c == 0 for c in freq):
            log_fn("WARNING: One or more classes have zero samples. Weighted loss may be undefined.\n")
        class_weights = [1.0 / c if c > 0 else 0.0 for c in freq]
        log_fn(f"Using Weighted Loss: {class_weights}\n")

    # Unify monitor for reduce-on-plateau with early stopping
    scheduler_params["warmup_epochs"] = warmup_epochs
    scheduler_params["monitor"] = early_stopping_monitor

    # Build model
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
        log_fn=log_fn
    )
    model.class_names = class_names

    # Apply advanced partial freezing AFTER constructing the model
    apply_partial_freeze(model, freeze_config)

    if enable_gradient_checkpointing:
        try:
            model.enable_gradient_checkpointing()
        except Exception as e:
            log_fn(f"Could not enable gradient checkpointing: {e}\n")

    # -- Fix #2: Use the same monitor and mode for both checkpoint and early-stopping
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

    logger = None
    tb_log_dir = None
    if use_tensorboard:
        logger = TensorBoardLogger(save_dir="tb_logs", name="experiment")
        tb_log_dir = logger.log_dir

    log_fn("Initializing Trainer...\n")
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        precision=16 if use_mixed_precision else 32,
        accumulate_grad_batches=accumulate_grad_batches if enable_grad_accum else 1,
        check_val_every_n_epoch=check_val_every_n_epoch
    )

    collate_fn = None
    if use_mixup or use_cutmix:
        collate_fn = CollateFnWrapper(use_mixup=use_mixup, use_cutmix=use_cutmix, alpha=mix_alpha)

    train_ds = AlbumentationsDataset(
        train_samples, transform=train_transform, classes=class_names, allow_grayscale=allow_grayscale
    )
    val_ds = AlbumentationsDataset(
        val_samples, transform=val_transform, classes=class_names, allow_grayscale=allow_grayscale
    )
    test_ds = AlbumentationsDataset(
        test_samples, transform=test_transform, classes=class_names, allow_grayscale=allow_grayscale
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    if use_lr_finder:
        log_fn("Running LR finder...\n")
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        new_lr = lr_finder.suggestion()
        log_fn(f"LR Finder suggests learning rate: {new_lr}\n")
        if accept_lr_suggestion:
            log_fn(f"Applying LR Finder suggestion: {new_lr}\n")
            model.lr = new_lr
        else:
            log_fn("User declined LR suggestion. Keeping original LR.\n")

    log_fn(f"Starting training with batch_size={batch_size}...\n")
    trainer.fit(model, train_loader, val_loader)
    log_fn("Training finished.\n")

    best_ckpt_path = ckpt_callback.best_model_path
    log_fn(f"Best checkpoint: {best_ckpt_path}\n")

    # Evaluate on val set
    val_results = trainer.validate(model, val_loader, ckpt_path=best_ckpt_path)
    val_loss = val_results[0]["val_loss"] if len(val_results) > 0 else None

    log_fn("Running test...\n")
    test_results = trainer.test(ckpt_path=best_ckpt_path, dataloaders=test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = MaidClassifier.load_from_checkpoint(best_ckpt_path)
    best_model.class_names = model.class_names
    best_model.to(device)
    best_model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = best_model(x)

            if (best_model.loss_function in ("bce_single_logit", "bce")) and best_model.num_classes == 1:
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
        if (best_model.loss_function in ("bce_single_logit", "bce")) and best_model.num_classes == 1:
            # binary classification
            cm = confusion_matrix(all_targets, all_preds)
            cr = classification_report(all_targets, all_preds, zero_division=0)
            auc_roc = roc_auc_score(all_targets, all_probs)
        else:
            # multi-class
            cm = confusion_matrix(all_targets, all_preds)
            cr = classification_report(
                all_targets, all_preds,
                target_names=best_model.class_names,
                zero_division=0
            )
            if best_model.num_classes == 2:
                auc_roc = roc_auc_score(all_targets, all_probs[:, 1])
            else:
                auc_roc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except Exception as e:
        log_fn(f"Could not compute some metrics: {e}\n")

    if cm is not None:
        log_fn(f"Confusion Matrix:\n{cm}\n")
    if cr is not None:
        log_fn(f"Classification Report:\n{cr}\n")
    if auc_roc is not None:
        log_fn(f"AUC-ROC: {auc_roc:.4f}\n")

    test_metrics = {
        "test_results": test_results,
        "confusion_matrix": cm.tolist() if cm is not None else None,
        "class_report": cr,
        "auc_roc": auc_roc,
        "tb_log_dir": tb_log_dir
    }

    return best_ckpt_path, test_metrics, val_loss


class TrainThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str, dict)

    def __init__(self, data_params, train_params, parent=None):
        super().__init__(parent)
        self.data_params = data_params
        self.train_params = train_params

    def run(self):
        try:
            seed_everything(42, workers=True)
            try:
                self._execute_training()
            except RuntimeError as re:
                if "CUDA out of memory" in str(re):
                    self.log_signal.emit("CUDA OOM error encountered. Trying to auto-adjust batch size.\n")
                    self.auto_adjust_batch_size()
                else:
                    raise
        except Exception as e:
            err_msg = f"ERROR during training: {e}\n{traceback.format_exc()}"
            self.log_signal.emit(err_msg)
            self.done_signal.emit("ERROR", {})

    def auto_adjust_batch_size(self):
        while True:
            try:
                current_bs = self.data_params["batch_size"]
                new_bs = current_bs // 2
                if new_bs < 1:
                    raise RuntimeError(
                        "Ran out of memory and cannot reduce batch size further. "
                        "Please try smaller images or different model architecture."
                    )
                self.log_signal.emit(
                    f"Out of memory at batch_size={current_bs}, trying batch_size={new_bs}...\n"
                )
                self.data_params["batch_size"] = new_bs
                self._execute_training()
                return
            except RuntimeError as re:
                if "CUDA out of memory" in str(re):
                    continue
                else:
                    raise

    def _execute_training(self):
        best_ckpt_path, test_metrics, _ = run_training_once(
            self.data_params, self.train_params, self.log_signal.emit
        )
        self.done_signal.emit(best_ckpt_path, test_metrics)


class OptunaTrainThread(QThread):
    """
    Optuna Tuning Thread.
    """
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str, dict)

    def __init__(self, data_params, base_train_params,
                 optuna_n_trials, optuna_timeout,
                 use_test_metric_for_optuna=False):
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

            def objective(trial: optuna.Trial):
                trial_train_params = dict(self.base_train_params)

                # Example hyperparam search space:
                lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
                dropout = trial.suggest_float("dropout_rate", 0.0, 0.7)
                optimizer_name = trial.suggest_categorical("optimizer_name", ["adam", "sgd", "adamw"])
                scheduler_name = trial.suggest_categorical("scheduler_name", ["none", "steplr", "cosineannealing"])

                trial_train_params["lr"] = lr
                trial_train_params["dropout_rate"] = dropout
                trial_train_params["optimizer_name"] = optimizer_name
                trial_train_params["scheduler_name"] = scheduler_name

                self.log_signal.emit(
                    f"Trial {trial.number}: lr={lr}, dropout={dropout}, "
                    f"opt={optimizer_name}, sch={scheduler_name}\n"
                )

                best_ckpt_path, test_metrics, val_loss = run_training_once(
                    self.data_params,
                    trial_train_params,
                    self.log_signal.emit
                )

                if not test_metrics:
                    return 9999.0

                # *** Not recommended to optimize directly on test set ***
                if self.use_test_metric_for_optuna:
                    # Use test_loss
                    test_loss = 9999.0
                    if "test_results" in test_metrics:
                        tr = test_metrics["test_results"]
                        if len(tr) > 0 and "test_loss" in tr[0]:
                            test_loss = tr[0]["test_loss"]
                    return test_loss
                else:
                    # Use val_loss
                    if val_loss is not None:
                        return val_loss
                    else:
                        return 9999.0

            study.optimize(
                objective,
                n_trials=self.optuna_n_trials,
                timeout=self.optuna_timeout if self.optuna_timeout > 0 else None
            )

            best_trial = study.best_trial
            self.log_signal.emit(f"Optuna best trial: {best_trial.number}, value={best_trial.value}\n")
            self.log_signal.emit(f"Best params: {best_trial.params}\n")

            # Re-train final model with best hyperparams
            best_params = dict(self.base_train_params)
            best_params["lr"] = best_trial.params["lr"]
            best_params["dropout_rate"] = best_trial.params["dropout_rate"]
            best_params["optimizer_name"] = best_trial.params["optimizer_name"]
            best_params["scheduler_name"] = best_trial.params["scheduler_name"]

            self.log_signal.emit("Re-training final model with best hyperparams...\n")
            best_ckpt_path, metrics_dict, _ = run_training_once(
                self.data_params, best_params, self.log_signal.emit
            )
            self.done_signal.emit(best_ckpt_path, metrics_dict)

        except Exception as e:
            err_msg = f"ERROR during Optuna tuning: {e}\n{traceback.format_exc()}"
            self.log_signal.emit(err_msg)
            self.done_signal.emit("ERROR", {})


class Plugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.plugin_name = "Training"

        self.widget_main = None
        self.train_data_dir_edit = None
        self.val_split_spin = None
        self.test_split_spin = None

        self.arch_combo = None
        self.weighted_loss_cb = None

        self.cb_bright_contrast = None
        self.cb_hue_sat = None
        self.cb_gauss_noise = None
        self.cb_rotation = None
        self.cb_flip = None
        self.flip_mode_combo = None
        self.flip_prob_spin = None
        self.cb_crop = None
        self.cb_elastic = None
        self.cb_normalize_pixel_intensity = None

        self.cb_grid_distortion = None
        self.cb_optical_distortion = None

        self.cb_mixup = None
        self.cb_cutmix = None
        self.mix_alpha_spin = None

        self.lr_spin = None
        self.momentum_spin = None
        self.wd_spin = None
        self.optimizer_combo = None
        self.scheduler_combo = None
        self.epochs_spin = None
        self.batch_spin = None
        self.cb_early_stopping = None
        self.scheduler_params_edit = None

        self.dropout_spin = None
        self.label_smoothing_spin = None
        self.cb_freeze_backbone = None
        self.loss_combo = None
        self.clip_val_spin = None

        self.cb_lr_finder = None
        self.cb_accept_lr_suggestion = None
        self.cb_tensorboard = None
        self.cb_mixed_precision = None
        self.warmup_epochs_spin = None
        self.cb_inception_299 = None

        self.num_workers_spin = None
        self.cb_grad_checkpoint = None
        self.cb_grad_accum = None
        self.accum_batches_spin = None
        self.check_val_every_n_epoch = None

        # New Early Stopping fields:
        self.es_monitor_combo = None
        self.es_patience_spin = None
        self.es_min_delta_spin = None
        self.es_mode_combo = None

        # Partial-freeze checkboxes:
        self.cb_freeze_conv1_bn1 = None
        self.cb_freeze_layer1 = None
        self.cb_freeze_layer2 = None
        self.cb_freeze_layer3 = None
        self.cb_freeze_layer4 = None

        self.cb_freeze_convnext_block0 = None
        self.cb_freeze_convnext_block1 = None
        self.cb_freeze_convnext_block2 = None
        self.cb_freeze_convnext_block3 = None

        self.cb_val_center_crop = None
        self.cb_allow_grayscale = None
        self.cb_optuna_use_test_metric = None

        self.random_crop_prob_spin = None
        self.random_crop_scale_min_spin = None
        self.random_crop_scale_max_spin = None

        self.btn_train = None
        self.btn_export_results = None
        self.progress_bar = None
        self.text_log = None

        self.train_thread = None
        self.best_ckpt_path = None
        self.last_test_metrics = None

        self.btn_tune_optuna = None
        self.optuna_trials_spin = None
        self.optuna_timeout_spin = None
        self.optuna_thread = None

    def create_tab(self) -> QWidget:
        self.widget_main = QWidget()
        layout = QVBoxLayout(self.widget_main)

        # === Dataset Folder ===
        h_data = QHBoxLayout()
        self.train_data_dir_edit = QLineEdit()
        btn_browse_data = QPushButton("Browse Data Folder...")
        btn_browse_data.clicked.connect(self.browse_dataset_folder)
        h_data.addWidget(QLabel("Dataset Folder:"))
        h_data.addWidget(self.train_data_dir_edit)
        h_data.addWidget(btn_browse_data)
        layout.addLayout(h_data)

        # === Splits ===
        h_splits = QHBoxLayout()
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0, 100)
        self.val_split_spin.setValue(15.0)
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0, 100)
        self.test_split_spin.setValue(15.0)
        h_splits.addWidget(QLabel("Val Split (%)"))
        h_splits.addWidget(self.val_split_spin)
        h_splits.addWidget(QLabel("Test Split (%)"))
        h_splits.addWidget(self.test_split_spin)
        layout.addLayout(h_splits)

        # === Architecture row ===
        h_arch = QHBoxLayout()
        self.arch_combo = QComboBox()
        self.arch_combo.addItems([
            "resnet18", "resnet50", "resnet101",
            "densenet", "vgg", "inception", "mobilenet",
            "efficientnet_b0", "convnext_tiny", "convnext_large"
        ])
        self.weighted_loss_cb = QCheckBox("Weighted Loss")
        self.cb_normalize_pixel_intensity = QCheckBox("Normalize Pixel")
        self.cb_inception_299 = QCheckBox("Use 299 for Inception")

        h_arch.addWidget(QLabel("Architecture:"))
        h_arch.addWidget(self.arch_combo)
        h_arch.addWidget(self.weighted_loss_cb)
        h_arch.addWidget(self.cb_normalize_pixel_intensity)
        h_arch.addWidget(self.cb_inception_299)
        layout.addLayout(h_arch)

        # Allow grayscale
        self.cb_allow_grayscale = QCheckBox("Allow Grayscale Images")
        layout.addWidget(self.cb_allow_grayscale)

        # === Augment row 1 ===
        group_aug = QHBoxLayout()
        self.cb_bright_contrast = QCheckBox("Brightness/Contrast")
        self.cb_hue_sat = QCheckBox("Hue/Sat")
        self.cb_gauss_noise = QCheckBox("GaussNoise")
        self.cb_rotation = QCheckBox("Rotation")
        group_aug.addWidget(self.cb_bright_contrast)
        group_aug.addWidget(self.cb_hue_sat)
        group_aug.addWidget(self.cb_gauss_noise)
        group_aug.addWidget(self.cb_rotation)
        layout.addLayout(group_aug)

        # === Augment row 2 ===
        group_aug2 = QHBoxLayout()
        self.cb_flip = QCheckBox("Flipping")
        self.flip_mode_combo = QComboBox()
        self.flip_mode_combo.addItems(["horizontal", "vertical", "both"])
        self.flip_prob_spin = QDoubleSpinBox()
        self.flip_prob_spin.setRange(0.0, 1.0)
        self.flip_prob_spin.setSingleStep(0.1)
        self.flip_prob_spin.setValue(0.5)
        self.cb_crop = QCheckBox("Random Crop")
        self.cb_elastic = QCheckBox("Elastic")
        group_aug2.addWidget(self.cb_flip)
        group_aug2.addWidget(QLabel("Mode:"))
        group_aug2.addWidget(self.flip_mode_combo)
        group_aug2.addWidget(QLabel("Prob:"))
        group_aug2.addWidget(self.flip_prob_spin)
        group_aug2.addWidget(self.cb_crop)
        group_aug2.addWidget(self.cb_elastic)
        layout.addLayout(group_aug2)

        # === Augment row 3 (random crop range) ===
        group_aug3 = QHBoxLayout()
        group_aug3.addWidget(QLabel("Random Crop p:"))
        self.random_crop_prob_spin = QDoubleSpinBox()
        self.random_crop_prob_spin.setRange(0.0, 1.0)
        self.random_crop_prob_spin.setSingleStep(0.1)
        self.random_crop_prob_spin.setValue(1.0)
        group_aug3.addWidget(self.random_crop_prob_spin)

        group_aug3.addWidget(QLabel("Scale Min:"))
        self.random_crop_scale_min_spin = QDoubleSpinBox()
        self.random_crop_scale_min_spin.setRange(0.0, 1.0)
        self.random_crop_scale_min_spin.setSingleStep(0.05)
        self.random_crop_scale_min_spin.setValue(0.8)
        group_aug3.addWidget(self.random_crop_scale_min_spin)

        group_aug3.addWidget(QLabel("Scale Max:"))
        self.random_crop_scale_max_spin = QDoubleSpinBox()
        self.random_crop_scale_max_spin.setRange(0.0, 1.0)
        self.random_crop_scale_max_spin.setSingleStep(0.05)
        self.random_crop_scale_max_spin.setValue(1.0)
        group_aug3.addWidget(self.random_crop_scale_max_spin)

        layout.addLayout(group_aug3)

        # === Augment row 4 (grid/optical, mixup/cutmix) ===
        group_aug4 = QHBoxLayout()
        self.cb_grid_distortion = QCheckBox("GridDistortion")
        self.cb_optical_distortion = QCheckBox("OpticalDistortion")
        self.cb_mixup = QCheckBox("MixUp")
        self.cb_cutmix = QCheckBox("CutMix")
        self.mix_alpha_spin = QDoubleSpinBox()
        self.mix_alpha_spin.setRange(0.0, 5.0)
        self.mix_alpha_spin.setSingleStep(0.1)
        self.mix_alpha_spin.setValue(1.0)
        group_aug4.addWidget(self.cb_grid_distortion)
        group_aug4.addWidget(self.cb_optical_distortion)
        group_aug4.addWidget(self.cb_mixup)
        group_aug4.addWidget(self.cb_cutmix)
        group_aug4.addWidget(QLabel("alpha:"))
        group_aug4.addWidget(self.mix_alpha_spin)
        layout.addLayout(group_aug4)

        # === Hyperparams row 1 ===
        h_params = QHBoxLayout()
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setValue(1e-4)

        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setSingleStep(0.1)
        self.momentum_spin.setValue(0.9)

        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0, 1.0)
        self.wd_spin.setDecimals(6)
        self.wd_spin.setValue(1e-4)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd", "adamw", "lamb"])

        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems([
            "none", "steplr", "reducelronplateau", "cosineannealing", "cycliclr"
        ])

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(5)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(8)

        self.cb_early_stopping = QCheckBox("Early Stopping")

        h_params.addWidget(QLabel("LR:"))
        h_params.addWidget(self.lr_spin)
        h_params.addWidget(QLabel("Momentum:"))
        h_params.addWidget(self.momentum_spin)
        h_params.addWidget(QLabel("WD:"))
        h_params.addWidget(self.wd_spin)
        h_params.addWidget(QLabel("Opt:"))
        h_params.addWidget(self.optimizer_combo)
        h_params.addWidget(QLabel("Sched:"))
        h_params.addWidget(self.scheduler_combo)
        h_params.addWidget(QLabel("Epochs:"))
        h_params.addWidget(self.epochs_spin)
        h_params.addWidget(QLabel("Batch:"))
        h_params.addWidget(self.batch_spin)
        h_params.addWidget(self.cb_early_stopping)
        layout.addLayout(h_params)

        # === Scheduler params ===
        self.scheduler_params_edit = QLineEdit("step_size=10,gamma=0.1")
        layout.addWidget(QLabel("Scheduler Params (key=val, comma-separated):"))
        layout.addWidget(self.scheduler_params_edit)

        # === Hyperparams row 2 (regularization) ===
        h_reg = QHBoxLayout()
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(0.0)

        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.9)
        self.label_smoothing_spin.setSingleStep(0.05)
        self.label_smoothing_spin.setValue(0.0)

        self.cb_freeze_backbone = QCheckBox("Freeze Entire Backbone")
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["cross_entropy", "focal", "bce", "bce_single_logit"])

        self.clip_val_spin = QDoubleSpinBox()
        self.clip_val_spin.setRange(0.0, 10.0)
        self.clip_val_spin.setSingleStep(0.1)
        self.clip_val_spin.setValue(0.0)

        h_reg.addWidget(QLabel("Dropout:"))
        h_reg.addWidget(self.dropout_spin)
        h_reg.addWidget(QLabel("LabelSmooth:"))
        h_reg.addWidget(self.label_smoothing_spin)
        h_reg.addWidget(self.cb_freeze_backbone)
        h_reg.addWidget(QLabel("Loss:"))
        h_reg.addWidget(self.loss_combo)
        h_reg.addWidget(QLabel("ClipVal:"))
        h_reg.addWidget(self.clip_val_spin)
        layout.addLayout(h_reg)

        # === Additional Options row 1 ===
        h_monitor = QHBoxLayout()
        self.cb_lr_finder = QCheckBox("LR Finder")
        self.cb_accept_lr_suggestion = QCheckBox("Accept LR Suggestion?")
        self.cb_tensorboard = QCheckBox("TensorBoard Logger")
        self.cb_mixed_precision = QCheckBox("Mixed Precision")
        self.warmup_epochs_spin = QSpinBox()
        self.warmup_epochs_spin.setRange(0, 100)
        self.warmup_epochs_spin.setValue(0)

        h_monitor.addWidget(self.cb_lr_finder)
        h_monitor.addWidget(self.cb_accept_lr_suggestion)
        h_monitor.addWidget(self.cb_tensorboard)
        h_monitor.addWidget(self.cb_mixed_precision)
        h_monitor.addWidget(QLabel("WarmupEpochs:"))
        h_monitor.addWidget(self.warmup_epochs_spin)
        layout.addLayout(h_monitor)

        # === Additional Options row 2 ===
        h_new1 = QHBoxLayout()
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 32)
        self.num_workers_spin.setValue(8)

        self.cb_grad_checkpoint = QCheckBox("Gradient Checkpointing")
        self.cb_grad_accum = QCheckBox("Gradient Accumulation")
        self.accum_batches_spin = QSpinBox()
        self.accum_batches_spin.setRange(1, 64)
        self.accum_batches_spin.setValue(2)
        self.accum_batches_spin.setEnabled(False)

        def toggle_accum_spin():
            self.accum_batches_spin.setEnabled(self.cb_grad_accum.isChecked())

        self.cb_grad_accum.stateChanged.connect(toggle_accum_spin)

        h_new1.addWidget(QLabel("Workers:"))
        h_new1.addWidget(self.num_workers_spin)
        h_new1.addWidget(self.cb_grad_checkpoint)
        h_new1.addWidget(self.cb_grad_accum)
        h_new1.addWidget(QLabel("Accumulate Batches:"))
        h_new1.addWidget(self.accum_batches_spin)
        layout.addLayout(h_new1)

        # === Additional Options row 3 ===
        h_new2 = QHBoxLayout()
        self.check_val_every_n_epoch = QSpinBox()
        self.check_val_every_n_epoch.setRange(1, 50)
        self.check_val_every_n_epoch.setValue(1)

        h_new2.addWidget(QLabel("Val every N epoch:"))
        h_new2.addWidget(self.check_val_every_n_epoch)
        layout.addLayout(h_new2)

        # === Advanced partial-freeze checkboxes for ResNet ===
        group_resnet_freeze = QHBoxLayout()
        group_resnet_freeze.addWidget(QLabel("ResNet Freeze:"))
        self.cb_freeze_conv1_bn1 = QCheckBox("conv1+bn1")
        self.cb_freeze_layer1 = QCheckBox("layer1")
        self.cb_freeze_layer2 = QCheckBox("layer2")
        self.cb_freeze_layer3 = QCheckBox("layer3")
        self.cb_freeze_layer4 = QCheckBox("layer4")
        group_resnet_freeze.addWidget(self.cb_freeze_conv1_bn1)
        group_resnet_freeze.addWidget(self.cb_freeze_layer1)
        group_resnet_freeze.addWidget(self.cb_freeze_layer2)
        group_resnet_freeze.addWidget(self.cb_freeze_layer3)
        group_resnet_freeze.addWidget(self.cb_freeze_layer4)
        layout.addLayout(group_resnet_freeze)

        # === Advanced partial-freeze checkboxes for ConvNeXt ===
        group_cnext_freeze = QHBoxLayout()
        group_cnext_freeze.addWidget(QLabel("ConvNeXt Freeze:"))
        self.cb_freeze_convnext_block0 = QCheckBox("Block0")
        self.cb_freeze_convnext_block1 = QCheckBox("Block1")
        self.cb_freeze_convnext_block2 = QCheckBox("Block2")
        self.cb_freeze_convnext_block3 = QCheckBox("Block3")
        group_cnext_freeze.addWidget(self.cb_freeze_convnext_block0)
        group_cnext_freeze.addWidget(self.cb_freeze_convnext_block1)
        group_cnext_freeze.addWidget(self.cb_freeze_convnext_block2)
        group_cnext_freeze.addWidget(self.cb_freeze_convnext_block3)
        layout.addLayout(group_cnext_freeze)

        # === Validation center crop ===
        self.cb_val_center_crop = QCheckBox("Center Crop for Validation/Test")
        layout.addWidget(self.cb_val_center_crop)

        # === Early Stopping parameter controls ===
        es_layout = QHBoxLayout()
        es_layout.addWidget(QLabel("ES Monitor:"))
        self.es_monitor_combo = QComboBox()
        self.es_monitor_combo.addItems(["val_loss", "val_acc"])
        es_layout.addWidget(self.es_monitor_combo)

        es_layout.addWidget(QLabel("Patience:"))
        self.es_patience_spin = QSpinBox()
        self.es_patience_spin.setRange(1, 20)
        self.es_patience_spin.setValue(5)
        es_layout.addWidget(self.es_patience_spin)

        es_layout.addWidget(QLabel("Min Delta:"))
        self.es_min_delta_spin = QDoubleSpinBox()
        self.es_min_delta_spin.setRange(0.0, 1.0)
        self.es_min_delta_spin.setSingleStep(0.0001)
        self.es_min_delta_spin.setValue(0.0)
        es_layout.addWidget(self.es_min_delta_spin)

        es_layout.addWidget(QLabel("Mode:"))
        self.es_mode_combo = QComboBox()
        self.es_mode_combo.addItems(["min", "max"])
        es_layout.addWidget(self.es_mode_combo)

        layout.addLayout(es_layout)

        # === Buttons & Logging ===
        self.btn_train = QPushButton("Start Training")
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)

        self.cb_optuna_use_test_metric = QCheckBox("Use Test Loss as Optuna Objective? (Not recommended)")
        self.cb_optuna_use_test_metric.setChecked(False)
        layout.addWidget(self.cb_optuna_use_test_metric)

        h_optuna = QHBoxLayout()
        self.btn_tune_optuna = QPushButton("Tune with Optuna")
        self.btn_tune_optuna.clicked.connect(self.start_optuna_tuning)
        self.optuna_trials_spin = QSpinBox()
        self.optuna_trials_spin.setRange(1, 100)
        self.optuna_trials_spin.setValue(5)
        self.optuna_timeout_spin = QSpinBox()
        self.optuna_timeout_spin.setRange(0, 100000)
        self.optuna_timeout_spin.setValue(0)
        h_optuna.addWidget(self.btn_tune_optuna)
        h_optuna.addWidget(QLabel("Trials:"))
        h_optuna.addWidget(self.optuna_trials_spin)
        h_optuna.addWidget(QLabel("Timeout (sec):"))
        h_optuna.addWidget(self.optuna_timeout_spin)
        layout.addLayout(h_optuna)

        self.btn_export_results = QPushButton("Export Results")
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
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Dataset Folder")
        if folder:
            self.train_data_dir_edit.setText(folder)

    def start_training(self):
        dataset_dir = self.train_data_dir_edit.text().strip()
        if not os.path.isdir(dataset_dir):
            QMessageBox.warning(self.widget_main, "Error", "Invalid dataset folder.")
            return

        val_split = self.val_split_spin.value() / 100.0
        test_split = self.test_split_spin.value() / 100.0

        # Parse scheduler params
        scheduler_params_str = self.scheduler_params_edit.text().strip()
        scheduler_params = {}
        if scheduler_params_str:
            parts = scheduler_params_str.split(",")
            for part in parts:
                k, v = part.split("=")
                k = k.strip()
                v = v.strip()
                try:
                    float_val = float(v)
                    scheduler_params[k] = float_val
                except ValueError:
                    err_msg = f"Invalid scheduler param value for {k}='{v}'. Must be numeric."
                    self.append_log(err_msg + "\n")
                    QMessageBox.warning(self.widget_main, "Invalid Scheduler Params", err_msg)
                    return

        data_params = {
            "root_dir": dataset_dir,
            "val_split": val_split,
            "test_split": test_split,
            "batch_size": self.batch_spin.value(),
            "allow_grayscale": self.cb_allow_grayscale.isChecked(),
        }

        # Build partial freeze config
        freeze_config = {
            "freeze_entire_backbone": self.cb_freeze_backbone.isChecked(),
            "conv1_bn1": self.cb_freeze_conv1_bn1.isChecked(),
            "layer1": self.cb_freeze_layer1.isChecked(),
            "layer2": self.cb_freeze_layer2.isChecked(),
            "layer3": self.cb_freeze_layer3.isChecked(),
            "layer4": self.cb_freeze_layer4.isChecked(),
            "convnext_block0": self.cb_freeze_convnext_block0.isChecked(),
            "convnext_block1": self.cb_freeze_convnext_block1.isChecked(),
            "convnext_block2": self.cb_freeze_convnext_block2.isChecked(),
            "convnext_block3": self.cb_freeze_convnext_block3.isChecked(),
        }

        train_params = {
            "max_epochs": self.epochs_spin.value(),
            "architecture": self.arch_combo.currentText(),
            "lr": self.lr_spin.value(),
            "momentum": self.momentum_spin.value(),
            "weight_decay": self.wd_spin.value(),
            "use_weighted_loss": self.weighted_loss_cb.isChecked(),
            "optimizer_name": self.optimizer_combo.currentText(),
            "scheduler_name": self.scheduler_combo.currentText(),
            "scheduler_params": scheduler_params,
            "do_early_stopping": self.cb_early_stopping.isChecked(),

            "early_stopping_monitor": self.es_monitor_combo.currentText(),
            "early_stopping_patience": self.es_patience_spin.value(),
            "early_stopping_min_delta": self.es_min_delta_spin.value(),
            "early_stopping_mode": self.es_mode_combo.currentText(),

            "brightness_contrast": self.cb_bright_contrast.isChecked(),
            "hue_saturation": self.cb_hue_sat.isChecked(),
            "gaussian_noise": self.cb_gauss_noise.isChecked(),
            "use_rotation": self.cb_rotation.isChecked(),
            "use_flip": self.cb_flip.isChecked(),
            "flip_mode": self.flip_mode_combo.currentText(),
            "flip_prob": self.flip_prob_spin.value(),
            "use_crop": self.cb_crop.isChecked(),
            "use_elastic": self.cb_elastic.isChecked(),
            "normalize_pixel_intensity": self.cb_normalize_pixel_intensity.isChecked(),

            "use_grid_distortion": self.cb_grid_distortion.isChecked(),
            "use_optical_distortion": self.cb_optical_distortion.isChecked(),

            "use_mixup": self.cb_mixup.isChecked(),
            "use_cutmix": self.cb_cutmix.isChecked(),
            "mix_alpha": self.mix_alpha_spin.value(),

            "dropout_rate": self.dropout_spin.value(),
            "label_smoothing": self.label_smoothing_spin.value(),
            "freeze_backbone": self.cb_freeze_backbone.isChecked(),
            "loss_function": self.loss_combo.currentText(),

            "gradient_clip_val": self.clip_val_spin.value(),
            "use_lr_finder": self.cb_lr_finder.isChecked(),
            "accept_lr_suggestion": self.cb_accept_lr_suggestion.isChecked(),
            "use_tensorboard": self.cb_tensorboard.isChecked(),
            "use_mixed_precision": self.cb_mixed_precision.isChecked(),
            "warmup_epochs": self.warmup_epochs_spin.value(),
            "use_inception_299": self.cb_inception_299.isChecked(),

            "enable_gradient_checkpointing": self.cb_grad_checkpoint.isChecked(),
            "enable_grad_accum": self.cb_grad_accum.isChecked(),
            "accumulate_grad_batches": self.accum_batches_spin.value(),
            "check_val_every_n_epoch": self.check_val_every_n_epoch.value(),

            "freeze_config": freeze_config,
            "num_workers": self.num_workers_spin.value(),
            "val_center_crop": self.cb_val_center_crop.isChecked(),

            "random_crop_prob": self.random_crop_prob_spin.value(),
            "random_crop_scale_min": self.random_crop_scale_min_spin.value(),
            "random_crop_scale_max": self.random_crop_scale_max_spin.value(),
        }

        self.btn_train.setEnabled(False)
        self.btn_export_results.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.train_thread = TrainThread(data_params, train_params)
        self.train_thread.log_signal.connect(self.append_log)
        self.train_thread.done_signal.connect(self.train_finished)
        self.train_thread.start()

    def append_log(self, msg: str):
        self.text_log.append(msg)
        self.text_log.ensureCursorVisible()

    def train_finished(self, ckpt_path: str, test_metrics: dict):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
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

    def export_all_results(self):
        if not self.last_test_metrics:
            QMessageBox.warning(self.widget_main, "No Results", "No test metrics available.")
            return

        fpath, _ = QFileDialog.getSaveFileName(self.widget_main, "Export Results", filter="*.txt")
        if not fpath:
            return

        cm = self.last_test_metrics.get("confusion_matrix")
        cr = self.last_test_metrics.get("class_report")
        auc_roc = self.last_test_metrics.get("auc_roc")
        tb_log_dir = self.last_test_metrics.get("tb_log_dir")

        with open(fpath, "w", encoding="utf-8") as f:
            f.write("=== CNN Training Exported Results ===\n\n")
            f.write(f"Best Checkpoint Path: {self.best_ckpt_path}\n\n")
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(f"Classification Report:\n{cr}\n\n")
            f.write(f"AUC-ROC: {auc_roc}\n")

        if tb_log_dir and os.path.isdir(tb_log_dir):
            zip_base = os.path.splitext(fpath)[0] + "_tb_logs"
            shutil.make_archive(zip_base, 'zip', tb_log_dir)
            self.append_log(f"TensorBoard logs exported to {zip_base}.zip\n")

        QMessageBox.information(self.widget_main, "Export Results", f"Results exported to {fpath}")

    def start_optuna_tuning(self):
        dataset_dir = self.train_data_dir_edit.text().strip()
        if not os.path.isdir(dataset_dir):
            QMessageBox.warning(self.widget_main, "Error", "Invalid dataset folder.")
            return

        val_split = self.val_split_spin.value() / 100.0
        test_split = self.test_split_spin.value() / 100.0

        scheduler_params_str = self.scheduler_params_edit.text().strip()
        scheduler_params = {}
        if scheduler_params_str:
            parts = scheduler_params_str.split(",")
            for part in parts:
                k, v = part.split("=")
                k = k.strip()
                v = v.strip()
                try:
                    float_val = float(v)
                    scheduler_params[k] = float_val
                except ValueError:
                    err_msg = f"Invalid scheduler param value for {k}='{v}'. Must be numeric."
                    self.append_log(err_msg + "\n")
                    QMessageBox.warning(self.widget_main, "Invalid Scheduler Params", err_msg)
                    return

        data_params = {
            "root_dir": dataset_dir,
            "val_split": val_split,
            "test_split": test_split,
            "batch_size": self.batch_spin.value(),
            "allow_grayscale": self.cb_allow_grayscale.isChecked(),
        }

        freeze_config = {
            "freeze_entire_backbone": self.cb_freeze_backbone.isChecked(),
            "conv1_bn1": self.cb_freeze_conv1_bn1.isChecked(),
            "layer1": self.cb_freeze_layer1.isChecked(),
            "layer2": self.cb_freeze_layer2.isChecked(),
            "layer3": self.cb_freeze_layer3.isChecked(),
            "layer4": self.cb_freeze_layer4.isChecked(),
            "convnext_block0": self.cb_freeze_convnext_block0.isChecked(),
            "convnext_block1": self.cb_freeze_convnext_block1.isChecked(),
            "convnext_block2": self.cb_freeze_convnext_block2.isChecked(),
            "convnext_block3": self.cb_freeze_convnext_block3.isChecked(),
        }

        base_train_params = {
            "max_epochs": self.epochs_spin.value(),
            "architecture": self.arch_combo.currentText(),
            "lr": self.lr_spin.value(),
            "momentum": self.momentum_spin.value(),
            "weight_decay": self.wd_spin.value(),
            "use_weighted_loss": self.weighted_loss_cb.isChecked(),
            "optimizer_name": self.optimizer_combo.currentText(),
            "scheduler_name": self.scheduler_combo.currentText(),
            "scheduler_params": scheduler_params,
            "do_early_stopping": self.cb_early_stopping.isChecked(),

            "early_stopping_monitor": self.es_monitor_combo.currentText(),
            "early_stopping_patience": self.es_patience_spin.value(),
            "early_stopping_min_delta": self.es_min_delta_spin.value(),
            "early_stopping_mode": self.es_mode_combo.currentText(),

            "brightness_contrast": self.cb_bright_contrast.isChecked(),
            "hue_saturation": self.cb_hue_sat.isChecked(),
            "gaussian_noise": self.cb_gauss_noise.isChecked(),
            "use_rotation": self.cb_rotation.isChecked(),
            "use_flip": self.cb_flip.isChecked(),
            "flip_mode": self.flip_mode_combo.currentText(),
            "flip_prob": self.flip_prob_spin.value(),
            "use_crop": self.cb_crop.isChecked(),
            "use_elastic": self.cb_elastic.isChecked(),
            "normalize_pixel_intensity": self.cb_normalize_pixel_intensity.isChecked(),

            "use_grid_distortion": self.cb_grid_distortion.isChecked(),
            "use_optical_distortion": self.cb_optical_distortion.isChecked(),

            "use_mixup": self.cb_mixup.isChecked(),
            "use_cutmix": self.cb_cutmix.isChecked(),
            "mix_alpha": self.mix_alpha_spin.value(),

            "dropout_rate": self.dropout_spin.value(),
            "label_smoothing": self.label_smoothing_spin.value(),
            "freeze_backbone": self.cb_freeze_backbone.isChecked(),
            "loss_function": self.loss_combo.currentText(),

            "gradient_clip_val": self.clip_val_spin.value(),
            "use_lr_finder": self.cb_lr_finder.isChecked(),
            "accept_lr_suggestion": self.cb_accept_lr_suggestion.isChecked(),
            "use_tensorboard": self.cb_tensorboard.isChecked(),
            "use_mixed_precision": self.cb_mixed_precision.isChecked(),
            "warmup_epochs": self.warmup_epochs_spin.value(),
            "use_inception_299": self.cb_inception_299.isChecked(),

            "enable_gradient_checkpointing": self.cb_grad_checkpoint.isChecked(),
            "enable_grad_accum": self.cb_grad_accum.isChecked(),
            "accumulate_grad_batches": self.accum_batches_spin.value(),
            "check_val_every_n_epoch": self.check_val_every_n_epoch.value(),

            "freeze_config": freeze_config,
            "num_workers": self.num_workers_spin.value(),
            "val_center_crop": self.cb_val_center_crop.isChecked(),

            "random_crop_prob": self.random_crop_prob_spin.value(),
            "random_crop_scale_min": self.random_crop_scale_min_spin.value(),
            "random_crop_scale_max": self.random_crop_scale_max_spin.value(),
        }

        self.btn_train.setEnabled(False)
        self.btn_tune_optuna.setEnabled(False)
        self.btn_export_results.setEnabled(False)
        self.progress_bar.setVisible(True)

        n_trials = self.optuna_trials_spin.value()
        timeout = self.optuna_timeout_spin.value()

        use_test_metric_for_optuna = self.cb_optuna_use_test_metric.isChecked()

        self.optuna_thread = OptunaTrainThread(
            data_params, base_train_params,
            n_trials, timeout,
            use_test_metric_for_optuna=use_test_metric_for_optuna
        )
        self.optuna_thread.log_signal.connect(self.append_log)
        self.optuna_thread.done_signal.connect(self.optuna_tuning_finished)
        self.optuna_thread.start()

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
