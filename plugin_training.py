# plugin_training.py

import os
import glob
import traceback
import shutil
import random
import time
import gc
import importlib.util
import platform
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

# For GPU VRAM checks (optional)
try:
    import pynvml
    HAVE_PYNVML = True
except ImportError:
    HAVE_PYNVML = False

# For plotting confusion matrix, etc. (optional)
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# For advanced optimizers
try:
    from pytorch_optimizer import Lamb as LambClass
    from pytorch_optimizer import Lion as LionClass
    HAVE_LAMB = True
    HAVE_LION = True
except ImportError:
    LambClass = None
    LionClass = None
    HAVE_LAMB = False
    HAVE_LION = False

# Attempt to import timm for additional models
try:
    import timm
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

# Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Sklearn metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

# === Removed QThread usage; keep PyQt imports for GUI ===
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QDoubleSpinBox, QSpinBox,
    QCheckBox, QComboBox, QProgressBar, QMessageBox,
    QGroupBox, QScrollArea
)

# For multiprocessing
import multiprocessing

# Base plugin stub if missing
try:
    from base_plugin import BasePlugin
except ImportError:
    class BasePlugin:
        def __init__(self):
            pass

# Optional: for performance in GPU training
torch.backends.cudnn.benchmark = True


# ======================
# 1) CONFIG CLASSES
# ======================
class DataConfig:
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
    def __init__(
        self,
        # Basic training params
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
        load_custom_model: bool = False,
        custom_model_path: str = "",
        custom_architecture_file: str = "",

        # Advanced augs
        random_gamma: bool = False,
        random_gamma_limit_low: float = 80.0,
        random_gamma_limit_high: float = 120.0,
        random_gamma_prob: float = 0.5,

        clahe: bool = False,
        clahe_clip_limit: float = 4.0,
        clahe_tile_size: int = 8,
        clahe_prob: float = 0.3,

        channel_shuffle: bool = False,
        channel_shuffle_prob: float = 0.2,

        use_posterize_solarize_equalize: bool = False,
        pse_prob: float = 0.2,
        posterize_bits: int = 4,
        solarize_threshold: int = 128,

        sharpen_denoise: bool = False,
        sharpen_prob: float = 0.3,
        sharpen_alpha_min: float = 0.2,
        sharpen_alpha_max: float = 0.5,
        sharpen_lightness_min: float = 0.5,
        sharpen_lightness_max: float = 1.0,

        gauss_vs_mult_noise: bool = False,
        gauss_mult_prob: float = 0.3,
        gauss_noise_var_limit_low: float = 10.0,
        gauss_noise_var_limit_high: float = 50.0,
        mult_noise_lower: float = 0.9,
        mult_noise_upper: float = 1.1,

        cutout_coarse_dropout: bool = False,
        cutout_max_holes: int = 8,
        cutout_max_height: int = 32,
        cutout_max_width: int = 32,
        cutout_prob: float = 0.5,

        use_shift_scale_rotate: bool = False,
        ssr_shift_limit: float = 0.1,
        ssr_scale_limit: float = 0.1,
        ssr_rotate_limit: int = 15,
        ssr_prob: float = 0.4,

        use_one_of_advanced_transforms: bool = False,
        one_of_advanced_transforms_prob: float = 0.5,

        # Specifically for CoAtNet: RandAugment usage
        use_randaugment: bool = False,

        # NEW: persistent_workers param
        persistent_workers: bool = False
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

        # Advanced augs
        self.random_gamma = random_gamma
        self.random_gamma_limit_low = random_gamma_limit_low
        self.random_gamma_limit_high = random_gamma_limit_high
        self.random_gamma_prob = random_gamma_prob

        self.clahe = clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.clahe_prob = clahe_prob

        self.channel_shuffle = channel_shuffle
        self.channel_shuffle_prob = channel_shuffle_prob

        self.use_posterize_solarize_equalize = use_posterize_solarize_equalize
        self.pse_prob = pse_prob
        self.posterize_bits = posterize_bits
        self.solarize_threshold = solarize_threshold

        self.sharpen_denoise = sharpen_denoise
        self.sharpen_prob = sharpen_prob
        self.sharpen_alpha_min = sharpen_alpha_min
        self.sharpen_alpha_max = sharpen_alpha_max
        self.sharpen_lightness_min = sharpen_lightness_min
        self.sharpen_lightness_max = sharpen_lightness_max

        self.gauss_vs_mult_noise = gauss_vs_mult_noise
        self.gauss_mult_prob = gauss_mult_prob
        self.gauss_noise_var_limit_low = gauss_noise_var_limit_low
        self.gauss_noise_var_limit_high = gauss_noise_var_limit_high
        self.mult_noise_lower = mult_noise_lower
        self.mult_noise_upper = mult_noise_upper

        self.cutout_coarse_dropout = cutout_coarse_dropout
        self.cutout_max_holes = cutout_max_holes
        self.cutout_max_height = cutout_max_height
        self.cutout_max_width = cutout_max_width
        self.cutout_prob = cutout_prob

        self.use_shift_scale_rotate = use_shift_scale_rotate
        self.ssr_shift_limit = ssr_shift_limit
        self.ssr_scale_limit = ssr_scale_limit
        self.ssr_rotate_limit = ssr_rotate_limit
        self.ssr_prob = ssr_prob

        self.use_one_of_advanced_transforms = use_one_of_advanced_transforms
        self.one_of_advanced_transforms_prob = one_of_advanced_transforms_prob

        # RandAugment for CoAtNet
        self.use_randaugment = use_randaugment

        # New param: persistent workers for DataLoader
        self.persistent_workers = persistent_workers


# ======================
# 2) DATASET & SPLITS
# ======================
def gather_samples_and_classes(root_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
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
                # Basic corrupt-file check
                try:
                    with open(fpath, "rb") as fp:
                        _ = fp.read(20)
                except OSError:
                    continue
                samples.append((fpath, class_to_idx[cls_name]))

    return samples, classes


class AlbumentationsDataset(Dataset):
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

        # Convert grayscale to BGR if allowed
        if len(image_bgr.shape) == 2:
            if not self.allow_grayscale:
                raise ValueError(
                    f"Single-channel image but allow_grayscale=False: {fpath}"
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


# ======================
# 3) MIXUP/CUTMIX
# ======================
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


# ======================
# 4) CUSTOM LOSSES
# ======================
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


# ======================
# 5) MAIN MODEL CLASS
# ======================
class MaidClassifier(pl.LightningModule):
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
        self.save_hyperparameters(ignore=["class_weights", "scheduler_params"])

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

        # Safety checks
        if self.loss_function == "bce" and self.num_classes > 1:
            print("[WARN] BCE with num_classes>1 is not recommended. Switching to cross_entropy.")
            self.loss_function = "cross_entropy"

        # Possibly load a custom model
        self.custom_model = None
        if self.load_custom_model:
            if self.custom_architecture_file and os.path.isfile(self.custom_architecture_file):
                # The user’s .py file with get_model(...)
                print(f"[INFO] Loading custom architecture from: {self.custom_architecture_file}")
                spec = importlib.util.spec_from_file_location("custom_arch", self.custom_architecture_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "get_model"):
                    self.custom_model = mod.get_model(num_classes=self.num_classes, pretrained=pretrained)
                    if os.path.isfile(self.custom_model_path):
                        self.custom_model.load_state_dict(
                            torch.load(self.custom_model_path, map_location="cpu")
                        )
                    else:
                        print(f"[WARN] Custom model weights not found: {self.custom_model_path}")
                else:
                    raise ValueError("No get_model() function found in custom architecture file.")
            else:
                # Load entire .pt / .pth
                if not os.path.isfile(self.custom_model_path):
                    raise ValueError(f"Custom model path not found: {self.custom_model_path}")
                print(f"[INFO] Loading entire custom model from: {self.custom_model_path}")
                self.custom_model = torch.load(self.custom_model_path, map_location="cpu")

        if self.custom_model is not None:
            self.backbone = None
            self.head = None
        else:
            self.backbone, in_feats = self._create_backbone(architecture, pretrained)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_feats, num_classes)
            )

        # Custom losses
        if self.loss_function == "focal":
            self.loss_fn = FocalLoss()
        elif self.loss_function == "bce_single_logit" and self.num_classes == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = None

    def _create_backbone(self, architecture: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """
        Create a backbone (and remove its final classification layer) for known architectures.
        Also supports some timm-based models if HAVE_TIMM = True.
        """

        def weight_if_pretrained_tv(weight_enum):
            return weight_enum if pretrained else None

        arch = architecture.lower()
        # TorchVision standard
        if arch == "resnet18":
            m = models.resnet18(weights=weight_if_pretrained_tv(models.ResNet18_Weights.IMAGENET1K_V1))
            in_feats = m.fc.in_features
            m.fc = nn.Identity()
            return m, in_feats
        elif arch == "resnet50":
            m = models.resnet50(weights=weight_if_pretrained_tv(models.ResNet50_Weights.IMAGENET1K_V2))
            in_feats = m.fc.in_features
            m.fc = nn.Identity()
            return m, in_feats
        elif arch == "resnet101":
            m = models.resnet101(weights=weight_if_pretrained_tv(models.ResNet101_Weights.IMAGENET1K_V2))
            in_feats = m.fc.in_features
            m.fc = nn.Identity()
            return m, in_feats
        elif arch == "densenet":
            densenet = models.densenet121(
                weights=weight_if_pretrained_tv(models.DenseNet121_Weights.IMAGENET1K_V1)
            )
            in_feats = densenet.classifier.in_features
            densenet.classifier = nn.Identity()
            return densenet, in_feats
        elif arch == "vgg":
            vgg = models.vgg16(weights=weight_if_pretrained_tv(models.VGG16_Weights.IMAGENET1K_V1))
            in_feats = vgg.classifier[6].in_features
            vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
            return vgg, in_feats
        elif arch == "inception":
            inception = models.inception_v3(
                weights=weight_if_pretrained_tv(models.Inception_V3_Weights.IMAGENET1K_V1),
                aux_logits=False
            )
            in_feats = inception.fc.in_features
            inception.fc = nn.Identity()
            return inception, in_feats
        elif arch == "mobilenet":
            mbnet = models.mobilenet_v2(
                weights=weight_if_pretrained_tv(models.MobileNet_V2_Weights.IMAGENET1K_V1)
            )
            in_feats = mbnet.classifier[1].in_features
            mbnet.classifier = nn.Identity()
            return mbnet, in_feats
        elif arch == "efficientnet_b0":
            effnet = models.efficientnet_b0(
                weights=weight_if_pretrained_tv(models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            )
            in_feats = effnet.classifier[1].in_features
            effnet.classifier = nn.Identity()
            return effnet, in_feats
        elif arch == "convnext_tiny":
            convnext = models.convnext_tiny(
                weights=weight_if_pretrained_tv(models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            )
            in_feats = convnext.classifier[2].in_features
            convnext.classifier[2] = nn.Identity()
            return convnext, in_feats
        elif arch == "convnext_large":
            convnext = models.convnext_large(
                weights=weight_if_pretrained_tv(models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
            )
            in_feats = convnext.classifier[2].in_features
            convnext.classifier[2] = nn.Identity()
            return convnext, in_feats
        elif arch == "convnext_base":
            if not hasattr(models, "convnext_base"):
                raise ValueError("convnext_base not available in your TorchVision version.")
            convnext = models.convnext_base(
                weights=weight_if_pretrained_tv(models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            )
            in_feats = convnext.classifier[2].in_features
            convnext.classifier[2] = nn.Identity()
            return convnext, in_feats
        elif arch == "convnext_extralarge":
            # Official name in TorchVision is "convnext_xlarge"
            if not hasattr(models, "convnext_xlarge"):
                raise ValueError("convnext_xlarge not available in your TorchVision version.")
            convnext = models.convnext_xlarge(
                weights=weight_if_pretrained_tv(models.ConvNeXt_XLarge_Weights.IMAGENET1K_V1)
            )
            in_feats = convnext.classifier[2].in_features
            convnext.classifier[2] = nn.Identity()
            return convnext, in_feats

        # For convnext_v2 or coatnet or others from timm
        else:
            if not HAVE_TIMM:
                raise ValueError(f"Architecture '{architecture}' requires timm. Please install timm.")
            if arch == "convnext_v2":
                # Example timm usage
                model_name = "convnextv2_base.fcmae_ft_in1k"
                m = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                in_feats = m.num_features
                return m, in_feats
            elif arch.startswith("coatnet_"):
                model_name = arch  # e.g. "coatnet_0", "coatnet_1", etc.
                m = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                in_feats = m.num_features
                return m, in_feats
            else:
                raise ValueError(f"Unknown architecture: {architecture}. Not in TorchVision or timm.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.custom_model is not None:
            return self.custom_model(x)
        feats = self.backbone(x)
        return self.head(feats)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing (if available) in the backbone."""
        if self.custom_model is not None:
            print("[WARN] Skipping gradient checkpointing for custom model.")
            return

        def recurse_gc(module: nn.Module):
            for _, child in module.named_children():
                if hasattr(child, "gradient_checkpointing_enable"):
                    child.gradient_checkpointing_enable()
                recurse_gc(child)
        recurse_gc(self.backbone)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        MixUp / CutMix are handled here so we can do multi-worker data loading on Windows without crashing.
        """
        x, y = batch
        use_mixup = getattr(self.hparams, "use_mixup", False)
        use_cutmix = getattr(self.hparams, "use_cutmix", False)
        alpha = getattr(self.hparams, "mix_alpha", 1.0)

        if use_mixup or use_cutmix:
            if use_mixup and use_cutmix:
                # Randomly pick one
                if random.random() < 0.5:
                    x, y_a, y_b, lam = mixup_data(x, y, alpha=alpha)
                else:
                    x, y_a, y_b, lam = cutmix_data(x, y, alpha=alpha)
            elif use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=alpha)
            else:
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=alpha)

            logits = self(x)
            if (self.hparams.loss_function == "bce_single_logit"
                and self.num_classes == 1
                and self.loss_fn is not None):
                y_a_f = y_a.float().unsqueeze(1)
                y_b_f = y_b.float().unsqueeze(1)
                loss_a = self.loss_fn(logits, y_a_f)
                loss_b = self.loss_fn(logits, y_b_f)
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
            if (self.hparams.loss_function == "bce_single_logit"
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
        x, y = batch
        logits = self(x)
        if (self.hparams.loss_function == "bce_single_logit"
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
        x, y = batch
        logits = self._forward_tta(x) if self.enable_tta else self(x)
        if (self.hparams.loss_function == "bce_single_logit"
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
        """Simple TTA: average predictions of [original, h-flip, v-flip]."""
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

        opt_name = self.optimizer_name.lower()
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
        elif opt_name == "lamb":
            if not HAVE_LAMB:
                raise ImportError("LAMB optimizer not available. Install pytorch-optimizer.")
            optimizer = LambClass(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt_name == "lion":
            if not HAVE_LION:
                raise ImportError("Lion optimizer not available. Install pytorch-optimizer.")
            optimizer = LionClass(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
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


def apply_partial_freeze(model: MaidClassifier, freeze_config: Dict[str, bool]):
    if model.custom_model is not None:
        print("[INFO] Skipping partial freeze because a custom model was loaded.")
        return

    if not model.backbone:
        return

    if freeze_config.get("freeze_entire_backbone", False):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("[INFO] Froze the entire backbone.")
        return

    arch = model.hparams["architecture"].lower()

    # For ResNet
    if arch.startswith("resnet"):
        if freeze_config.get("conv1_bn1", False):
            if hasattr(model.backbone, "conv1"):
                for p in model.backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(model.backbone, "bn1"):
                for p in model.backbone.bn1.parameters():
                    p.requires_grad = False
            print("[INFO] Froze conv1 + bn1 (ResNet).")

        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            if freeze_config.get(layer_name, False):
                layer_mod = getattr(model.backbone, layer_name, None)
                if layer_mod is not None:
                    for p in layer_mod.parameters():
                        p.requires_grad = False
                    print(f"[INFO] Froze {layer_name} (ResNet).")

    # For ConvNeXt
    elif arch.startswith("convnext"):
        if hasattr(model.backbone, "features"):
            feats = model.backbone.features
            if freeze_config.get("block0", False) and len(feats) > 0:
                for p in feats[0].parameters():
                    p.requires_grad = False
                print("[INFO] Froze block0 (ConvNeXt).")
            if freeze_config.get("block1", False) and len(feats) > 1:
                for p in feats[1].parameters():
                    p.requires_grad = False
                print("[INFO] Froze block1 (ConvNeXt).")
            if freeze_config.get("block2", False) and len(feats) > 2:
                for p in feats[2].parameters():
                    p.requires_grad = False
                print("[INFO] Froze block2 (ConvNeXt).")
            if freeze_config.get("block3", False) and len(feats) > 3:
                for p in feats[3].parameters():
                    p.requires_grad = False
                print("[INFO] Froze block3 (ConvNeXt).")

    # For CoAtNet in timm or other partial freeze
    elif arch.startswith("coatnet_"):
        pass


# ============================
# 6) TRAINING CALLBACKS
# ============================
class ProgressCallback(Callback):
    """
    Logs progress each epoch, optionally runs GC,
    and optionally profiles memory usage.
    """
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
        if current_epoch > 0:
            time_per_epoch = elapsed / current_epoch
            eta = time_per_epoch * epochs_left
        else:
            eta = 0.0

        train_loss = trainer.callback_metrics.get("train_loss", "N/A")
        val_loss = trainer.callback_metrics.get("val_loss", "N/A")
        val_acc = trainer.callback_metrics.get("val_acc", "N/A")
        msg = (
            f"[EPOCH {current_epoch}/{self.total_epochs}] "
            f"ETA: {eta:.1f}s | train_loss={train_loss}, val_loss={val_loss}, val_acc={val_acc}"
        )
        print(msg)

        if self.run_gc:
            gc.collect()
            torch.cuda.empty_cache()

        if self.profile_memory and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            print(f"[MEM] GPU allocated: {allocated:.2f} MB")


# ============================
# 7) MAIN TRAINING FUNCTION
# ============================
def run_training_once(
    data_params: DataConfig,
    train_params: TrainConfig
) -> Tuple[str, Dict[str, Any], Optional[float]]:
    seed_everything(42, workers=True)

    print("[INFO] Gathering samples...")
    samples, class_names = gather_samples_and_classes(data_params.root_dir)
    n_total = len(samples)
    if n_total < 2:
        raise ValueError("Dataset has insufficient images.")

    print(f"[INFO] Found {n_total} images among {len(class_names)} classes: {class_names}")

    # Split into train/val/test
    targets = [lbl for _, lbl in samples]
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=data_params.test_split, random_state=42)
    trainval_index, test_index = next(sss_test.split(np.arange(n_total), targets))
    trainval_samples = [samples[i] for i in trainval_index]
    trainval_targets = [targets[i] for i in trainval_index]

    if data_params.val_split > 0:
        val_ratio = data_params.val_split
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio / (1.0 - data_params.test_split),
            random_state=42
        )
        train_index, val_index = next(sss_val.split(trainval_samples, trainval_targets))
    else:
        train_index = range(len(trainval_samples))
        val_index = []

    train_samples = [trainval_samples[i] for i in train_index]
    val_samples = [trainval_samples[i] for i in val_index]
    test_samples = [samples[i] for i in test_index]

    print(f"[INFO] Splits => Train: {len(train_samples)}, "
          f"Val: {len(val_samples)}, Test: {len(test_samples)}")

    num_classes = len(class_names)
    if train_params.loss_function == "bce_single_logit":
        model_num_classes = 1
    else:
        model_num_classes = num_classes

    # Weighted loss?
    class_weights = None
    if train_params.use_weighted_loss and train_params.loss_function != "bce_single_logit":
        all_labels = [lbl for _, lbl in samples]
        freq = [all_labels.count(i) for i in range(num_classes)]
        class_weights = [1.0 / c if c > 0 else 0.0 for c in freq]
        print(f"[INFO] Using weighted loss: {class_weights}")

    # Build model
    model = MaidClassifier(
        architecture=train_params.architecture,
        num_classes=model_num_classes,
        lr=train_params.lr,
        momentum=train_params.momentum,
        weight_decay=train_params.weight_decay,
        use_weighted_loss=train_params.use_weighted_loss,
        class_weights=class_weights,
        optimizer_name=train_params.optimizer_name,
        scheduler_name=train_params.scheduler_name,
        scheduler_params=train_params.scheduler_params,
        dropout_rate=train_params.dropout_rate,
        label_smoothing=train_params.label_smoothing,
        freeze_backbone=train_params.freeze_backbone,
        loss_function=train_params.loss_function,
        pretrained=train_params.pretrained_weights,
        enable_tta=train_params.enable_tta,
        load_custom_model=train_params.load_custom_model,
        custom_model_path=train_params.custom_model_path,
        custom_architecture_file=train_params.custom_architecture_file
    )
    model.class_names = class_names
    apply_partial_freeze(model, train_params.freeze_config)

    if train_params.enable_gradient_checkpointing:
        try:
            model.enable_gradient_checkpointing()
        except Exception as e:
            print(f"[WARN] Could not enable gradient checkpointing: {e}")

    # Callbacks
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

    logger_obj = None
    tb_log_dir = None
    if train_params.use_tensorboard:
        logger_obj = TensorBoardLogger(save_dir="tb_logs", name="experiment")
        tb_log_dir = logger_obj.log_dir

    trainer_device = "gpu" if torch.cuda.is_available() else "cpu"
    devices_to_use = torch.cuda.device_count() if torch.cuda.is_available() else 1

    trainer = Trainer(
        max_epochs=train_params.max_epochs,
        accelerator=trainer_device,
        devices=devices_to_use,
        callbacks=callbacks,
        logger=logger_obj,
        gradient_clip_val=train_params.gradient_clip_val,
        precision=16 if (train_params.use_mixed_precision and torch.cuda.is_available()) else 32,
        accumulate_grad_batches=train_params.accumulate_grad_batches if train_params.enable_grad_accum else 1,
        check_val_every_n_epoch=train_params.check_val_every_n_epoch,
        enable_progress_bar=True
    )

    # Decide final image size
    if train_params.architecture.lower() == "inception" and train_params.use_inception_299:
        final_crop_dim = 299
        bigger_resize = 320
    else:
        final_crop_dim = 224
        bigger_resize = 256

    # Albumentations transforms
    aug_list = []
    if train_params.use_crop:
        aug_list.append(
            A.RandomResizedCrop(
                size=(final_crop_dim, final_crop_dim),
                scale=(train_params.random_crop_scale_min, train_params.random_crop_scale_max),
                ratio=(0.75, 1.3333),
                interpolation=cv2.INTER_LINEAR,
                p=train_params.random_crop_prob
            )
        )
    else:
        aug_list.append(
            A.Resize(
                height=final_crop_dim,
                width=final_crop_dim,
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            )
        )

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
                    A.VerticalFlip(p=1.0),
                ], p=train_params.flip_prob)
            )

    if train_params.use_elastic:
        aug_list.append(A.ElasticTransform(p=0.5))
    if train_params.use_grid_distortion:
        aug_list.append(A.GridDistortion(p=0.5))
    if train_params.use_optical_distortion:
        aug_list.append(A.OpticalDistortion(p=0.5))

    # Advanced augs
    if train_params.random_gamma:
        aug_list.append(
            A.RandomGamma(
                gamma_limit=(train_params.random_gamma_limit_low, train_params.random_gamma_limit_high),
                p=train_params.random_gamma_prob
            )
        )
    if train_params.clahe:
        aug_list.append(
            A.CLAHE(
                clip_limit=train_params.clahe_clip_limit,
                tile_grid_size=(train_params.clahe_tile_size, train_params.clahe_tile_size),
                p=train_params.clahe_prob
            )
        )
    if train_params.channel_shuffle:
        aug_list.append(A.ChannelShuffle(p=train_params.channel_shuffle_prob))

    if train_params.use_posterize_solarize_equalize:
        aug_list.append(
            A.OneOf([
                A.Posterize(num_bits=train_params.posterize_bits, p=1.0),
                A.Solarize(threshold=train_params.solarize_threshold, p=1.0),
                A.Equalize(p=1.0),
            ], p=train_params.pse_prob)
        )

    if train_params.sharpen_denoise:
        aug_list.append(
            A.Sharpen(
                alpha=(train_params.sharpen_alpha_min, train_params.sharpen_alpha_max),
                lightness=(train_params.sharpen_lightness_min, train_params.sharpen_lightness_max),
                p=train_params.sharpen_prob
            )
        )

    if train_params.gauss_vs_mult_noise:
        aug_list.append(
            A.OneOf([
                A.GaussNoise(
                    var_limit=(train_params.gauss_noise_var_limit_low, train_params.gauss_noise_var_limit_high),
                    p=1.0
                ),
                A.MultiplicativeNoise(
                    multiplier=(train_params.mult_noise_lower, train_params.mult_noise_upper),
                    p=1.0
                ),
            ], p=train_params.gauss_mult_prob)
        )

    if train_params.cutout_coarse_dropout:
        aug_list.append(
            A.CoarseDropout(
                max_holes=train_params.cutout_max_holes,
                max_height=train_params.cutout_max_height,
                max_width=train_params.cutout_max_width,
                fill_value=0,
                p=train_params.cutout_prob
            )
        )

    if train_params.use_shift_scale_rotate:
        aug_list.append(
            A.ShiftScaleRotate(
                shift_limit=train_params.ssr_shift_limit,
                scale_limit=train_params.ssr_scale_limit,
                rotate_limit=train_params.ssr_rotate_limit,
                p=train_params.ssr_prob
            )
        )

    if train_params.use_one_of_advanced_transforms:
        adv_candidates = [
            A.RandomFog(p=1.0),
            A.RandomRain(p=1.0),
            A.GaussianBlur(p=1.0),
        ]
        aug_list.append(A.OneOf(adv_candidates, p=train_params.one_of_advanced_transforms_prob))

    # RandAugment if CoAtNet + user requests
    if train_params.architecture.lower().startswith("coatnet_") and train_params.use_randaugment:
        if hasattr(A, "RandAugment"):
            aug_list.append(A.RandAugment(num_ops=2, magnitude=9, p=1.0))
        else:
            print("[WARN] Albumentations RandAugment is not available. Please update Albumentations.")

    if train_params.normalize_pixel_intensity:
        aug_list.append(A.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225), p=1.0))

    aug_list.append(ToTensorV2())
    train_transform = A.Compose(aug_list)

    # Validation transform
    val_augs = []
    val_augs.append(
        A.Resize(
            height=bigger_resize,
            width=bigger_resize,
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        )
    )
    if train_params.val_center_crop:
        val_augs.append(A.CenterCrop(height=final_crop_dim, width=final_crop_dim))
    if train_params.normalize_pixel_intensity:
        val_augs.append(A.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225)))
    val_augs.append(ToTensorV2())
    val_transform = A.Compose(val_augs)

    train_ds = AlbumentationsDataset(
        train_samples, transform=train_transform,
        classes=class_names, allow_grayscale=data_params.allow_grayscale
    )
    val_ds = AlbumentationsDataset(
        val_samples, transform=val_transform,
        classes=class_names, allow_grayscale=data_params.allow_grayscale
    )
    test_ds = AlbumentationsDataset(
        test_samples, transform=val_transform,
        classes=class_names, allow_grayscale=data_params.allow_grayscale
    )

    # Determine if we can enable persistent_workers (requires num_workers>0)
    persistent_workers = train_params.persistent_workers and (train_params.num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_params.batch_size,
        shuffle=True,
        num_workers=train_params.num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_params.batch_size,
        shuffle=False,
        num_workers=train_params.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_params.batch_size,
        shuffle=False,
        num_workers=train_params.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    # LR finder if requested
    if train_params.use_lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        suggested_lr = lr_finder.suggestion()
        print(f"[LR FINDER] Suggested LR: {suggested_lr}")
        if train_params.accept_lr_suggestion:
            model.lr = suggested_lr
            print(f"[LR FINDER] Updated model.lr to {suggested_lr}")

    # Train
    trainer.fit(model, train_loader, val_loader)

    # === LOAD BEST CHECKPOINT AND USE FOR FINAL VAL/TEST ===
    best_checkpoint_path = ckpt_callback.best_model_path
    print(f"[INFO] Validating & testing best checkpoint from: {best_checkpoint_path}")
    best_model = MaidClassifier.load_from_checkpoint(best_checkpoint_path)

    # Copy class_names and relevant hparams to best_model so it can do TTA or other steps
    best_model.class_names = model.class_names
    best_model.hparams.enable_tta = model.hparams.enable_tta

    # Evaluate on validation set
    val_results = trainer.validate(best_model, val_loader, verbose=False)
    final_val_loss = None
    if len(val_results) > 0 and "val_loss" in val_results[0]:
        final_val_loss = float(val_results[0]["val_loss"])

    # Evaluate on test set
    trainer.test(best_model, test_loader, verbose=False)
    test_metrics_dict = {
        "test_results": trainer.callback_metrics,
        "confusion_matrix": None,
        "class_report": None,
        "auc_roc": None,
        "tb_log_dir": tb_log_dir,
        "cm_fig_path": None
    }

    # Gather predictions
    best_model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(best_model.device)
            yb = yb.to(best_model.device)
            logits = best_model._forward_tta(xb) if best_model.hparams.enable_tta else best_model(xb)

            if model_num_classes == 1 and train_params.loss_function == "bce_single_logit":
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().view(-1)
            else:
                preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    cm = confusion_matrix(all_targets, all_preds)
    test_metrics_dict["confusion_matrix"] = cm.tolist()

    cr = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    test_metrics_dict["class_report"] = cr

    if HAVE_MPL:
        fig_cm = plt.figure(figsize=(6, 6))
        ax = fig_cm.add_subplot(1, 1, 1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig_path = "confusion_matrix.png"
        fig_cm.savefig(fig_path)
        plt.close(fig_cm)
        test_metrics_dict["cm_fig_path"] = os.path.abspath(fig_path)

    print(f"[INFO] Finished training. Best checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path, test_metrics_dict, final_val_loss


# ============================
# 8) WORKER FUNCTIONS FOR MULTIPROCESS
# ============================

def _try_run_training(data_dict, train_dict):
    """
    Helper to run training once. Catches OOM error, returns results or raises.
    """
    data_config = DataConfig(**data_dict)
    train_config = TrainConfig(**train_dict)
    best_ckpt_path, test_metrics, val_loss = run_training_once(data_config, train_config)
    return best_ckpt_path, test_metrics


def _auto_adjust_batch_size(data_dict, train_dict, original_bs) -> (Optional[str], Optional[dict]):
    """
    Binary-search fallback for GPU OOM. Returns (ckpt_path, test_metrics) or (None, None) if fails.
    """
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
            low = mid + 1  # try bigger
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
    """
    This function runs in a child process to do training.
    We pass results back via return_dict from a multiprocessing.Manager() dict.
    """
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
                print(f"[ERROR] Training crashed: {re}\n{traceback.format_exc()}")
                return_dict["status"] = "ERROR"
                return_dict["ckpt_path"] = ""
                return_dict["test_metrics"] = {}
    except Exception as e:
        print(f"[ERROR] Training crashed: {e}\n{traceback.format_exc()}")
        return_dict["status"] = "ERROR"
        return_dict["ckpt_path"] = ""
        return_dict["test_metrics"] = {}


def optuna_worker(data_params, base_train_params, optuna_n_trials, optuna_timeout, use_test_metric_for_optuna, return_dict):
    """
    This function runs in a child process for Optuna hyperparam tuning.
    """
    try:
        seed_everything(42, workers=True)
        # We'll define an objective that modifies a copy of base_train_params
        def objective(trial: optuna.Trial) -> float:
            trial_train_params = dict(base_train_params)
            lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
            dropout = trial.suggest_float("dropout_rate", 0.0, 0.7)
            optimizer_name = trial.suggest_categorical("optimizer_name", ["adam", "sgd", "adamw", "lamb", "lion"])
            scheduler_name = trial.suggest_categorical("scheduler_name", ["none", "steplr", "cosineannealing"])

            trial_train_params["lr"] = lr
            trial_train_params["dropout_rate"] = dropout
            trial_train_params["optimizer_name"] = optimizer_name
            trial_train_params["scheduler_name"] = scheduler_name

            print(f"[OPTUNA] Trial {trial.number}: lr={lr}, dropout={dropout}, "
                  f"opt={optimizer_name}, sch={scheduler_name}")

            # Attempt the run
            try:
                bcp, tmetrics = _try_run_training(data_params, trial_train_params)
            except RuntimeError as re:
                # If OOM, we can either fail or attempt fallback; for simplicity, fail the trial
                if "CUDA out of memory" in str(re):
                    print("[OPTUNA] OOM encountered. Setting trial value = 9999.")
                    return 9999.0
                else:
                    raise

            if not tmetrics:
                return 9999.0

            # We check val_loss or test_loss
            if use_test_metric_for_optuna:
                # check test_loss
                if ("test_results" in tmetrics
                    and len(tmetrics["test_results"]) > 0
                    and "test_loss" in tmetrics["test_results"][0]):
                    return float(tmetrics["test_results"][0]["test_loss"])
                else:
                    return 9999.0
            else:
                # We didn't store val_loss here, but run_training_once returns a val_loss
                _, _, v_loss = run_training_once(DataConfig(**data_params), TrainConfig(**trial_train_params))
                if v_loss is None:
                    return 9999.0
                return v_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=optuna_n_trials,
            timeout=optuna_timeout if optuna_timeout > 0 else None
        )

        best_trial = study.best_trial
        print(f"[OPTUNA] Best trial: {best_trial.number}, value={best_trial.value}")
        print(f"[OPTUNA] Params: {best_trial.params}")

        # Re-train final model with best hyperparams
        best_params = dict(base_train_params)
        best_params["lr"] = best_trial.params["lr"]
        best_params["dropout_rate"] = best_trial.params["dropout_rate"]
        best_params["optimizer_name"] = best_trial.params["optimizer_name"]
        best_params["scheduler_name"] = best_trial.params["scheduler_name"]

        print("[OPTUNA] Re-training final model with best hyperparams...")
        bcp, tmetrics = _try_run_training(data_params, best_params)

        return_dict["status"] = "OK"
        return_dict["ckpt_path"] = bcp
        return_dict["test_metrics"] = tmetrics

    except Exception as e:
        print(f"[ERROR] Optuna tuning crashed: {e}\n{traceback.format_exc()}")
        return_dict["status"] = "ERROR"
        return_dict["ckpt_path"] = ""
        return_dict["test_metrics"] = {}


# ===============================
# 9) MAIN PLUGIN CLASS (the GUI)
# ===============================
class Plugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.plugin_name = "Training"

        self.widget_main: Optional[QWidget] = None
        self.progress_bar: Optional[QProgressBar] = None

        # --- Data / split inputs
        self.train_data_dir_edit: Optional[QLineEdit] = None
        self.val_split_spin: Optional[QDoubleSpinBox] = None
        self.test_split_spin: Optional[QDoubleSpinBox] = None
        self.cb_allow_grayscale: Optional[QCheckBox] = None

        # --- Architecture + Loss
        self.arch_combo: Optional[QComboBox] = None
        self.weighted_loss_cb: Optional[QCheckBox] = None
        self.cb_normalize_pixel_intensity: Optional[QCheckBox] = None
        self.cb_inception_299: Optional[QCheckBox] = None
        self.loss_combo: Optional[QComboBox] = None

        # --- Basic Augs
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

        # --- Hyperparams
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

        # NEW: persistent workers
        self.cb_persistent_workers: Optional[QCheckBox] = None

        # --- Partial freeze
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

        # --- Custom model
        self.cb_load_custom_model: Optional[QCheckBox] = None
        self.custom_model_path_edit: Optional[QLineEdit] = None
        self.custom_arch_file_edit: Optional[QLineEdit] = None

        # --- Buttons
        self.btn_train: Optional[QPushButton] = None
        self.btn_export_results: Optional[QPushButton] = None
        self.btn_tune_optuna: Optional[QPushButton] = None
        self.cb_optuna_use_test_metric: Optional[QCheckBox] = None
        self.optuna_trials_spin: Optional[QSpinBox] = None
        self.optuna_timeout_spin: Optional[QSpinBox] = None

        # Advanced
        self.cb_random_gamma: Optional[QCheckBox] = None
        self.spin_gamma_low: Optional[QDoubleSpinBox] = None
        self.spin_gamma_high: Optional[QDoubleSpinBox] = None
        self.spin_gamma_prob: Optional[QDoubleSpinBox] = None

        self.cb_clahe: Optional[QCheckBox] = None
        self.spin_clahe_clip: Optional[QDoubleSpinBox] = None
        self.spin_clahe_tile: Optional[QSpinBox] = None
        self.spin_clahe_prob: Optional[QDoubleSpinBox] = None

        self.cb_channel_shuffle: Optional[QCheckBox] = None
        self.spin_channel_shuffle_prob: Optional[QDoubleSpinBox] = None

        self.cb_posterize_solarize_equalize: Optional[QCheckBox] = None
        self.spin_pse_prob: Optional[QDoubleSpinBox] = None
        self.spin_posterize_bits: Optional[QSpinBox] = None
        self.spin_solarize_threshold: Optional[QSpinBox] = None

        self.cb_sharpen_denoise: Optional[QCheckBox] = None
        self.spin_sharpen_prob: Optional[QDoubleSpinBox] = None
        self.spin_sharpen_alpha_min: Optional[QDoubleSpinBox] = None
        self.spin_sharpen_alpha_max: Optional[QDoubleSpinBox] = None
        self.spin_sharpen_lightness_min: Optional[QDoubleSpinBox] = None
        self.spin_sharpen_lightness_max: Optional[QDoubleSpinBox] = None

        self.cb_gauss_vs_mult_noise: Optional[QCheckBox] = None
        self.spin_gauss_mult_prob: Optional[QDoubleSpinBox] = None
        self.spin_gauss_var_low: Optional[QDoubleSpinBox] = None
        self.spin_gauss_var_high: Optional[QDoubleSpinBox] = None
        self.spin_mult_lower: Optional[QDoubleSpinBox] = None
        self.spin_mult_upper: Optional[QDoubleSpinBox] = None

        self.cb_cutout_coarse_dropout: Optional[QCheckBox] = None
        self.spin_cutout_max_holes: Optional[QSpinBox] = None
        self.spin_cutout_max_height: Optional[QSpinBox] = None
        self.spin_cutout_max_width: Optional[QSpinBox] = None
        self.spin_cutout_prob: Optional[QDoubleSpinBox] = None

        self.cb_shift_scale_rotate: Optional[QCheckBox] = None
        self.spin_ssr_shift_limit: Optional[QDoubleSpinBox] = None
        self.spin_ssr_scale_limit: Optional[QDoubleSpinBox] = None
        self.spin_ssr_rotate_limit: Optional[QSpinBox] = None
        self.spin_ssr_prob: Optional[QDoubleSpinBox] = None

        self.cb_use_one_of_advanced_transforms: Optional[QCheckBox] = None
        self.spin_one_of_adv_prob: Optional[QDoubleSpinBox] = None

        # RandAugment
        self.cb_randaugment: Optional[QCheckBox] = None

        self.best_ckpt_path: Optional[str] = None
        self.last_test_metrics: Optional[Dict[str, Any]] = None

        # Multiprocessing variables
        self.train_process: Optional[multiprocessing.Process] = None
        self.train_result_manager = None
        self.train_result_dict = None

        self.optuna_process: Optional[multiprocessing.Process] = None
        self.optuna_result_manager = None
        self.optuna_result_dict = None

        self.training_timer: Optional[QTimer] = None
        self.optuna_timer: Optional[QTimer] = None

    def create_tab(self) -> QWidget:
        """
        Build the entire GUI layout (scrollable) with all group boxes.
        """
        self.widget_main = QWidget()
        main_layout = QVBoxLayout(self.widget_main)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

        # ========== DATA GROUP ==========
        group_data = QGroupBox("Dataset & Splits")
        data_layout = QVBoxLayout(group_data)

        h_data = QHBoxLayout()
        self.train_data_dir_edit = QLineEdit()
        btn_browse_data = QPushButton("Browse Data Folder...")
        btn_browse_data.clicked.connect(self.browse_dataset_folder)
        h_data.addWidget(QLabel("Dataset Folder:"))
        h_data.addWidget(self.train_data_dir_edit)
        h_data.addWidget(btn_browse_data)
        data_layout.addLayout(h_data)

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
        data_layout.addLayout(h_splits)

        self.cb_allow_grayscale = QCheckBox("Allow Grayscale Images")
        data_layout.addWidget(self.cb_allow_grayscale)

        container_layout.addWidget(group_data)

        # ========== ARCH GROUP ==========
        group_arch = QGroupBox("Architecture & Basic Options")
        arch_layout = QVBoxLayout(group_arch)

        h_arch = QHBoxLayout()
        self.arch_combo = QComboBox()
        self.arch_combo.addItems([
            "resnet18", "resnet50", "resnet101",
            "densenet", "vgg", "inception", "mobilenet",
            "efficientnet_b0",
            "convnext_tiny", "convnext_large", "convnext_base", "convnext_extralarge",
            "convnext_v2",
            "coatnet_0", "coatnet_1", "coatnet_2", "coatnet_3", "coatnet_4"
        ])
        self.weighted_loss_cb = QCheckBox("Weighted Loss")
        self.cb_normalize_pixel_intensity = QCheckBox("Normalize Pixel")
        self.cb_inception_299 = QCheckBox("Use 299 for Inception")

        h_arch.addWidget(QLabel("Architecture:"))
        h_arch.addWidget(self.arch_combo)
        h_arch.addWidget(self.weighted_loss_cb)
        h_arch.addWidget(self.cb_normalize_pixel_intensity)
        h_arch.addWidget(self.cb_inception_299)
        arch_layout.addLayout(h_arch)

        h_loss = QHBoxLayout()
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["cross_entropy", "focal", "bce", "bce_single_logit"])
        h_loss.addWidget(QLabel("Loss:"))
        h_loss.addWidget(self.loss_combo)
        arch_layout.addLayout(h_loss)

        container_layout.addWidget(group_arch)

        # ========== AUGS (BASIC) GROUP ==========
        group_basic_augs = QGroupBox("Basic Augmentations")
        basic_augs_layout = QVBoxLayout(group_basic_augs)

        row1 = QHBoxLayout()
        self.cb_bright_contrast = QCheckBox("Brightness/Contrast")
        self.cb_hue_sat = QCheckBox("Hue/Sat")
        self.cb_gauss_noise = QCheckBox("GaussNoise")
        self.cb_rotation = QCheckBox("Rotation")
        row1.addWidget(self.cb_bright_contrast)
        row1.addWidget(self.cb_hue_sat)
        row1.addWidget(self.cb_gauss_noise)
        row1.addWidget(self.cb_rotation)
        basic_augs_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.cb_flip = QCheckBox("Flipping")
        self.flip_mode_combo = QComboBox()
        self.flip_mode_combo.addItems(["horizontal", "vertical", "both"])
        self.flip_prob_spin = QDoubleSpinBox()
        self.flip_prob_spin.setRange(0.0, 1.0)
        self.flip_prob_spin.setValue(0.5)
        self.cb_crop = QCheckBox("Random Crop")
        self.cb_elastic = QCheckBox("Elastic")
        row2.addWidget(self.cb_flip)
        row2.addWidget(QLabel("Mode:"))
        row2.addWidget(self.flip_mode_combo)
        row2.addWidget(QLabel("Prob:"))
        row2.addWidget(self.flip_prob_spin)
        row2.addWidget(self.cb_crop)
        row2.addWidget(self.cb_elastic)
        basic_augs_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Random Crop p:"))
        self.random_crop_prob_spin = QDoubleSpinBox()
        self.random_crop_prob_spin.setRange(0.0, 1.0)
        self.random_crop_prob_spin.setValue(1.0)
        row3.addWidget(self.random_crop_prob_spin)
        row3.addWidget(QLabel("Scale Min:"))
        self.random_crop_scale_min_spin = QDoubleSpinBox()
        self.random_crop_scale_min_spin.setRange(0.0, 1.0)
        self.random_crop_scale_min_spin.setValue(0.8)
        row3.addWidget(self.random_crop_scale_min_spin)
        row3.addWidget(QLabel("Scale Max:"))
        self.random_crop_scale_max_spin = QDoubleSpinBox()
        self.random_crop_scale_max_spin.setRange(0.0, 1.0)
        self.random_crop_scale_max_spin.setValue(1.0)
        row3.addWidget(self.random_crop_scale_max_spin)
        basic_augs_layout.addLayout(row3)

        row4 = QHBoxLayout()
        self.cb_grid_distortion = QCheckBox("GridDistortion")
        self.cb_optical_distortion = QCheckBox("OpticalDistortion")
        self.cb_mixup = QCheckBox("MixUp")
        self.cb_cutmix = QCheckBox("CutMix")
        self.mix_alpha_spin = QDoubleSpinBox()
        self.mix_alpha_spin.setRange(0.0, 5.0)
        self.mix_alpha_spin.setValue(1.0)
        row4.addWidget(self.cb_grid_distortion)
        row4.addWidget(self.cb_optical_distortion)
        row4.addWidget(self.cb_mixup)
        row4.addWidget(self.cb_cutmix)
        row4.addWidget(QLabel("alpha:"))
        row4.addWidget(self.mix_alpha_spin)
        basic_augs_layout.addLayout(row4)

        container_layout.addWidget(group_basic_augs)

        # ========== AUGS (ADVANCED) GROUP ==========
        group_adv_augs = QGroupBox("Advanced Augmentations")
        adv_layout = QVBoxLayout(group_adv_augs)

        # RandomGamma
        rand_gamma_layout = QHBoxLayout()
        self.cb_random_gamma = QCheckBox("RandomGamma")
        self.spin_gamma_low = QDoubleSpinBox()
        self.spin_gamma_low.setRange(1.0, 300.0)
        self.spin_gamma_low.setValue(80.0)
        self.spin_gamma_high = QDoubleSpinBox()
        self.spin_gamma_high.setRange(1.0, 300.0)
        self.spin_gamma_high.setValue(120.0)
        self.spin_gamma_prob = QDoubleSpinBox()
        self.spin_gamma_prob.setRange(0.0, 1.0)
        self.spin_gamma_prob.setValue(0.5)

        rand_gamma_layout.addWidget(self.cb_random_gamma)
        rand_gamma_layout.addWidget(QLabel("GammaLow:"))
        rand_gamma_layout.addWidget(self.spin_gamma_low)
        rand_gamma_layout.addWidget(QLabel("GammaHigh:"))
        rand_gamma_layout.addWidget(self.spin_gamma_high)
        rand_gamma_layout.addWidget(QLabel("Prob:"))
        rand_gamma_layout.addWidget(self.spin_gamma_prob)
        adv_layout.addLayout(rand_gamma_layout)

        # CLAHE
        clahe_layout = QHBoxLayout()
        self.cb_clahe = QCheckBox("CLAHE")
        self.spin_clahe_clip = QDoubleSpinBox()
        self.spin_clahe_clip.setRange(1.0, 50.0)
        self.spin_clahe_clip.setValue(4.0)
        self.spin_clahe_tile = QSpinBox()
        self.spin_clahe_tile.setRange(1, 64)
        self.spin_clahe_tile.setValue(8)
        self.spin_clahe_prob = QDoubleSpinBox()
        self.spin_clahe_prob.setRange(0.0, 1.0)
        self.spin_clahe_prob.setValue(0.3)
        clahe_layout.addWidget(self.cb_clahe)
        clahe_layout.addWidget(QLabel("ClipLimit:"))
        clahe_layout.addWidget(self.spin_clahe_clip)
        clahe_layout.addWidget(QLabel("TileSize:"))
        clahe_layout.addWidget(self.spin_clahe_tile)
        clahe_layout.addWidget(QLabel("Prob:"))
        clahe_layout.addWidget(self.spin_clahe_prob)
        adv_layout.addLayout(clahe_layout)

        # ChannelShuffle
        cshuffle_layout = QHBoxLayout()
        self.cb_channel_shuffle = QCheckBox("ChannelShuffle")
        self.spin_channel_shuffle_prob = QDoubleSpinBox()
        self.spin_channel_shuffle_prob.setRange(0.0, 1.0)
        self.spin_channel_shuffle_prob.setValue(0.2)
        cshuffle_layout.addWidget(self.cb_channel_shuffle)
        cshuffle_layout.addWidget(QLabel("Prob:"))
        cshuffle_layout.addWidget(self.spin_channel_shuffle_prob)
        adv_layout.addLayout(cshuffle_layout)

        # Posterize/Solarize/Equalize
        pse_layout = QHBoxLayout()
        self.cb_posterize_solarize_equalize = QCheckBox("Posterize/Solarize/Equalize")
        self.spin_pse_prob = QDoubleSpinBox()
        self.spin_pse_prob.setRange(0.0, 1.0)
        self.spin_pse_prob.setValue(0.2)
        self.spin_posterize_bits = QSpinBox()
        self.spin_posterize_bits.setRange(1, 8)
        self.spin_posterize_bits.setValue(4)
        self.spin_solarize_threshold = QSpinBox()
        self.spin_solarize_threshold.setRange(0, 255)
        self.spin_solarize_threshold.setValue(128)
        pse_layout.addWidget(self.cb_posterize_solarize_equalize)
        pse_layout.addWidget(QLabel("Prob:"))
        pse_layout.addWidget(self.spin_pse_prob)
        pse_layout.addWidget(QLabel("PosterizeBits:"))
        pse_layout.addWidget(self.spin_posterize_bits)
        pse_layout.addWidget(QLabel("SolarizeThr:"))
        pse_layout.addWidget(self.spin_solarize_threshold)
        adv_layout.addLayout(pse_layout)

        # Sharpen
        sharpen_layout = QHBoxLayout()
        self.cb_sharpen_denoise = QCheckBox("Sharpen")
        self.spin_sharpen_prob = QDoubleSpinBox()
        self.spin_sharpen_prob.setRange(0.0, 1.0)
        self.spin_sharpen_prob.setValue(0.3)
        self.spin_sharpen_alpha_min = QDoubleSpinBox()
        self.spin_sharpen_alpha_min.setRange(0.0, 1.0)
        self.spin_sharpen_alpha_min.setValue(0.2)
        self.spin_sharpen_alpha_max = QDoubleSpinBox()
        self.spin_sharpen_alpha_max.setRange(0.0, 1.0)
        self.spin_sharpen_alpha_max.setValue(0.5)
        self.spin_sharpen_lightness_min = QDoubleSpinBox()
        self.spin_sharpen_lightness_min.setRange(0.0, 3.0)
        self.spin_sharpen_lightness_min.setValue(0.5)
        self.spin_sharpen_lightness_max = QDoubleSpinBox()
        self.spin_sharpen_lightness_max.setRange(0.0, 3.0)
        self.spin_sharpen_lightness_max.setValue(1.0)

        sharpen_layout.addWidget(self.cb_sharpen_denoise)
        sharpen_layout.addWidget(QLabel("Prob:"))
        sharpen_layout.addWidget(self.spin_sharpen_prob)
        sharpen_layout.addWidget(QLabel("AlphaMin:"))
        sharpen_layout.addWidget(self.spin_sharpen_alpha_min)
        sharpen_layout.addWidget(QLabel("AlphaMax:"))
        sharpen_layout.addWidget(self.spin_sharpen_alpha_max)
        sharpen_layout.addWidget(QLabel("LightMin:"))
        sharpen_layout.addWidget(self.spin_sharpen_lightness_min)
        sharpen_layout.addWidget(QLabel("LightMax:"))
        sharpen_layout.addWidget(self.spin_sharpen_lightness_max)
        adv_layout.addLayout(sharpen_layout)

        # Gauss vs Mult Noise
        gm_layout = QHBoxLayout()
        self.cb_gauss_vs_mult_noise = QCheckBox("Gauss vs Mult Noise")
        self.spin_gauss_mult_prob = QDoubleSpinBox()
        self.spin_gauss_mult_prob.setRange(0.0, 1.0)
        self.spin_gauss_mult_prob.setValue(0.3)
        self.spin_gauss_var_low = QDoubleSpinBox()
        self.spin_gauss_var_low.setRange(0.0, 1000.0)
        self.spin_gauss_var_low.setValue(10.0)
        self.spin_gauss_var_high = QDoubleSpinBox()
        self.spin_gauss_var_high.setRange(0.0, 1000.0)
        self.spin_gauss_var_high.setValue(50.0)
        self.spin_mult_lower = QDoubleSpinBox()
        self.spin_mult_lower.setRange(0.0, 10.0)
        self.spin_mult_lower.setValue(0.9)
        self.spin_mult_upper = QDoubleSpinBox()
        self.spin_mult_upper.setRange(0.0, 10.0)
        self.spin_mult_upper.setValue(1.1)
        gm_layout.addWidget(self.cb_gauss_vs_mult_noise)
        gm_layout.addWidget(QLabel("Prob:"))
        gm_layout.addWidget(self.spin_gauss_mult_prob)
        gm_layout.addWidget(QLabel("GaussVarLow:"))
        gm_layout.addWidget(self.spin_gauss_var_low)
        gm_layout.addWidget(QLabel("GaussVarHigh:"))
        gm_layout.addWidget(self.spin_gauss_var_high)
        gm_layout.addWidget(QLabel("MultLower:"))
        gm_layout.addWidget(self.spin_mult_lower)
        gm_layout.addWidget(QLabel("MultUpper:"))
        gm_layout.addWidget(self.spin_mult_upper)
        adv_layout.addLayout(gm_layout)

        # Cutout
        cutout_layout = QHBoxLayout()
        self.cb_cutout_coarse_dropout = QCheckBox("Cutout/CoarseDropout")
        self.spin_cutout_max_holes = QSpinBox()
        self.spin_cutout_max_holes.setRange(1, 100)
        self.spin_cutout_max_holes.setValue(8)
        self.spin_cutout_max_height = QSpinBox()
        self.spin_cutout_max_height.setRange(1, 512)
        self.spin_cutout_max_height.setValue(32)
        self.spin_cutout_max_width = QSpinBox()
        self.spin_cutout_max_width.setRange(1, 512)
        self.spin_cutout_max_width.setValue(32)
        self.spin_cutout_prob = QDoubleSpinBox()
        self.spin_cutout_prob.setRange(0.0, 1.0)
        self.spin_cutout_prob.setValue(0.5)
        cutout_layout.addWidget(self.cb_cutout_coarse_dropout)
        cutout_layout.addWidget(QLabel("MaxHoles:"))
        cutout_layout.addWidget(self.spin_cutout_max_holes)
        cutout_layout.addWidget(QLabel("MaxH:"))
        cutout_layout.addWidget(self.spin_cutout_max_height)
        cutout_layout.addWidget(QLabel("MaxW:"))
        cutout_layout.addWidget(self.spin_cutout_max_width)
        cutout_layout.addWidget(QLabel("Prob:"))
        cutout_layout.addWidget(self.spin_cutout_prob)
        adv_layout.addLayout(cutout_layout)

        # ShiftScaleRotate
        ssr_layout = QHBoxLayout()
        self.cb_shift_scale_rotate = QCheckBox("ShiftScaleRotate")
        self.spin_ssr_shift_limit = QDoubleSpinBox()
        self.spin_ssr_shift_limit.setRange(0.0, 1.0)
        self.spin_ssr_shift_limit.setValue(0.1)
        self.spin_ssr_scale_limit = QDoubleSpinBox()
        self.spin_ssr_scale_limit.setRange(0.0, 1.0)
        self.spin_ssr_scale_limit.setValue(0.1)
        self.spin_ssr_rotate_limit = QSpinBox()
        self.spin_ssr_rotate_limit.setRange(0, 180)
        self.spin_ssr_rotate_limit.setValue(15)
        self.spin_ssr_prob = QDoubleSpinBox()
        self.spin_ssr_prob.setRange(0.0, 1.0)
        self.spin_ssr_prob.setValue(0.4)
        ssr_layout.addWidget(self.cb_shift_scale_rotate)
        ssr_layout.addWidget(QLabel("ShiftLimit:"))
        ssr_layout.addWidget(self.spin_ssr_shift_limit)
        ssr_layout.addWidget(QLabel("ScaleLimit:"))
        ssr_layout.addWidget(self.spin_ssr_scale_limit)
        ssr_layout.addWidget(QLabel("RotateLimit:"))
        ssr_layout.addWidget(self.spin_ssr_rotate_limit)
        ssr_layout.addWidget(QLabel("Prob:"))
        ssr_layout.addWidget(self.spin_ssr_prob)
        adv_layout.addLayout(ssr_layout)

        # OneOf advanced transforms
        oneof_layout = QHBoxLayout()
        self.cb_use_one_of_advanced_transforms = QCheckBox("Use OneOf Advanced Transforms?")
        self.spin_one_of_adv_prob = QDoubleSpinBox()
        self.spin_one_of_adv_prob.setRange(0.0, 1.0)
        self.spin_one_of_adv_prob.setValue(0.5)
        oneof_layout.addWidget(self.cb_use_one_of_advanced_transforms)
        oneof_layout.addWidget(QLabel("OneOf Prob:"))
        oneof_layout.addWidget(self.spin_one_of_adv_prob)
        adv_layout.addLayout(oneof_layout)

        # RandAugment for CoAtNet
        self.cb_randaugment = QCheckBox("RandAugment (CoAtNet only)")
        adv_layout.addWidget(self.cb_randaugment)

        container_layout.addWidget(group_adv_augs)

        # ========== HYPERPARAMS GROUP ==========
        group_hparams = QGroupBox("Hyperparameters & Training")
        hp_layout = QVBoxLayout(group_hparams)

        row_hp1 = QHBoxLayout()
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
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd", "adamw", "lamb", "lion"])
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["none", "steplr", "reducelronplateau",
                                       "cosineannealing", "cycliclr"])
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(5)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(8)
        self.cb_early_stopping = QCheckBox("Early Stopping")

        row_hp1.addWidget(QLabel("LR:"))
        row_hp1.addWidget(self.lr_spin)
        row_hp1.addWidget(QLabel("Momentum:"))
        row_hp1.addWidget(self.momentum_spin)
        row_hp1.addWidget(QLabel("WD:"))
        row_hp1.addWidget(self.wd_spin)
        row_hp1.addWidget(QLabel("Opt:"))
        row_hp1.addWidget(self.optimizer_combo)
        row_hp1.addWidget(QLabel("Sched:"))
        row_hp1.addWidget(self.scheduler_combo)
        row_hp1.addWidget(QLabel("Epochs:"))
        row_hp1.addWidget(self.epochs_spin)
        row_hp1.addWidget(QLabel("Batch:"))
        row_hp1.addWidget(self.batch_spin)
        row_hp1.addWidget(self.cb_early_stopping)
        hp_layout.addLayout(row_hp1)

        self.scheduler_params_edit = QLineEdit("step_size=10,gamma=0.1")
        hp_layout.addWidget(QLabel("Scheduler Params (comma key=val):"))
        hp_layout.addWidget(self.scheduler_params_edit)

        row_hp2 = QHBoxLayout()
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setValue(0.0)
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.9)
        self.label_smoothing_spin.setValue(0.0)
        self.cb_freeze_backbone = QCheckBox("Freeze Entire Backbone")
        self.clip_val_spin = QDoubleSpinBox()
        self.clip_val_spin.setRange(0.0, 10.0)
        self.clip_val_spin.setValue(0.0)

        row_hp2.addWidget(QLabel("Dropout:"))
        row_hp2.addWidget(self.dropout_spin)
        row_hp2.addWidget(QLabel("LabelSmooth:"))
        row_hp2.addWidget(self.label_smoothing_spin)
        row_hp2.addWidget(self.cb_freeze_backbone)
        row_hp2.addWidget(QLabel("ClipVal:"))
        row_hp2.addWidget(self.clip_val_spin)
        hp_layout.addLayout(row_hp2)

        row_hp3 = QHBoxLayout()
        self.cb_lr_finder = QCheckBox("LR Finder")
        self.cb_accept_lr_suggestion = QCheckBox("Accept LR Suggestion?")
        self.cb_tensorboard = QCheckBox("TensorBoard Logger")
        self.cb_mixed_precision = QCheckBox("Mixed Precision")
        self.warmup_epochs_spin = QSpinBox()
        self.warmup_epochs_spin.setRange(0, 100)
        self.warmup_epochs_spin.setValue(0)

        row_hp3.addWidget(self.cb_lr_finder)
        row_hp3.addWidget(self.cb_accept_lr_suggestion)
        row_hp3.addWidget(self.cb_tensorboard)
        row_hp3.addWidget(self.cb_mixed_precision)
        row_hp3.addWidget(QLabel("Warmup:"))
        row_hp3.addWidget(self.warmup_epochs_spin)
        hp_layout.addLayout(row_hp3)

        row_hp4 = QHBoxLayout()
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 32)
        self.num_workers_spin.setValue(8)
        self.cb_grad_checkpoint = QCheckBox("Gradient Checkpointing")
        self.cb_grad_accum = QCheckBox("Grad Accumulation")
        self.accum_batches_spin = QSpinBox()
        self.accum_batches_spin.setRange(1, 64)
        self.accum_batches_spin.setValue(2)
        self.accum_batches_spin.setEnabled(False)

        # NEW: persistent workers
        self.cb_persistent_workers = QCheckBox("Persistent Dataloader Workers")

        def toggle_accum_spin():
            self.accum_batches_spin.setEnabled(self.cb_grad_accum.isChecked())
        self.cb_grad_accum.stateChanged.connect(toggle_accum_spin)

        row_hp4.addWidget(QLabel("Workers:"))
        row_hp4.addWidget(self.num_workers_spin)
        row_hp4.addWidget(self.cb_grad_checkpoint)
        row_hp4.addWidget(self.cb_grad_accum)
        row_hp4.addWidget(QLabel("Accumulate Batches:"))
        row_hp4.addWidget(self.accum_batches_spin)

        # Add the new persistent workers checkbox
        row_hp4.addWidget(self.cb_persistent_workers)

        hp_layout.addLayout(row_hp4)

        row_hp5 = QHBoxLayout()
        self.check_val_every_n_epoch = QSpinBox()
        self.check_val_every_n_epoch.setRange(1, 50)
        self.check_val_every_n_epoch.setValue(1)

        self.btn_train = QPushButton("Start Training")
        self.btn_train.clicked.connect(self.start_training)
        row_hp5.addWidget(QLabel("Val every N epoch:"))
        row_hp5.addWidget(self.check_val_every_n_epoch)
        row_hp5.addWidget(self.btn_train)
        hp_layout.addLayout(row_hp5)

        container_layout.addWidget(group_hparams)

        # ========== PARTIAL FREEZE GROUP ==========
        group_freeze = QGroupBox("Partial Freeze Options")
        freeze_layout = QVBoxLayout(group_freeze)

        row_freeze1 = QHBoxLayout()
        row_freeze1.addWidget(QLabel("ResNet/ConvNeXt Freeze:"))
        self.cb_freeze_conv1_bn1 = QCheckBox("conv1+bn1")
        self.cb_freeze_layer1 = QCheckBox("layer1")
        self.cb_freeze_layer2 = QCheckBox("layer2")
        self.cb_freeze_layer3 = QCheckBox("layer3")
        self.cb_freeze_layer4 = QCheckBox("layer4")
        self.cb_freeze_convnext_block0 = QCheckBox("block0")
        self.cb_freeze_convnext_block1 = QCheckBox("block1")
        self.cb_freeze_convnext_block2 = QCheckBox("block2")
        self.cb_freeze_convnext_block3 = QCheckBox("block3")

        row_freeze1.addWidget(self.cb_freeze_conv1_bn1)
        row_freeze1.addWidget(self.cb_freeze_layer1)
        row_freeze1.addWidget(self.cb_freeze_layer2)
        row_freeze1.addWidget(self.cb_freeze_layer3)
        row_freeze1.addWidget(self.cb_freeze_layer4)
        row_freeze1.addWidget(self.cb_freeze_convnext_block0)
        row_freeze1.addWidget(self.cb_freeze_convnext_block1)
        row_freeze1.addWidget(self.cb_freeze_convnext_block2)
        row_freeze1.addWidget(self.cb_freeze_convnext_block3)
        freeze_layout.addLayout(row_freeze1)

        self.cb_val_center_crop = QCheckBox("Center Crop for Validation/Test")
        freeze_layout.addWidget(self.cb_val_center_crop)
        container_layout.addWidget(group_freeze)

        # ========== EARLY STOPPING GROUP ==========
        group_es = QGroupBox("Early Stopping Settings")
        es_layout = QHBoxLayout(group_es)

        es_layout.addWidget(QLabel("ES Monitor:"))
        self.es_monitor_combo = QComboBox()
        self.es_monitor_combo.addItems(["val_loss", "val_acc"])
        es_layout.addWidget(self.es_monitor_combo)

        es_layout.addWidget(QLabel("Patience:"))
        self.es_patience_spin = QSpinBox()
        self.es_patience_spin.setRange(1, 20)
        self.es_patience_spin.setValue(5)
        es_layout.addWidget(self.es_patience_spin)

        es_layout.addWidget(QLabel("MinDelta:"))
        self.es_min_delta_spin = QDoubleSpinBox()
        self.es_min_delta_spin.setRange(0.0, 1.0)
        self.es_min_delta_spin.setDecimals(4)
        self.es_min_delta_spin.setSingleStep(0.0001)
        self.es_min_delta_spin.setValue(0.0)
        es_layout.addWidget(self.es_min_delta_spin)

        es_layout.addWidget(QLabel("Mode:"))
        self.es_mode_combo = QComboBox()
        self.es_mode_combo.addItems(["min", "max"])
        es_layout.addWidget(self.es_mode_combo)

        container_layout.addWidget(group_es)

        # ========== OPTUNA GROUP ==========
        group_optuna = QGroupBox("Optuna Hyperparam Tuning")
        optuna_layout = QVBoxLayout(group_optuna)

        self.cb_optuna_use_test_metric = QCheckBox("Use Test Loss as Optuna Objective?")
        optuna_layout.addWidget(self.cb_optuna_use_test_metric)

        row_optuna = QHBoxLayout()
        self.btn_tune_optuna = QPushButton("Tune with Optuna")
        self.btn_tune_optuna.clicked.connect(self.start_optuna_tuning)
        self.optuna_trials_spin = QSpinBox()
        self.optuna_trials_spin.setRange(1, 100)
        self.optuna_trials_spin.setValue(5)
        self.optuna_timeout_spin = QSpinBox()
        self.optuna_timeout_spin.setRange(0, 100000)
        self.optuna_timeout_spin.setValue(0)
        row_optuna.addWidget(self.btn_tune_optuna)
        row_optuna.addWidget(QLabel("Trials:"))
        row_optuna.addWidget(self.optuna_trials_spin)
        row_optuna.addWidget(QLabel("Timeout (sec):"))
        row_optuna.addWidget(self.optuna_timeout_spin)
        optuna_layout.addLayout(row_optuna)

        container_layout.addWidget(group_optuna)

        # ========== EXTRA OPTIONS GROUP ==========
        group_extra = QGroupBox("Extra Options")
        extra_layout = QHBoxLayout(group_extra)
        self.cb_pretrained_weights = QCheckBox("Use Pretrained Weights")
        self.cb_pretrained_weights.setChecked(True)
        self.cb_run_gc = QCheckBox("Run GC Each Epoch")
        self.cb_enable_tta = QCheckBox("Enable TTA")
        self.cb_profile_memory = QCheckBox("Profile Memory")
        extra_layout.addWidget(self.cb_pretrained_weights)
        extra_layout.addWidget(self.cb_run_gc)
        extra_layout.addWidget(self.cb_enable_tta)
        extra_layout.addWidget(self.cb_profile_memory)
        container_layout.addWidget(group_extra)

        # ========== CUSTOM MODEL GROUP ==========
        group_custom = QGroupBox("Load Custom Model")
        custom_layout = QHBoxLayout(group_custom)
        self.cb_load_custom_model = QCheckBox("Load Custom Model")
        self.custom_model_path_edit = QLineEdit()
        btn_browse_custom_model = QPushButton("Browse Model...")
        btn_browse_custom_model.clicked.connect(self.browse_custom_model)
        self.custom_arch_file_edit = QLineEdit()
        btn_browse_arch_file = QPushButton("Browse Arch File...")
        btn_browse_arch_file.clicked.connect(self.browse_arch_file)

        custom_layout.addWidget(self.cb_load_custom_model)
        custom_layout.addWidget(QLabel("Weights Path:"))
        custom_layout.addWidget(self.custom_model_path_edit)
        custom_layout.addWidget(btn_browse_custom_model)
        custom_layout.addWidget(QLabel("Arch File:"))
        custom_layout.addWidget(self.custom_arch_file_edit)
        custom_layout.addWidget(btn_browse_arch_file)
        container_layout.addWidget(group_custom)

        # ========== EXPORT & PROGRESS ==========
        h_export = QHBoxLayout()
        self.btn_export_results = QPushButton("Export Results")
        self.btn_export_results.setEnabled(False)
        self.btn_export_results.clicked.connect(self.export_all_results)
        h_export.addWidget(self.btn_export_results)
        container_layout.addLayout(h_export)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        container_layout.addWidget(self.progress_bar)

        return self.widget_main

    def browse_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self.widget_main, "Select Dataset Folder")
        if folder:
            self.train_data_dir_edit.setText(folder)

    def browse_custom_model(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self.widget_main, "Select Custom Model File",
            filter="*.pth *.pt"
        )
        if fpath:
            self.custom_model_path_edit.setText(fpath)

    def browse_arch_file(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self.widget_main, "Select Architecture Python File", filter="*.py"
        )
        if fpath:
            self.custom_arch_file_edit.setText(fpath)

    def _parse_common_params(self) -> Tuple[DataConfig, TrainConfig]:
        dataset_dir = self.train_data_dir_edit.text().strip()
        if not os.path.isdir(dataset_dir):
            raise ValueError("Invalid dataset folder.")

        val_split = self.val_split_spin.value() / 100.0
        test_split = self.test_split_spin.value() / 100.0
        if val_split + test_split > 1.0:
            raise ValueError("val + test split cannot exceed 100%.")

        # Parse scheduler params
        scheduler_params_str = self.scheduler_params_edit.text().strip()
        scheduler_params = {}
        if scheduler_params_str:
            parts = scheduler_params_str.split(",")
            for part in parts:
                if "=" not in part:
                    raise ValueError(f"Invalid scheduler param format: {part}")
                k, v = part.split("=")
                scheduler_params[k.strip()] = float(v.strip())

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
            custom_architecture_file=self.custom_arch_file_edit.text().strip(),

            # Advanced augs
            random_gamma=self.cb_random_gamma.isChecked(),
            random_gamma_limit_low=self.spin_gamma_low.value(),
            random_gamma_limit_high=self.spin_gamma_high.value(),
            random_gamma_prob=self.spin_gamma_prob.value(),

            clahe=self.cb_clahe.isChecked(),
            clahe_clip_limit=self.spin_clahe_clip.value(),
            clahe_tile_size=self.spin_clahe_tile.value(),
            clahe_prob=self.spin_clahe_prob.value(),

            channel_shuffle=self.cb_channel_shuffle.isChecked(),
            channel_shuffle_prob=self.spin_channel_shuffle_prob.value(),

            use_posterize_solarize_equalize=self.cb_posterize_solarize_equalize.isChecked(),
            pse_prob=self.spin_pse_prob.value(),
            posterize_bits=self.spin_posterize_bits.value(),
            solarize_threshold=self.spin_solarize_threshold.value(),

            sharpen_denoise=self.cb_sharpen_denoise.isChecked(),
            sharpen_prob=self.spin_sharpen_prob.value(),
            sharpen_alpha_min=self.spin_sharpen_alpha_min.value(),
            sharpen_alpha_max=self.spin_sharpen_alpha_max.value(),
            sharpen_lightness_min=self.spin_sharpen_lightness_min.value(),
            sharpen_lightness_max=self.spin_sharpen_lightness_max.value(),

            gauss_vs_mult_noise=self.cb_gauss_vs_mult_noise.isChecked(),
            gauss_mult_prob=self.spin_gauss_mult_prob.value(),
            gauss_noise_var_limit_low=self.spin_gauss_var_low.value(),
            gauss_noise_var_limit_high=self.spin_gauss_var_high.value(),
            mult_noise_lower=self.spin_mult_lower.value(),
            mult_noise_upper=self.spin_mult_upper.value(),

            cutout_coarse_dropout=self.cb_cutout_coarse_dropout.isChecked(),
            cutout_max_holes=self.spin_cutout_max_holes.value(),
            cutout_max_height=self.spin_cutout_max_height.value(),
            cutout_max_width=self.spin_cutout_max_width.value(),
            cutout_prob=self.spin_cutout_prob.value(),

            use_shift_scale_rotate=self.cb_shift_scale_rotate.isChecked(),
            ssr_shift_limit=self.spin_ssr_shift_limit.value(),
            ssr_scale_limit=self.spin_ssr_scale_limit.value(),
            ssr_rotate_limit=self.spin_ssr_rotate_limit.value(),
            ssr_prob=self.spin_ssr_prob.value(),

            use_one_of_advanced_transforms=self.cb_use_one_of_advanced_transforms.isChecked(),
            one_of_advanced_transforms_prob=self.spin_one_of_adv_prob.value(),

            # RandAugment
            use_randaugment=self.cb_randaugment.isChecked(),

            # NEW: persistent_workers
            persistent_workers=self.cb_persistent_workers.isChecked()
        )
        return data_config, train_config

    def start_training(self):
        try:
            data_config, train_config = self._parse_common_params()
        except ValueError as e:
            QMessageBox.warning(self.widget_main, "Invalid Input", str(e))
            return

        self.btn_train.setEnabled(False)
        self.btn_tune_optuna.setEnabled(False)
        self.btn_export_results.setEnabled(False)
        self.progress_bar.setVisible(True)

        # Prepare manager dict for results
        self.train_result_manager = multiprocessing.Manager()
        self.train_result_dict = self.train_result_manager.dict()

        data_dict = data_config.__dict__
        train_dict = train_config.__dict__

        # Start process
        self.train_process = multiprocessing.Process(
            target=train_worker,
            args=(data_dict, train_dict, self.train_result_dict)
        )
        self.train_process.start()

        # Start a QTimer to poll for results
        self.training_timer = QTimer()
        self.training_timer.timeout.connect(self.check_train_process)
        self.training_timer.start(1000)

    def check_train_process(self):
        if self.train_process is not None:
            if not self.train_process.is_alive():
                # Process finished
                self.train_process.join()
                self.train_process = None
                self.training_timer.stop()
                self.train_finished()

    def train_finished(self):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_tune_optuna.setEnabled(True)

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

    def start_optuna_tuning(self):
        try:
            data_config, base_train_config = self._parse_common_params()
        except ValueError as e:
            QMessageBox.warning(self.widget_main, "Invalid Input", str(e))
            return

        self.btn_train.setEnabled(False)
        self.btn_tune_optuna.setEnabled(False)
        self.btn_export_results.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.optuna_result_manager = multiprocessing.Manager()
        self.optuna_result_dict = self.optuna_result_manager.dict()

        data_dict = data_config.__dict__
        base_train_dict = base_train_config.__dict__
        n_trials = self.optuna_trials_spin.value()
        timeout = self.optuna_timeout_spin.value()
        use_test_metric_for_optuna = self.cb_optuna_use_test_metric.isChecked()

        self.optuna_process = multiprocessing.Process(
            target=optuna_worker,
            args=(data_dict, base_train_dict, n_trials, timeout, use_test_metric_for_optuna, self.optuna_result_dict)
        )
        self.optuna_process.start()

        self.optuna_timer = QTimer()
        self.optuna_timer.timeout.connect(self.check_optuna_process)
        self.optuna_timer.start(1000)

    def check_optuna_process(self):
        if self.optuna_process is not None:
            if not self.optuna_process.is_alive():
                self.optuna_process.join()
                self.optuna_process = None
                self.optuna_timer.stop()
                self.optuna_tuning_finished()

    def optuna_tuning_finished(self):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_tune_optuna.setEnabled(True)

        status = self.optuna_result_dict.get("status", "ERROR")
        if status == "OK":
            self.best_ckpt_path = self.optuna_result_dict["ckpt_path"]
            self.last_test_metrics = self.optuna_result_dict["test_metrics"]
            self.btn_export_results.setEnabled(True)
            print(f"[INFO] Optuna tuning finished. Best checkpoint: {self.best_ckpt_path}")
        else:
            print("[ERROR] Optuna tuning encountered an error.")

        self.optuna_result_manager = None
        self.optuna_result_dict = None

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
        cm_fig_path = self.last_test_metrics.get("cm_fig_path")
        tb_log_dir = self.last_test_metrics.get("tb_log_dir")

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
            print(f"[INFO] TensorBoard logs exported to {zip_base}.zip")

        QMessageBox.information(
            self.widget_main,
            "Export Results",
            f"Results exported to {fpath}"
        )


# ===============================
# 10) UNIT & INTEGRATION TESTS
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
