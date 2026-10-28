"""
Deep learning model factories for manufacturing AI benchmarks (T2–T4).

Models:
  T2: MobileNet-v2 (3.5M params) — edge visual inspection
  T3: ResNet-50 (25M params) — surface defect detection
  T3: LSTM (128 hidden) — remaining useful life prediction
  T4: ResNet-152 (60M params) — complex defect detection (large CNN)
  T4: ViT-B/16 (86M params) — complex defect detection (transformer)

Each factory returns (model, train_fn, inference_fn) for uniform CarbonTracker
integration.  All vision models support ImageNet-pretrained weights for
transfer learning experiments (Method C).

References:
  Geißler et al. (2024) — transfer learning energy savings (40–62 %)
  Luccioni et al. (2024) — task-specific vs. general-purpose comparison
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ModelFactory = Tuple[Any, Callable[..., Any], Callable[..., np.ndarray]]


# ======================================================================
# Shared training utilities
# ======================================================================

def _get_device() -> str:
    """Detect best available PyTorch device."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_training_loop(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    criterion: Any,
    optimizer: Any,
    epochs: int = 50,
    device: str = "cuda",
) -> Any:
    """Standard PyTorch training loop with per-epoch logging.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs: Number of training epochs (default 50 per research protocol).
        device: Compute device string.

    Returns:
        Trained model.
    """
    import torch

    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss: {train_loss/len(train_loader):.4f} | "
                f"val_loss: {val_loss/len(val_loader):.4f} | "
                f"val_acc: {100.*correct/total:.1f}%"
            )

    return model


def _get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    classification: bool = True,
) -> Tuple[Any, Any]:
    """Create PyTorch DataLoaders from numpy arrays.

    Args:
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.
        batch_size: Batch size for training DataLoader.
        classification: If True, cast targets to long (class indices);
                        if False, keep as float32 (regression).

    Returns:
        (train_loader, val_loader) tuple.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    y_train_t = torch.from_numpy(y_train).long() if classification else torch.from_numpy(y_train).float()
    y_test_t = torch.from_numpy(y_test).long() if classification else torch.from_numpy(y_test).float()

    train_ds = TensorDataset(torch.from_numpy(X_train), y_train_t)
    val_ds = TensorDataset(torch.from_numpy(X_test), y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader


# ======================================================================
# T2: MobileNet-v2
# ======================================================================

def create_mobilenet_v2(
    num_classes: int = 6,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> ModelFactory:
    """Create MobileNet-v2 for edge manufacturing visual inspection (T2).

    Use case: Steel surface defect detection on edge devices (Jetson, Coral).
    3.5M parameters — the most carbon-efficient CNN for visual tasks.

    Args:
        num_classes: Number of defect classes (6 for NEU steel dataset).
        freeze_backbone: Freeze ImageNet backbone weights (transfer learning).
        pretrained: Use ImageNet pretrained weights.

    Returns:
        (model, train_fn, inference_fn) tuple.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    device = _get_device()

    # Replace classifier head for manufacturing defect classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    def train_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Any:
        logger.info(f"Training MobileNet-v2 (T2) on {X_train.shape[0]} images")
        train_loader, val_loader = _get_dataloaders(
            X_train, y_train, X_test, y_test, batch_size,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
        return _default_training_loop(
            model, train_loader, val_loader, criterion, optimizer, epochs, device,
        )

    def inference_fn(X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        import torch
        model.eval()
        model.to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i+batch_size]).to(device)
                out = model(batch).argmax(dim=1).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds)

    logger.info(f"Created MobileNet-v2 (T2, {sum(p.numel() for p in model.parameters()):,} params)")
    return model, train_fn, inference_fn


# ======================================================================
# T3: ResNet-50
# ======================================================================

def create_resnet50(
    num_classes: int = 6,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> ModelFactory:
    """Create ResNet-50 for manufacturing defect detection (T3).

    Use case: Steel surface defect detection — enables T2→T3 carbon efficiency
    comparison against MobileNet-v2 on the same task.
    25M parameters, the industry standard for visual inspection.

    Args:
        num_classes: Number of defect classes.
        freeze_backbone: Use transfer learning (freeze ImageNet weights).
        pretrained: Use ImageNet pretrained weights.

    Returns:
        (model, train_fn, inference_fn) tuple.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    device = _get_device()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    def train_fn(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        logger.info(f"Training ResNet-50 (T3) on {X_train.shape[0]} images")
        train_loader, val_loader = _get_dataloaders(
            X_train, y_train, X_test, y_test, batch_size,
        )
        criterion = nn.CrossEntropyLoss()
        params = model.fc.parameters() if freeze_backbone else model.parameters()
        lr = 1e-3 if freeze_backbone else 1e-4
        optimizer = torch.optim.Adam(params, lr=lr)
        return _default_training_loop(
            model, train_loader, val_loader, criterion, optimizer, epochs, device,
        )

    def inference_fn(X, batch_size=32):
        import torch
        model.eval()
        model.to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i+batch_size]).to(device)
                out = model(batch).argmax(dim=1).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds)

    logger.info(f"Created ResNet-50 (T3, {sum(p.numel() for p in model.parameters()):,} params)")
    return model, train_fn, inference_fn


# ======================================================================
# T4: ResNet-152
# ======================================================================

def create_resnet152(
    num_classes: int = 6,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> ModelFactory:
    """Create ResNet-152 for complex manufacturing defect detection (T4).

    Use case: Same defect detection as ResNet-50 — enables T3→T4 same-family
    scaling comparison.  60M params, 2.4× ResNet-50 size.

    Args:
        num_classes: Number of defect classes.
        freeze_backbone: Use transfer learning.
        pretrained: Use ImageNet pretrained weights.

    Returns:
        (model, train_fn, inference_fn) tuple.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet152, ResNet152_Weights

    weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet152(weights=weights)
    device = _get_device()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    def train_fn(X_train, y_train, X_test, y_test, epochs=50, batch_size=16):
        logger.info(f"Training ResNet-152 (T4) on {X_train.shape[0]} images")
        train_loader, val_loader = _get_dataloaders(
            X_train, y_train, X_test, y_test, batch_size,
        )
        criterion = nn.CrossEntropyLoss()
        params = model.fc.parameters() if freeze_backbone else model.parameters()
        lr = 1e-3 if freeze_backbone else 1e-4
        optimizer = torch.optim.Adam(params, lr=lr)
        return _default_training_loop(
            model, train_loader, val_loader, criterion, optimizer, epochs, device,
        )

    def inference_fn(X, batch_size=16):
        import torch
        model.eval()
        model.to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i+batch_size]).to(device)
                out = model(batch).argmax(dim=1).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds)

    logger.info(f"Created ResNet-152 (T4, {sum(p.numel() for p in model.parameters()):,} params)")
    return model, train_fn, inference_fn


# ======================================================================
# T4: ViT-B/16
# ======================================================================

def create_vit_b16(
    num_classes: int = 6,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> ModelFactory:
    """Create Vision Transformer (ViT-B/16) for defect detection (T4).

    Use case: Steel surface defect detection — enables CNN→Transformer comparison.
    86M parameters.  Tests the "carbon wall" hypothesis: does the accuracy gain
    from T3→T4 justify the carbon cost?

    Args:
        num_classes: Number of defect classes.
        freeze_backbone: Freeze transformer encoder weights.
        pretrained: Use ImageNet-21k pretrained weights.

    Returns:
        (model, train_fn, inference_fn) tuple.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    device = _get_device()

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.parameters():
            param.requires_grad = True

    def train_fn(X_train, y_train, X_test, y_test, epochs=50, batch_size=16):
        logger.info(f"Training ViT-B/16 (T4) on {X_train.shape[0]} images")
        train_loader, val_loader = _get_dataloaders(
            X_train, y_train, X_test, y_test, batch_size,
        )
        criterion = nn.CrossEntropyLoss()
        params = model.heads.parameters() if freeze_backbone else model.parameters()
        lr = 1e-3 if freeze_backbone else 1e-4
        optimizer = torch.optim.Adam(params, lr=lr)
        return _default_training_loop(
            model, train_loader, val_loader, criterion, optimizer, epochs, device,
        )

    def inference_fn(X, batch_size=16):
        import torch
        model.eval()
        model.to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i+batch_size]).to(device)
                out = model(batch).argmax(dim=1).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds)

    logger.info(f"Created ViT-B/16 (T4, {sum(p.numel() for p in model.parameters()):,} params)")
    return model, train_fn, inference_fn


# ======================================================================
# T3: LSTM for RUL prediction
# ======================================================================

def create_lstm_rul(
    input_features: int = 26,
    hidden_size: int = 128,
    num_layers: int = 1,
) -> ModelFactory:
    """Create LSTM for remaining useful life (RUL) prediction (T3).

    Use case: NASA CMAPSS turbofan engine degradation — enables T2 XGBoost vs.
    T3 LSTM comparison on the same regression task.

    Args:
        input_features: Number of sensor channels (26 for CMAPSS).
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers (1 for T3).

    Returns:
        (model, train_fn, inference_fn) tuple.
    """
    import torch
    import torch.nn as nn

    class _LSTM_RUL(nn.Module):
        """Simple LSTM regressor for RUL prediction."""

        def __init__(self, in_f: int, hidden: int, layers: int):
            super().__init__()
            self.lstm = nn.LSTM(in_f, hidden, layers, batch_first=True)
            self.regressor = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.regressor(out[:, -1, :]).squeeze(-1)

    model = _LSTM_RUL(input_features, hidden_size, num_layers)
    device = _get_device()

    def train_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Any:
        logger.info(f"Training LSTM (T3) — {X_train.shape[0]} sequences")
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train).float(),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.to(device)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        val_loss += criterion(model(batch_x), batch_y).item()
                logger.info(
                    f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"train_loss: {train_loss/len(train_loader):.4f} | "
                    f"val_loss: {val_loss/len(val_loader):.4f}"
                )

        return model

    def inference_fn(X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        import torch
        model.eval()
        model.to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i+batch_size]).to(device)
                preds.append(model(batch).cpu().numpy())
        return np.concatenate(preds)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Created LSTM RUL (T3, {param_count:,} params)")
    return model, train_fn, inference_fn


# ======================================================================
# Method C: Transfer learning comparison
# ======================================================================

def compare_training_modes(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 6,
) -> Dict[str, Dict[str, float]]:
    """Compare three training modes for transfer learning carbon analysis (Method C).

    Tests three conditions on the same model architecture:
      1. From scratch — random initialisation, full training.
      2. Frozen backbone — pretrained encoder, train classifier head only.
      3. Full fine-tuning — pretrained, all parameters trainable.

    This implements the methodology from Geißler et al. (2024) who found that
    transfer learning saves 40–62 % of training energy.

    Args:
        model_name: 'resnet_50' or 'vit_b_16' (models with pretrained weights).
        X_train, X_test: Image arrays.
        y_train, y_test: Labels.
        num_classes: Number of defect classes.

    Returns:
        Dictionary mapping mode → {energy_saved_pct, ...}.
        Note: actual carbon values are measured externally by CarbonTracker.
    """
    logger.info(f"=== Transfer Learning Comparison: {model_name} ===")

    results: Dict[str, Dict[str, float]] = {}

    # Mode 1: From scratch
    logger.info("Mode 1: Training from scratch (random init)...")
    if model_name == "resnet_50":
        _, train_fn, _ = create_resnet50(num_classes, freeze_backbone=False, pretrained=False)
    else:
        _, train_fn, _ = create_vit_b16(num_classes, freeze_backbone=False, pretrained=False)
    # Training happens inside CarbonTracker in the experiment script
    results["from_scratch"] = {"description": "Random initialisation, full training"}

    # Mode 2: Frozen backbone (transfer learning)
    logger.info("Mode 2: Frozen backbone (transfer learning)...")
    if model_name == "resnet_50":
        _, train_fn, _ = create_resnet50(num_classes, freeze_backbone=True, pretrained=True)
    else:
        _, train_fn, _ = create_vit_b16(num_classes, freeze_backbone=True, pretrained=True)
    results["frozen_backbone"] = {"description": "Pretrained encoder frozen, classifier only"}

    # Mode 3: Full fine-tuning
    logger.info("Mode 3: Full fine-tuning...")
    if model_name == "resnet_50":
        _, train_fn, _ = create_resnet50(num_classes, freeze_backbone=False, pretrained=True)
    else:
        _, train_fn, _ = create_vit_b16(num_classes, freeze_backbone=False, pretrained=True)
    results["full_finetune"] = {"description": "Pretrained, all parameters trainable"}

    return results
