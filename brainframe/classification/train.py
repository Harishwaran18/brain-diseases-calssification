"""Training loop for the supervised classification baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from brainframe.config import ClassificationConfig
from brainframe.utils.io import ensure_dir, save_json
from brainframe.utils.logging import get_logger
from brainframe.utils.metrics import accuracy
from brainframe.utils.seed import set_seed

log = get_logger("classification.train")


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    metrics: dict[str, float]


@dataclass
class TrainHistory:
    epochs: list[EpochResult] = field(default_factory=list)
    best_epoch: int = 0
    best_metric: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "epochs": [e.__dict__ for e in self.epochs],
        }


def _build_optimizer(cfg: ClassificationConfig, params) -> torch.optim.Optimizer:
    name = cfg.train.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.train.learning_rate, momentum=0.9, weight_decay=cfg.train.weight_decay
        )
    raise ValueError(f"Unknown optimizer {cfg.train.optimizer}")


def _build_scheduler(cfg: ClassificationConfig, optimizer, steps: int):
    name = cfg.train.scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, steps // 4), gamma=0.5)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def _classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    tgt = targets.detach().cpu().numpy()
    acc = accuracy(preds, tgt)
    # binary F1/AUC for the common 2-class case
    f1 = float(acc)  # simplified; replaced below if sklearn available
    try:
        from sklearn.metrics import f1_score, roc_auc_score

        if logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            f1 = float(f1_score(tgt, preds, zero_division=0))
            try:
                auc = (
                    float(roc_auc_score(tgt, probs)) if len(set(tgt.tolist())) > 1 else float("nan")
                )
            except ValueError:
                auc = float("nan")
            return {"accuracy": acc, "f1": f1, "auc": auc}
    except ImportError:  # pragma: no cover
        pass
    return {"accuracy": acc, "f1": f1}


def train_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: ClassificationConfig,
    device: torch.device | str = "cpu",
    history_path: str | None = None,
) -> tuple[nn.Module, TrainHistory]:
    """Train ``model`` with early stopping and checkpointing.

    ``train_loader`` / ``val_loader`` are iterables yielding ``(volume, label)`` tensors.
    """
    set_seed()
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(cfg, model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    scheduler = _build_scheduler(cfg, optimizer, cfg.train.epochs)

    history = TrainHistory()
    best_metric = float("inf")
    patience_left = cfg.train.early_stopping_patience
    ensure_dir(Path(cfg.train.checkpoint).parent)

    for epoch in range(cfg.train.epochs):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            volumes, labels = batch
            volumes = volumes.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(volumes)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        val_losses, val_logits_all, val_targets_all = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                volumes, labels = batch
                volumes = volumes.to(device)
                labels = labels.to(device)
                logits = model(volumes)
                val_losses.append(float(criterion(logits, labels).item()))
                val_logits_all.append(logits.cpu())
                val_targets_all.append(labels.cpu())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        if val_logits_all:
            metrics = _classification_metrics(torch.cat(val_logits_all), torch.cat(val_targets_all))
        else:
            metrics = {"accuracy": 0.0}

        res = EpochResult(epoch, train_loss, val_loss, metrics)
        history.epochs.append(res)
        log.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f %s",
            epoch + 1,
            cfg.train.epochs,
            train_loss,
            val_loss,
            {k: round(v, 4) for k, v in metrics.items()},
        )

        monitor = (
            val_loss
            if cfg.train.early_stopping_metric == "val_loss"
            else metrics.get(cfg.train.early_stopping_metric, val_loss)
        )
        improved = (
            monitor < best_metric - 1e-6
            if cfg.train.early_stopping_mode == "min"
            else monitor > best_metric + 1e-6
        )
        if improved:
            best_metric = monitor
            history.best_epoch = epoch
            history.best_metric = monitor
            patience_left = cfg.train.early_stopping_patience
            torch.save(model.state_dict(), cfg.train.checkpoint)
            log.info("  -> new best (%.4f) saved to %s", monitor, cfg.train.checkpoint)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1,
                    cfg.train.early_stopping_patience,
                )
                break

        scheduler.step()

    if Path(cfg.train.checkpoint).exists():
        model.load_state_dict(torch.load(cfg.train.checkpoint, map_location=device))
    if history_path:
        save_json(history.to_dict(), history_path)
    return model, history
