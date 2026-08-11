#!/usr/bin/env python
"""Train the feature-based disease MLP classifier.

Generates a transparent, signature-derived synthetic training dataset and
trains a 3-layer MLP, saving weights to ``assets/models/disease_mlp.pt``.

Run::
    python scripts/train_disease_classifier.py [--epochs 200] [--samples 400]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from brainframe.classification.diseases import num_classes
from brainframe.classification.trained_model import (
    _FEATURE_DIM,
    DiseaseMLP,
    generate_training_samples,
)
from brainframe.utils.logging import get_logger

log = get_logger("train_disease_classifier")

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "assets" / "models" / "disease_mlp.pt"


def train(epochs: int = 200, n_per_class: int = 400, seed: int = 42) -> Path:
    n_classes = num_classes()
    X, y = generate_training_samples(n_per_class=n_per_class, seed=seed)
    log.info("Training set: %d samples, %d features, %d classes", *X.shape, n_classes)

    # Train/val split.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.85 * len(X))
    tr_X, tr_y = X[idx[:cut]], y[idx[:cut]]
    va_X, va_y = X[idx[cut:]], y[idx[cut:]]

    model = DiseaseMLP(_FEATURE_DIM, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(torch.from_numpy(tr_X))
        loss = loss_fn(logits, torch.from_numpy(tr_y))
        loss.backward()
        opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                va_logits = model(torch.from_numpy(va_X))
                va_acc = (va_logits.argmax(1) == torch.from_numpy(va_y)).float().mean().item()
                va_loss = loss_fn(va_logits, torch.from_numpy(va_y)).item()
            log.info(
                "epoch %3d  loss %.4f  val_acc %.3f  val_loss %.4f",
                ep + 1,
                loss.item(),
                va_acc,
                va_loss,
            )
            if va_acc >= best_acc:
                best_acc = va_acc

    # Final accuracy.
    model.eval()
    with torch.no_grad():
        full = model(torch.from_numpy(X)).argmax(1).numpy()
    acc = float((full == y).mean())
    log.info("Final training accuracy: %.3f (best val %.3f)", acc, best_acc)

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    log.info("Saved weights to %s", WEIGHTS_PATH)
    return WEIGHTS_PATH


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--samples", type=int, default=400, help="samples per class")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(epochs=args.epochs, n_per_class=args.samples, seed=args.seed)


if __name__ == "__main__":
    main()
