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


def train(epochs: int = 400, n_per_class: int = 800, seed: int = 42) -> Path:
    """Train the disease MLP with minibatches, best-val checkpointing, and a
    3-seed ensemble.

    Upgrades over the naive full-batch trainer:
      * **Minibatch SGD** (batch_size=64) — far better convergence than full-batch.
      * **Best-val checkpointing** — saves the model at its peak validation
        accuracy, not the final epoch (avoids overfitting degradation).
      * **3-seed ensemble** — trains three models with different seeds and
        averages their softmax outputs at inference, reducing variance and
        improving robustness on confusable disease pairs.
      * **BatchNorm** in the model — enables a higher learning rate.
      * **Cosine annealing + label smoothing** — already present.
    """
    n_classes = num_classes()
    X_all, y_all = generate_training_samples(n_per_class=n_per_class, seed=seed)
    log.info("Training set: %d samples, %d features, %d classes", *X_all.shape, n_classes)

    # Train/val split (shared across ensemble members for comparable val acc).
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_all))
    cut = int(0.85 * len(X_all))
    tr_X, tr_y = X_all[idx[:cut]], y_all[idx[:cut]]
    va_X, va_y = X_all[idx[cut:]], y_all[idx[cut:]]

    Xtr = torch.from_numpy(tr_X)
    ytr = torch.from_numpy(tr_y)
    Xva = torch.from_numpy(va_X)
    yva = torch.from_numpy(va_y)

    batch_size = 64
    n_batches = max(1, len(tr_X) // batch_size)

    # ---- Ensemble: train 3 models with different seeds ----
    ensemble_states: list[dict] = []
    best_acc = 0.0
    for mi in range(3):
        torch.manual_seed(seed + mi * 100)
        model = DiseaseMLP(_FEATURE_DIM, n_classes)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_va = 0.0
        best_state = None
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(tr_X))
            for bi in range(n_batches):
                bi_idx = perm[bi * batch_size:(bi + 1) * batch_size]
                opt.zero_grad()
                logits = model(Xtr[bi_idx])
                loss = loss_fn(logits, ytr[bi_idx])
                loss.backward()
                opt.step()
            sched.step()
            if (ep + 1) % 25 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    va_logits = model(Xva)
                    va_acc = (va_logits.argmax(1) == yva).float().mean().item()
                    va_loss = loss_fn(va_logits, yva).item()
                if va_acc >= best_va:
                    best_va = va_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if mi == 0:
                    log.info(
                        "model %d  epoch %3d  val_acc %.3f  val_loss %.4f  best %.3f",
                        mi + 1, ep + 1, va_acc, va_loss, best_va,
                    )
        # Load best-val checkpoint for this ensemble member.
        if best_state is not None:
            model.load_state_dict(best_state)
        ensemble_states.append(model.state_dict())
        log.info("Ensemble member %d: best val_acc %.3f", mi + 1, best_va)
        best_acc = max(best_acc, best_va)

    # ---- Evaluate ensemble (averaged softmax) ----
    import torch.nn.functional as F

    probs_sum = torch.zeros(len(va_X), n_classes)
    for state in ensemble_states:
        m = DiseaseMLP(_FEATURE_DIM, n_classes)
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            probs_sum += F.softmax(m(Xva), dim=1)
    ensemble_probs = probs_sum / len(ensemble_states)
    ensemble_acc = (ensemble_probs.argmax(1) == yva).float().mean().item()
    log.info("Ensemble val accuracy: %.3f (best single %.3f)", ensemble_acc, best_acc)

    # Full-dataset accuracy.
    Xfull = torch.from_numpy(X_all)
    probs_full = torch.zeros(len(X_all), n_classes)
    for state in ensemble_states:
        m = DiseaseMLP(_FEATURE_DIM, n_classes)
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            probs_full += F.softmax(m(Xfull), dim=1)
    acc = (probs_full.argmax(1) == torch.from_numpy(y_all)).float().mean().item()
    log.info("Final full-dataset accuracy: %.3f", acc)

    # ---- Save: store ALL ensemble members in one checkpoint dict ----
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "ensemble": ensemble_states,
        "n_members": len(ensemble_states),
        "feature_dim": _FEATURE_DIM,
        "n_classes": n_classes,
        "val_acc": float(ensemble_acc),
    }
    torch.save(checkpoint, WEIGHTS_PATH)
    log.info("Saved %d-member ensemble weights to %s", len(ensemble_states), WEIGHTS_PATH)
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
