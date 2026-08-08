"""Training utilities for graph classification folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hscn.models.hscn_graph import HSCNGraphClassifier


def forward_graph_batch(model: HSCNGraphClassifier, batch) -> torch.Tensor:
    z = getattr(batch, "z", None)
    return model(batch.x, batch.edge_index, batch.batch, z=z)


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for batch in loader:
        batch = batch.to(device)
        logits = forward_graph_batch(model, batch)
        loss = F.cross_entropy(logits, batch.y.view(-1))
        loss_sum += float(loss.item()) * batch.num_graphs
        pred = logits.argmax(dim=-1)
        correct += int((pred == batch.y.view(-1)).sum().item())
        total += batch.num_graphs

    acc = correct / max(total, 1)
    return {"acc": acc, "loss": loss_sum / max(total, 1)}


def train_one_epoch(model, loader, optimizer, device, label_smoothing, grad_clip) -> float:
    model.train()
    loss_sum = 0.0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_graph_batch(model, batch)
        loss = F.cross_entropy(logits, batch.y.view(-1), label_smoothing=label_smoothing)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_sum += float(loss.item()) * batch.num_graphs
        total += batch.num_graphs

    return loss_sum / max(total, 1)


def build_optimizer(model, args: Any):
    optimizer_name = str(getattr(args, "optimizer", "adam")).lower()
    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_graph_fold(
    model: HSCNGraphClassifier,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    args: Any,
    run_dir: Optional[Path] = None,
    verbose: bool = True,
    on_epoch_end: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Train one fold with validation-based early stopping.

    Returns test accuracy at the epoch with best validation accuracy.
    """
    optimizer = build_optimizer(model, args)
    scheduler_factor = float(getattr(args, "lr_factor", 0.5))
    scheduler_min_lr = float(getattr(args, "lr_min", 1e-5))
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=args.lr_patience,
        min_lr=scheduler_min_lr,
    )

    best_val = -1.0
    best_test_at_val = 0.0
    best_train_at_val = 0.0
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    min_epochs = int(getattr(args, "min_epochs", 0) or 0)
    min_epochs_for_best = int(getattr(args, "min_epochs_for_best", 0) or 0)
    eval_train_every = int(getattr(args, "eval_train_every", 1) or 0)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.label_smoothing,
            args.grad_clip,
        )
        if eval_train_every > 0 and (epoch == 1 or epoch % eval_train_every == 0):
            train_metrics = evaluate(model, train_loader, device)
        else:
            train_metrics = {"acc": float("nan"), "loss": float("nan")}
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        scheduler.step(val_metrics["acc"])

        if epoch >= min_epochs_for_best and val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_epoch = epoch
            best_test_at_val = test_metrics["acc"]
            best_train_at_val = (
                train_metrics["acc"]
                if eval_train_every > 0
                else float("nan")
            )
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if run_dir is not None:
                torch.save(best_state, run_dir / "best_val_model.pt")
        elif epoch >= min_epochs_for_best:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % args.log_every == 0):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch={epoch:4d} loss={loss:.4f} train={train_metrics['acc']:.4f} "
                f"val={val_metrics['acc']:.4f} test={test_metrics['acc']:.4f} lr={lr:.6f}"
            )

        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "train_acc": train_metrics["acc"],
                    "val_acc": val_metrics["acc"],
                    "test_acc": test_metrics["acc"],
                    "best_val_acc": best_val,
                    "test_acc_at_best_val": best_test_at_val,
                    "bad_epochs": bad_epochs,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if epoch >= min_epochs and bad_epochs >= args.patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_epoch": float(best_epoch),
        "best_val_acc": best_val,
        "test_acc_at_best_val": best_test_at_val,
        "train_acc_at_best_val": best_train_at_val,
    }
