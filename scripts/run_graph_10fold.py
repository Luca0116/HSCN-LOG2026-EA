#!/usr/bin/env python3
"""
One-seed × 10-fold stratified CV for HSCN graph classification.

Each fold:
    ~81% train / ~9% val / ~10% test (stratified)
    early stopping on validation
    report test accuracy at the best-validation checkpoint

This script aggregates mean ± std **over folds** for a single seed.
For the paper Table 1 protocol (mean ± std **across five seeds**), use
``scripts/run_paper_seeds.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hscn.data.graph_loaders import (
    load_tu_graphs,
    make_graph_loaders,
    stratified_kfold_masks,
)
from hscn.graph_task import train_graph_fold
from hscn.models.hscn_graph import CLASSIFIER_MODES, HSCNGraphClassifier
from hscn.nn.graph_readout import READOUT_MODES
from hscn.nn.hscn_layer import GEOMETRY_MODES, MULTIPLIER_MODES
from hscn.utils.seed import set_seed


def _flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten yaml; nested model.* keys are promoted to top-level."""
    out: Dict[str, Any] = {}
    for key, value in cfg.items():
        key = str(key).replace("-", "_")
        if key == "model" and isinstance(value, dict):
            for mk, mv in value.items():
                out[str(mk).replace("-", "_")] = mv
            continue
        out[key] = value
    return out


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _diffusion_scales_list(model) -> List[float]:
    if not getattr(model, "layers", None):
        return []
    scales = model.layers[0].diffusion_scales().detach().cpu().tolist()
    return [float(x) for x in scales]


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    defaults: Dict[str, Any] = {}
    if pre_args.config:
        import yaml

        with open(pre_args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        defaults.update(_flatten_config(cfg))

    def cfg_default(key: str, fallback: Any) -> Any:
        return defaults.get(key, fallback)

    parser = argparse.ArgumentParser(description="10-fold CV graph classification")
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--dataset", type=str, default=cfg_default("dataset", "MUTAG"))
    parser.add_argument(
        "--data_root",
        type=str,
        default=cfg_default("data_root", str(PROJECT_ROOT / "data")),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=cfg_default("out_dir", str(PROJECT_ROOT / "runs")),
    )

    parser.add_argument("--hidden_dim", type=int, default=cfg_default("hidden_dim", 32))
    parser.add_argument("--num_layers", type=int, default=cfg_default("num_layers", 2))
    _layer_dims_cfg = cfg_default("layer_hidden_dims", None)
    if isinstance(_layer_dims_cfg, (list, tuple)):
        _layer_dims_cfg = ",".join(str(int(x)) for x in _layer_dims_cfg)
    parser.add_argument(
        "--layer_hidden_dims",
        type=str,
        default=_layer_dims_cfg,
        help="Comma-separated per-layer channels, e.g. '64,96,128'",
    )
    parser.add_argument(
        "--channel_pattern",
        type=str,
        default=cfg_default("channel_pattern", "uniform"),
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        default=cfg_default("feature_mode", "mlp"),
        choices=["mlp", "direct_conv"],
    )
    parser.add_argument("--ball_dim", type=int, default=cfg_default("ball_dim", 2))
    parser.add_argument("--radius", type=float, default=cfg_default("radius", 1.0))
    parser.add_argument("--sigma", type=float, default=cfg_default("sigma", 0.0))
    parser.add_argument("--num_lambdas", type=int, default=cfg_default("num_lambdas", 12))
    parser.add_argument("--lambda_max", type=float, default=cfg_default("lambda_max", 8.0))
    parser.add_argument(
        "--num_directions", type=int, default=cfg_default("num_directions", 8)
    )
    parser.add_argument("--num_scales", type=int, default=cfg_default("num_scales", 4))
    parser.add_argument("--min_scale", type=float, default=cfg_default("min_scale", 0.05))
    parser.add_argument("--max_scale", type=float, default=cfg_default("max_scale", 5.0))
    parser.add_argument(
        "--scale_mode", type=str, default=cfg_default("scale_mode", "logspace")
    )
    parser.add_argument(
        "--learnable_scales",
        action="store_true",
        default=cfg_default("learnable_scales", False),
        help="Must remain False in the paper release (fixed scales).",
    )
    parser.add_argument("--sarkar_tau", type=float, default=cfg_default("sarkar_tau", 0.5))
    parser.add_argument("--sarkar_root", type=int, default=cfg_default("sarkar_root", None))
    parser.add_argument("--dropout", type=float, default=cfg_default("dropout", 0.5))
    parser.add_argument(
        "--conv_dropout", type=float, default=cfg_default("conv_dropout", None)
    )
    parser.add_argument(
        "--encoder_dropout", type=float, default=cfg_default("encoder_dropout", None)
    )
    parser.add_argument(
        "--classifier_dropout",
        type=float,
        default=cfg_default("classifier_dropout", None),
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default=cfg_default("norm_mode", "pre"),
        choices=["pre"],
    )
    parser.add_argument(
        "--use_post_conv_mlp",
        action="store_true",
        default=cfg_default("use_post_conv_mlp", False),
    )
    parser.add_argument(
        "--post_conv_mlp_hidden_dim",
        type=int,
        default=cfg_default("post_conv_mlp_hidden_dim", None),
    )
    parser.add_argument(
        "--post_conv_dropout",
        type=float,
        default=cfg_default("post_conv_dropout", None),
    )
    parser.add_argument(
        "--activation", type=str, default=cfg_default("activation", "gelu")
    )
    parser.add_argument(
        "--readout_mode",
        type=str,
        default=cfg_default("readout_mode", "mean"),
        choices=sorted(READOUT_MODES),
    )
    parser.add_argument(
        "--readout_dropout", type=float, default=cfg_default("readout_dropout", None)
    )
    parser.add_argument(
        "--classifier_mode",
        type=str,
        default=cfg_default("classifier_mode", "mlp"),
        choices=sorted(CLASSIFIER_MODES),
    )
    _n_hidden = cfg_default(
        "classifier_num_hidden_layers",
        cfg_default("classifier_num_layers", 2),
    )
    parser.add_argument(
        "--classifier_num_hidden_layers",
        type=int,
        default=_n_hidden,
        help="Number of hidden layers in the MLP classifier (paper: 2 × width 256).",
    )
    parser.add_argument(
        "--classifier_num_layers",
        type=int,
        default=None,
        help="Deprecated alias for --classifier_num_hidden_layers.",
    )
    parser.add_argument(
        "--classifier_hidden_dim",
        type=int,
        default=cfg_default("classifier_hidden_dim", 256),
    )
    parser.add_argument(
        "--geometry_mode",
        type=str,
        default=cfg_default("geometry_mode", "hyperbolic"),
        choices=sorted(GEOMETRY_MODES),
    )
    parser.add_argument(
        "--multiplier_mode",
        type=str,
        default=cfg_default("multiplier_mode", "full"),
        choices=sorted(MULTIPLIER_MODES),
    )
    parser.add_argument(
        "--complex_pair_mode",
        type=str,
        default=cfg_default("complex_pair_mode", "reim"),
        choices=["reim"],
    )

    parser.add_argument("--batch_size", type=int, default=cfg_default("batch_size", 32))
    parser.add_argument("--n_folds", type=int, default=cfg_default("n_folds", 10))
    parser.add_argument("--fold", type=int, default=cfg_default("fold", None))
    parser.add_argument(
        "--val_ratio_within_trainval",
        type=float,
        default=cfg_default("val_ratio_within_trainval", 0.1),
    )
    parser.add_argument("--lr", type=float, default=cfg_default("lr", 0.005))
    parser.add_argument(
        "--weight_decay", type=float, default=cfg_default("weight_decay", 1e-3)
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=cfg_default("label_smoothing", 0.0)
    )
    parser.add_argument("--epochs", type=int, default=cfg_default("epochs", 300))
    parser.add_argument("--patience", type=int, default=cfg_default("patience", 50))
    parser.add_argument("--min_epochs", type=int, default=cfg_default("min_epochs", 0))
    parser.add_argument(
        "--min_epochs_for_best",
        type=int,
        default=cfg_default("min_epochs_for_best", 0),
    )
    parser.add_argument(
        "--lr_patience", type=int, default=cfg_default("lr_patience", 20)
    )
    parser.add_argument("--grad_clip", type=float, default=cfg_default("grad_clip", 1.0))
    parser.add_argument(
        "--optimizer",
        type=str,
        default=cfg_default("optimizer", "adam"),
        choices=["adam", "adamw"],
    )
    parser.add_argument("--lr_factor", type=float, default=cfg_default("lr_factor", 0.5))
    parser.add_argument("--lr_min", type=float, default=cfg_default("lr_min", 1e-5))
    parser.add_argument("--seed", type=int, default=cfg_default("seed", 42))
    parser.add_argument("--device", type=str, default=cfg_default("device", "auto"))
    parser.add_argument(
        "--eval_train_every", type=int, default=cfg_default("eval_train_every", 1)
    )
    parser.add_argument("--log_every", type=int, default=cfg_default("log_every", 20))
    parser.add_argument("--run_name", type=str, default=cfg_default("run_name", None))

    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model(args, in_dim: int, num_classes: int, device: torch.device):
    if args.learnable_scales:
        raise ValueError("Paper release requires learnable_scales=False.")
    n_hidden = args.classifier_num_hidden_layers
    if args.classifier_num_layers is not None:
        n_hidden = args.classifier_num_layers
    return HSCNGraphClassifier(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        ball_dim=args.ball_dim,
        radius=args.radius,
        sigma=args.sigma,
        num_lambdas=args.num_lambdas,
        num_directions=args.num_directions,
        num_scales=args.num_scales,
        lambda_max=args.lambda_max,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        scale_mode=args.scale_mode,
        learnable_scales=False,
        dropout=args.dropout,
        conv_dropout=args.conv_dropout,
        encoder_dropout=args.encoder_dropout,
        classifier_dropout=args.classifier_dropout,
        norm_mode="pre",
        activation=args.activation,
        sarkar_tau=args.sarkar_tau,
        sarkar_root=args.sarkar_root,
        feature_mode=args.feature_mode,
        readout_mode=args.readout_mode,
        readout_dropout=args.readout_dropout,
        classifier_mode="mlp",
        classifier_num_hidden_layers=n_hidden,
        classifier_hidden_dim=args.classifier_hidden_dim,
        complex_pair_mode="reim",
        use_post_conv_mlp=args.use_post_conv_mlp,
        post_conv_mlp_hidden_dim=args.post_conv_mlp_hidden_dim,
        post_conv_dropout=args.post_conv_dropout,
        layer_hidden_dims=args.layer_hidden_dims,
        channel_pattern=args.channel_pattern,
        geometry_mode=args.geometry_mode,
        multiplier_mode=args.multiplier_mode,
    ).to(device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    graphs, labels, num_classes, in_dim = load_tu_graphs(
        args.dataset,
        args.data_root,
        sarkar_tau=args.sarkar_tau,
        sarkar_root=args.sarkar_root,
        sarkar_radius=args.radius,
        geometry_mode=args.geometry_mode,
    )
    fold_masks = stratified_kfold_masks(
        labels,
        n_folds=args.n_folds,
        val_ratio_within_trainval=args.val_ratio_within_trainval,
        seed=args.seed,
    )

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.fold is not None:
        if args.fold < 0 or args.fold >= args.n_folds:
            raise ValueError(f"fold must be in [0, {args.n_folds}), got {args.fold}")
        run_dir = (
            Path(args.out_dir)
            / f"{args.dataset.lower()}_fold{args.fold:02d}_{run_name}_seed{args.seed}"
        )
        fold_indices = [args.fold]
    else:
        run_dir = (
            Path(args.out_dir)
            / f"{args.dataset.lower()}_10fold_{run_name}_seed{args.seed}"
        )
        fold_indices = list(range(args.n_folds))
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"dataset={args.dataset} graphs={len(graphs)} in_dim={in_dim} classes={num_classes}")
    if args.fold is not None:
        print(
            f"protocol=single fold {args.fold}/{args.n_folds}, "
            f"val={args.val_ratio_within_trainval:.0%} of trainval"
        )
    else:
        print(
            f"protocol=10-fold stratified CV, "
            f"val={args.val_ratio_within_trainval:.0%} of trainval"
        )
    print("selection=test @ best val per fold")
    print(
        f"model=HSCNGraphClassifier readout={args.readout_mode} "
        f"geometry={args.geometry_mode} multiplier={args.multiplier_mode} "
        f"complex_pair=reim norm=pre"
    )
    print(f"run_dir={run_dir}")

    fold_results = []
    t0 = time.time()

    for fold, (train_mask, val_mask, test_mask) in enumerate(fold_masks):
        if fold not in fold_indices:
            continue
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
        print(
            f"train/val/test={int(train_mask.sum())}/"
            f"{int(val_mask.sum())}/{int(test_mask.sum())}"
        )

        set_seed(args.seed + fold)
        train_loader, val_loader, test_loader = make_graph_loaders(
            graphs,
            train_mask,
            val_mask,
            test_mask,
            batch_size=args.batch_size,
        )
        model = build_model(args, in_dim, num_classes, device)
        param_count = int(sum(p.numel() for p in model.parameters()))
        fold_dir = run_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_t0 = time.time()

        metrics = train_graph_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            args=args,
            run_dir=fold_dir,
            verbose=True,
        )
        metrics["fold"] = fold
        metrics["train_graphs"] = int(train_mask.sum())
        metrics["val_graphs"] = int(val_mask.sum())
        metrics["test_graphs"] = int(test_mask.sum())
        metrics["parameter_count"] = param_count
        metrics["training_time"] = float(time.time() - fold_t0)
        metrics["radius"] = float(args.radius)
        metrics["K"] = int(args.num_scales)
        metrics["diffusion_scales"] = _diffusion_scales_list(model)
        metrics["readout"] = str(args.readout_mode)
        metrics["hidden_dim"] = int(args.hidden_dim)
        metrics["num_layers"] = int(args.num_layers)
        metrics["config_path"] = args.config
        metrics["git_commit"] = _git_commit()
        metrics["seed"] = int(args.seed)
        metrics["dataset"] = str(args.dataset)
        fold_results.append(metrics)

        print(
            f"  fold {fold}: val={metrics['best_val_acc']:.4f} "
            f"test@best_val={metrics['test_acc_at_best_val']:.4f} "
            f"epoch={int(metrics['best_epoch'])}"
        )

    tests = np.array([r["test_acc_at_best_val"] for r in fold_results], dtype=float)
    vals = np.array([r["best_val_acc"] for r in fold_results], dtype=float)

    param_counts = [int(r.get("parameter_count", 0)) for r in fold_results]
    summary = {
        "task": "graph_classification_10fold",
        "dataset": args.dataset,
        "run_dir": str(run_dir),
        "n_folds": args.n_folds,
        "val_ratio_within_trainval": args.val_ratio_within_trainval,
        "protocol": (
            "single-seed stratified 10-fold; early stop on val; "
            "report test @ best val; mean±std over folds"
        ),
        "aggregation": "over_folds",
        "seed": int(args.seed),
        "mean_test_acc": float(tests.mean()),
        "std_test_acc": float(tests.std(ddof=0)),
        "mean_val_acc": float(vals.mean()),
        "std_val_acc": float(vals.std(ddof=0)),
        "parameter_count": int(param_counts[0]) if param_counts else 0,
        "diffusion_scales": fold_results[0].get("diffusion_scales") if fold_results else [],
        "fold_results": fold_results,
        "elapsed_sec": time.time() - t0,
        "config_path": args.config,
        "git_commit": _git_commit(),
        "args": vars(args),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        import csv

        csv_path = run_dir / "fold_results.csv"
        if fold_results:
            fieldnames = list(fold_results[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                for row in fold_results:
                    flat = {
                        k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                        for k, v in row.items()
                    }
                    writer.writerow(flat)
    except Exception as exc:
        print(f"warning: could not write fold_results.csv: {exc}")

    md_lines = [
        f"# {args.dataset} 10-fold CV",
        "",
        f"- protocol: stratified {args.n_folds}-fold, "
        f"val={args.val_ratio_within_trainval:.0%} of trainval",
        "- selection: test @ best validation (per fold)",
        f"- **test acc: {summary['mean_test_acc']:.4f} ± {summary['std_test_acc']:.4f}**",
        f"- val acc: {summary['mean_val_acc']:.4f} ± {summary['std_val_acc']:.4f}",
        "",
        "| fold | val | test@best_val | epoch |",
        "|---:|---:|---:|---:|",
    ]
    for r in fold_results:
        md_lines.append(
            f"| {int(r['fold'])} | {r['best_val_acc']:.4f} | "
            f"{r['test_acc_at_best_val']:.4f} | {int(r['best_epoch'])} |"
        )
    (run_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n=== 10-Fold Summary ===")
    print(f"test @ best val: {summary['mean_test_acc']:.4f} ± {summary['std_test_acc']:.4f}")
    print(f"val:             {summary['mean_val_acc']:.4f} ± {summary['std_val_acc']:.4f}")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
