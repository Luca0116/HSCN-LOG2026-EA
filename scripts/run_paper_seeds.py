#!/usr/bin/env python3
"""
Paper Table 1 evaluation: five seeds × 10-fold CV.

For each seed s ∈ {42, 0, 1, 2, 3} this launches ``run_graph_10fold.py``,
which returns the mean test accuracy a_s over 10 folds. The script then
reports mean ± std **across seeds**, matching the paper protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [42, 0, 1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-seed paper evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter for child runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = PROJECT_ROOT / "scripts" / "run_graph_10fold.py"
    seed_means: List[float] = []
    seed_summaries = []

    for seed in args.seeds:
        cmd = [args.python, str(script), "--config", args.config, "--seed", str(seed)]
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.data_root is not None:
            cmd += ["--data_root", args.data_root]
        if args.out_dir is not None:
            cmd += ["--out_dir", args.out_dir]
        print("\n" + "=" * 72)
        print("Running:", " ".join(cmd))
        print("=" * 72, flush=True)
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

        # Locate the summary written by the child (seed-suffixed run_dir).
        # Prefer the most recently modified matching summary.json under out_dir.
        import yaml

        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        dataset = str(cfg.get("dataset", "MUTAG")).lower()
        run_name = str(cfg.get("run_name", "run"))
        out_dir = Path(args.out_dir or cfg.get("out_dir") or (PROJECT_ROOT / "runs"))
        run_dir = out_dir / f"{dataset}_10fold_{run_name}_seed{seed}"
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing summary for seed={seed}: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        a_s = float(summary["mean_test_acc"])
        seed_means.append(a_s)
        seed_summaries.append(
            {
                "seed": int(seed),
                "mean_test_acc_over_folds": a_s,
                "std_test_acc_over_folds": float(summary["std_test_acc"]),
                "run_dir": str(run_dir),
            }
        )
        print(f"seed={seed}: fold-mean test acc = {a_s:.4f}")

    arr = np.asarray(seed_means, dtype=float)
    paper = {
        "task": "graph_classification_paper_seeds",
        "config": args.config,
        "seeds": list(args.seeds),
        "protocol": (
            "for each seed: stratified 10-fold mean test@best-val; "
            "then mean ± sample-std across seeds"
        ),
        "aggregation": "over_seeds",
        "seed_results": seed_summaries,
        "mean_test_acc": float(arr.mean()),
        "std_test_acc": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }

    out_root = Path(args.out_dir or cfg.get("out_dir") or (PROJECT_ROOT / "runs"))
    tag = Path(args.config).stem
    paper_dir = out_root / f"paper_seeds_{tag}"
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "summary.json").write_text(json.dumps(paper, indent=2), encoding="utf-8")

    print("\n=== Paper five-seed summary (± is seed std, ddof=1) ===")
    print(f"test acc: {paper['mean_test_acc']:.4f} ± {paper['std_test_acc']:.4f}")
    print(f"saved to {paper_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
