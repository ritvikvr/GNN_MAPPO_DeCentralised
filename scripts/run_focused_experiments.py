#!/usr/bin/env python
"""Three focused research experiments:
1. Comm Gate x Protocol Head 2x2 Factorial Test
2. Fixed vs Learned Confirmation-Window Duration Sweep
3. Conditional vs Unconditional Multi-Task Bundle Chaining
All evaluated with paired seeds and bootstrap confidence intervals.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

import _common  # noqa: F401
from uav_allocation.config import Config, load_config, with_changes
from uav_allocation.evaluation.evaluate import evaluate_method, seeds_from
from uav_allocation.evaluation.metrics import aggregate
from uav_allocation.utils import get_logger

log = get_logger("focused_experiments")


def run_factorial_comm_protocol(cfg: Config, seeds: List[int], out_dir: Path, checkpoint: str):
    """Experiment 1: Communication gate x protocol head factorial test (2x2)."""
    log.info("Running Experiment 1: Comm Gate x Protocol Head (2x2 Factorial)...")
    grid = {
        "neither": {"coordination": {"no_protocol": True, "no_comm_gate": True, "bundle_size": 1}},
        "protocol_only": {"coordination": {"no_protocol": False, "no_comm_gate": True, "bundle_size": 1}},
        "comm_gate_only": {"coordination": {"no_protocol": True, "no_comm_gate": False, "bundle_size": 1}},
        "both_enabled": {"coordination": {"no_protocol": False, "no_comm_gate": False, "bundle_size": 1}},
    }
    rows = []
    for name, change in grid.items():
        ccfg = with_changes(cfg, change)
        df = evaluate_method("gnn_mappo", ccfg, seeds, checkpoint=checkpoint, condition=name)
        df["factorial_arm"] = name
        rows.append(df)
    res_df = pd.concat(rows, ignore_index=True)
    res_df.to_csv(out_dir / "exp1_factorial_comm_protocol.csv", index=False)
    summary = aggregate(res_df, ["factorial_arm"])
    summary.to_csv(out_dir / "exp1_factorial_summary.csv", index=False)
    log.info("Experiment 1 complete. Saved to %s", out_dir / "exp1_factorial_summary.csv")
    return summary


def run_window_sweep(cfg: Config, seeds: List[int], out_dir: Path, checkpoint: str):
    """Experiment 2: Fixed vs Learned confirmation-window sweep (W in {1, 3, 5, 8, 12, learned})."""
    log.info("Running Experiment 2: Confirmation Window Sweep...")
    arms = {
        "fixed_w1": {"coordination": {"no_adaptive_window": True, "confirm_window": 1, "bundle_size": 1}},
        "fixed_w3": {"coordination": {"no_adaptive_window": True, "confirm_window": 3, "bundle_size": 1}},
        "fixed_w5": {"coordination": {"no_adaptive_window": True, "confirm_window": 5, "bundle_size": 1}},
        "fixed_w8": {"coordination": {"no_adaptive_window": True, "confirm_window": 8, "bundle_size": 1}},
        "fixed_w12": {"coordination": {"no_adaptive_window": True, "confirm_window": 12, "bundle_size": 1}},
        "learned_adaptive": {"coordination": {"no_adaptive_window": False, "bundle_size": 1}},
    }
    rows = []
    for name, change in arms.items():
        ccfg = with_changes(cfg, change)
        df = evaluate_method("gnn_mappo", ccfg, seeds, checkpoint=checkpoint, condition=name)
        df["window_mode"] = name
        rows.append(df)
    res_df = pd.concat(rows, ignore_index=True)
    res_df.to_csv(out_dir / "exp2_window_sweep.csv", index=False)
    summary = aggregate(res_df, ["window_mode"])
    summary.to_csv(out_dir / "exp2_window_summary.csv", index=False)
    log.info("Experiment 2 complete. Saved to %s", out_dir / "exp2_window_summary.csv")
    return summary


def run_bundle_chaining_test(cfg: Config, seeds: List[int], out_dir: Path, checkpoint: str):
    """Experiment 3: Single vs Conditional vs Unconditional bundle chaining."""
    log.info("Running Experiment 3: Conditional vs Single Bundle Chaining...")
    arms = {
        "single_task": {"coordination": {"bundle_size": 1}},
        "bundle_k2_conditional": {"coordination": {"bundle_size": 2}},
        "bundle_k3_conditional": {"coordination": {"bundle_size": 3}},
    }
    rows = []
    for name, change in arms.items():
        ccfg = with_changes(cfg, change)
        df = evaluate_method("gnn_mappo", ccfg, seeds, checkpoint=checkpoint, condition=name)
        df["bundle_arm"] = name
        rows.append(df)
    res_df = pd.concat(rows, ignore_index=True)
    res_df.to_csv(out_dir / "exp3_bundle_chaining.csv", index=False)
    summary = aggregate(res_df, ["bundle_arm"])
    summary.to_csv(out_dir / "exp3_bundle_summary.csv", index=False)
    log.info("Experiment 3 complete. Saved to %s", out_dir / "exp3_bundle_summary.csv")
    return summary


def main():
    p = argparse.ArgumentParser(description="Run the 3 focused research experiments.")
    p.add_argument("--config", default="configs/evaluation.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--episodes", type=int, default=5, help="Paired test seeds")
    p.add_argument("--out-dir", default="results/focused_experiments")
    args = p.parse_args()

    cfg = load_config(args.config)
    seeds = seeds_from(0, args.episodes)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Launching focused experiments with %d paired seeds...", len(seeds))
    run_factorial_comm_protocol(cfg, seeds, out, args.checkpoint)
    run_window_sweep(cfg, seeds, out, args.checkpoint)
    run_bundle_chaining_test(cfg, seeds, out, args.checkpoint)
    log.info("All 3 focused experiments completed!")


if __name__ == "__main__":
    main()
