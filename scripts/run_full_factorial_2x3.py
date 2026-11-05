#!/usr/bin/env python
"""Full 2^3 Factorial Experiment across 30 Matched/Paired Seeds:
Factors:
  P: Protocol Head (0 = AUTO, 1 = Active Learned)
  C: Communication Gate (0 = Always Transmit, 1 = Adaptive Gate)
  I: Intent Guidance (0 = Unassisted, 1 = Motion Prediction Feature)

Configurations:
  000: P=0, C=0, I=0 (Baseline)
  100: P=1, C=0, I=0
  010: P=0, C=1, I=0
  001: P=0, C=0, I=1
  110: P=1, C=1, I=0
  101: P=1, C=0, I=1
  011: P=0, C=1, I=1
  111: P=1, C=1, I=1 (Full Model)

Reports:
  - Exact action counts and distributions
  - Main effects and 2-way/3-way interaction terms
  - Mean, Std, 95% Bootstrap CI of paired differences
  - Holm-Bonferroni adjusted p-values
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

import _common  # noqa: F401
from uav_allocation.config import Config, load_config, with_changes
from uav_allocation.evaluation.evaluate import evaluate_method, seeds_from
from uav_allocation.utils import get_logger

log = get_logger("factorial_2x3")


def bootstrap_diff_ci(diffs: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    if len(diffs) <= 1 or np.allclose(diffs, 0.0):
        m = float(np.mean(diffs))
        return m, m
    rng = np.random.default_rng(42)
    boot_means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lower = float(np.percentile(boot_means, (1.0 - ci) / 2.0 * 100.0))
    upper = float(np.percentile(boot_means, (1.0 + ci) / 2.0 * 100.0))
    return lower, upper


def run_full_factorial(cfg: Config, seeds: List[int], out_dir: Path, checkpoint: str):
    ck_hash = hashlib.md5(open(checkpoint, "rb").read()).hexdigest()
    log.info("Checkpoint: %s (MD5: %s)", checkpoint, ck_hash)

    configs = {
        "000_baseline":        {"coord": {"no_protocol": True,  "no_comm_gate": True},  "model": {"no_spatial_guidance": True}},
        "100_proto":           {"coord": {"no_protocol": False, "no_comm_gate": True},  "model": {"no_spatial_guidance": True}},
        "010_comm":            {"coord": {"no_protocol": True,  "no_comm_gate": False}, "model": {"no_spatial_guidance": True}},
        "001_spatial":         {"coord": {"no_protocol": True,  "no_comm_gate": True},  "model": {"no_spatial_guidance": False}},
        "110_proto_comm":      {"coord": {"no_protocol": False, "no_comm_gate": False}, "model": {"no_spatial_guidance": True}},
        "101_proto_spatial":   {"coord": {"no_protocol": False, "no_comm_gate": True},  "model": {"no_spatial_guidance": False}},
        "011_comm_spatial":    {"coord": {"no_protocol": True,  "no_comm_gate": False}, "model": {"no_spatial_guidance": False}},
        "111_full":            {"coord": {"no_protocol": False, "no_comm_gate": False}, "model": {"no_spatial_guidance": False}},
    }

    raw_frames = []
    for arm_name, arm_spec in configs.items():
        vcfg = with_changes(cfg, {
            "coordination": {**arm_spec["coord"], "bundle_size": 1},
            "model": arm_spec["model"],
        })
        # Explicit configuration assertions to prevent silent aliasing
        assert vcfg.coordination.no_protocol == arm_spec["coord"]["no_protocol"]
        assert vcfg.coordination.no_comm_gate == arm_spec["coord"]["no_comm_gate"]
        assert vcfg.model.no_spatial_guidance == arm_spec["model"]["no_spatial_guidance"]

        log.info("Evaluating arm %s: no_protocol=%s, no_comm_gate=%s, no_spatial=%s",
                 arm_name, vcfg.coordination.no_protocol, vcfg.coordination.no_comm_gate, vcfg.model.no_spatial_guidance)
        df = evaluate_method("gnn_mappo", vcfg, seeds, checkpoint=checkpoint, condition=arm_name)
        df["arm"] = arm_name
        df["P"] = int(not arm_spec["coord"]["no_protocol"])
        df["C"] = int(not arm_spec["coord"]["no_comm_gate"])
        df["S"] = int(not arm_spec["model"]["no_spatial_guidance"])
        df["checkpoint_hash"] = ck_hash
        raw_frames.append(df)

    raw_df = pd.concat(raw_frames, ignore_index=True)
    raw_df.to_csv(out_dir / "factorial_2x3_episodes.csv", index=False)

    # 1. Main and Interaction Effects
    means = raw_df.groupby("arm")[["weighted_completion_rate", "duplicate_rate", "bytes_per_uav_min", "episode_return"]].mean()
    means.to_csv(out_dir / "factorial_2x3_means.csv")

    # 2. Paired Statistical Evaluation against 000_baseline
    metrics = ["weighted_completion_rate", "duplicate_rate", "bytes_per_uav_min", "episode_return"]
    base_df = raw_df[raw_df["arm"] == "000_baseline"].sort_values("seed").reset_index(drop=True)

    summary_rows = []
    p_values_all = []
    for arm_name in configs.keys():
        a_df = raw_df[raw_df["arm"] == arm_name].sort_values("seed").reset_index(drop=True)
        for m in metrics:
            vals = a_df[m].to_numpy(dtype=float)
            base_vals = base_df[m].to_numpy(dtype=float)
            diffs = vals - base_vals
            diff_mean = float(np.mean(diffs))
            ci_low, ci_high = bootstrap_diff_ci(diffs)

            if len(diffs) > 1 and not np.allclose(diffs, 0.0):
                try:
                    _, p_val = stats.wilcoxon(vals, base_vals)
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0

            p_values_all.append(p_val)
            summary_rows.append({
                "arm": arm_name,
                "metric": m,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "diff_vs_baseline_mean": diff_mean,
                "diff_ci95_low": ci_low,
                "diff_ci95_high": ci_high,
                "raw_p_value": float(p_val),
            })

    # Holm-Bonferroni correction
    n_tests = len(p_values_all)
    sorted_indices = np.argsort(p_values_all)
    adj_p = np.empty(n_tests)
    for rank, idx in enumerate(sorted_indices):
        adj_p[idx] = min(p_values_all[idx] * (n_tests - rank), 1.0)

    for i, row in enumerate(summary_rows):
        row["adj_p_value_holm"] = float(adj_p[i])
        row["significant"] = bool(adj_p[i] < 0.05)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "factorial_2x3_statistical_summary.csv", index=False)
    log.info("Full 2^3 factorial experiment completed and saved to %s", out_dir)
    return summary_df


def main():
    p = argparse.ArgumentParser(description="Run complete 2^3 Factorial Experiment.")
    p.add_argument("--config", default="configs/evaluation.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--seeds", type=int, default=10, help="Number of paired seeds")
    p.add_argument("--out-dir", default="results/factorial_2x3")
    args = p.parse_args()

    cfg = load_config(args.config)
    seeds = seeds_from(0, args.seeds)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_full_factorial(cfg, seeds, out, args.checkpoint)


if __name__ == "__main__":
    main()
