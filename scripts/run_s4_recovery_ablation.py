#!/usr/bin/env python
"""S4 Recovery Ablation Benchmark comparing Arms A, B, C, and D across 30 matched test seeds."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from uav_allocation.config import Config, load_config, with_changes
from uav_allocation.evaluation.evaluate import evaluate_method, seeds_from
from uav_allocation.utils import get_logger

log = get_logger("s4_recovery_ablation")


def bootstrap_ci(arr: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    if len(arr) <= 1 or np.allclose(arr, arr[0]):
        m = float(np.mean(arr))
        return m, m
    rng = np.random.default_rng(42)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boot, (1.0 - ci) / 2.0 * 100.0)), float(np.percentile(boot, (1.0 + ci) / 2.0 * 100.0))


def run_s4_ablation(cfg: Config, checkpoint: str, n_seeds: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds_from(0, n_seeds)
    ck_hash = hashlib.md5(open(checkpoint, "rb").read()).hexdigest()

    # S4 Scenario: Combined Stress (35% loss, 1-4 step delay, 40s partitions, 0.002 failure rate)
    s4_changes = {
        "comm": {"loss_prob": 0.35, "delay_min": 1, "delay_max": 4, "partition_prob": 0.05, "partition_duration": 40},
        "env": {"failure_rate": 0.002},
    }
    s4_cfg = with_changes(cfg, s4_changes)

    # 4 Arms:
    # Arm A: Current (failure_aware_release=False, recovery_takeover=False)
    # Arm B: Failure-Aware Release (failure_aware_release=True, recovery_takeover=False)
    # Arm C: Failure-Aware + Recovery Takeover (failure_aware_release=True, recovery_takeover=True)
    # Arm D: Original GNN-MAPPO (no_protocol=True, no_comm_gate=True)
    arms = [
        ("Arm_A_Current", "gnn_mappo", {"coordination": {"no_protocol": False, "no_comm_gate": False, "failure_aware_release": False, "recovery_takeover": False}}),
        ("Arm_B_FailRelease", "gnn_mappo", {"coordination": {"no_protocol": False, "no_comm_gate": False, "failure_aware_release": True, "recovery_takeover": False}}),
        ("Arm_C_FailTakeover", "gnn_mappo", {"coordination": {"no_protocol": False, "no_comm_gate": False, "failure_aware_release": True, "recovery_takeover": True}}),
        ("Arm_D_OriginalMAPPO", "gnn_mappo", {"coordination": {"no_protocol": True, "no_comm_gate": True, "failure_aware_release": False, "recovery_takeover": False}}),
    ]

    all_frames = []
    for label, m_type, m_changes in arms:
        acfg = with_changes(s4_cfg, m_changes)
        log.info("Running %s...", label)
        df = evaluate_method(m_type, acfg, seeds, checkpoint=checkpoint, condition="S4_combined_stress")
        df["arm"] = label
        df["checkpoint_hash"] = ck_hash
        all_frames.append(df)

    ep_df = pd.concat(all_frames, ignore_index=True)
    ep_df.to_csv(out_dir / "s4_ablation_episodes.csv", index=False)

    # Summarize with 95% bootstrap CIs
    metrics = [
        "weighted_completion_rate", "duplicate_rate", "bytes_per_uav_min",
        "energy_per_task", "reassignment_rate", "deadline_violations", "emergency_landings", "episode_return"
    ]
    summary_rows = []
    for label, grp in ep_df.groupby("arm"):
        row = {"arm": label, "episodes": len(grp)}
        for m in metrics:
            vals = grp[m].dropna().to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            ci_low, ci_high = bootstrap_ci(vals) if len(vals) else (np.nan, np.nan)
            row[f"{m}_ci_low"] = ci_low
            row[f"{m}_ci_high"] = ci_high
        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(out_dir / "s4_ablation_summary.csv", index=False)

    # Paired Statistical Tests against Arm A (Current)
    base_df = ep_df[ep_df["arm"] == "Arm_A_Current"].sort_values("seed").reset_index(drop=True)
    comp_rows = []
    for label in ["Arm_B_FailRelease", "Arm_C_FailTakeover", "Arm_D_OriginalMAPPO"]:
        a_df = ep_df[ep_df["arm"] == label].sort_values("seed").reset_index(drop=True)
        for m in ["weighted_completion_rate", "duplicate_rate", "deadline_violations", "reassignment_rate", "bytes_per_uav_min"]:
            v_a = a_df[m].to_numpy(dtype=float)
            v_base = base_df[m].to_numpy(dtype=float)
            diffs = v_a - v_base
            ci_low, ci_high = bootstrap_ci(diffs)
            p_val = float(stats.wilcoxon(v_a, v_base).pvalue) if not np.allclose(diffs, 0.0) else 1.0
            comp_rows.append({
                "arm": label,
                "metric": m,
                "diff_vs_armA_mean": float(np.mean(diffs)),
                "diff_ci95_low": ci_low,
                "diff_ci95_high": ci_high,
                "p_value": p_val,
            })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_dir / "s4_ablation_paired_comparisons.csv", index=False)
    log.info("S4 recovery ablation complete! Saved to %s", out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/evaluation.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--out-dir", default="results/s4_recovery_ablation_30seeds")
    args = p.parse_args()

    cfg = load_config(args.config)
    run_s4_ablation(cfg, args.checkpoint, args.seeds, Path(args.out_dir))


if __name__ == "__main__":
    main()
