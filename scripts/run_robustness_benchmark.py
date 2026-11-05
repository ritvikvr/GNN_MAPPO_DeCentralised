#!/usr/bin/env python
"""Robustness benchmark across S0-S4 comparing GNN-MAPPO against baselines with 30 paired seeds."""
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

log = get_logger("robustness_benchmark")


def bootstrap_ci(arr: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    if len(arr) <= 1 or np.allclose(arr, arr[0]):
        m = float(np.mean(arr))
        return m, m
    rng = np.random.default_rng(42)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boot, (1.0 - ci) / 2.0 * 100.0)), float(np.percentile(boot, (1.0 + ci) / 2.0 * 100.0))


def run_robustness_benchmark(cfg: Config, checkpoint: str, n_seeds: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds_from(0, n_seeds)
    ck_hash = hashlib.md5(open(checkpoint, "rb").read()).hexdigest()

    # 1. Five Scenarios (S0 - S4)
    scenarios = {
        "S0_normal": {
            "comm": {"loss_prob": 0.05, "delay_min": 0, "delay_max": 2, "partition_prob": 0.0},
            "env": {"failure_rate": 0.0},
        },
        "S1_degraded_comm": {
            "comm": {"loss_prob": 0.35, "delay_min": 1, "delay_max": 4, "partition_prob": 0.0},
            "env": {"failure_rate": 0.0},
        },
        "S2_partition": {
            "comm": {"loss_prob": 0.05, "delay_min": 0, "delay_max": 2, "partition_prob": 0.05, "partition_duration": 40},
            "env": {"failure_rate": 0.0},
        },
        "S3_uav_failure": {
            "comm": {"loss_prob": 0.05, "delay_min": 0, "delay_max": 2, "partition_prob": 0.0},
            "env": {"failure_rate": 0.002},
        },
        "S4_combined_stress": {
            "comm": {"loss_prob": 0.35, "delay_min": 1, "delay_max": 4, "partition_prob": 0.05, "partition_duration": 40},
            "env": {"failure_rate": 0.002},
        },
    }

    # 2. Methods to benchmark:
    # - gnn_mappo_proposed: Full coordination mechanisms + guardrail
    # - gnn_mappo_original: Ablation without protocol & comm gate
    # - cbba: Distributed bundle algorithm
    # - cbba_etc: Event-triggered communication CBBA
    # - greedy: Centralized greedy auction
    # - hungarian: Centralized optimal assignment
    method_configs = [
        ("gnn_mappo_proposed", "gnn_mappo", {"coordination": {"no_protocol": False, "no_comm_gate": False}}),
        ("gnn_mappo_original", "gnn_mappo", {"coordination": {"no_protocol": True, "no_comm_gate": True}}),
        ("cbba", "cbba", {}),
        ("cbba_etc", "cbba_etc", {}),
        ("greedy", "greedy", {}),
        ("hungarian", "hungarian", {}),
    ]

    all_frames = []
    for s_name, s_changes in scenarios.items():
        log.info("=== Running Scenario %s ===", s_name)
        scfg = with_changes(cfg, s_changes)
        for label, m_type, m_changes in method_configs:
            mcfg = with_changes(scfg, m_changes)
            log.info("  Method: %s (type: %s)", label, m_type)
            df = evaluate_method(m_type, mcfg, seeds, checkpoint=checkpoint, condition=s_name)
            df["method_label"] = label
            df["scenario"] = s_name
            df["checkpoint_hash"] = ck_hash
            all_frames.append(df)

    ep_df = pd.concat(all_frames, ignore_index=True)
    ep_df.to_csv(out_dir / "robustness_episodes.csv", index=False)

    # 3. Summary aggregation with 95% bootstrap CIs
    metrics = [
        "weighted_completion_rate", "duplicate_rate", "bytes_per_uav_min",
        "energy_per_task", "reassignment_rate", "deadline_violations", "episode_return"
    ]
    summary_rows = []
    for (s_name, label), grp in ep_df.groupby(["scenario", "method_label"]):
        row = {"scenario": s_name, "method": label, "episodes": len(grp)}
        for m in metrics:
            vals = grp[m].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_ci_low"] = np.nan
                row[f"{m}_ci_high"] = np.nan
            else:
                row[f"{m}_mean"] = float(np.mean(vals))
                ci_low, ci_high = bootstrap_ci(vals)
                row[f"{m}_ci_low"] = ci_low
                row[f"{m}_ci_high"] = ci_high
        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(out_dir / "robustness_summary.csv", index=False)

    # 4. Degradation Delta calculation: Delta C_s = C_normal - C_s
    deg_rows = []
    for label in [m[0] for m in method_configs]:
        s0_grp = ep_df[(ep_df["scenario"] == "S0_normal") & (ep_df["method_label"] == label)].sort_values("seed")
        s0_comp = s0_grp["weighted_completion_rate"].to_numpy(dtype=float)
        for s_name in ["S1_degraded_comm", "S2_partition", "S3_uav_failure", "S4_combined_stress"]:
            s_grp = ep_df[(ep_df["scenario"] == s_name) & (ep_df["method_label"] == label)].sort_values("seed")
            s_comp = s_grp["weighted_completion_rate"].to_numpy(dtype=float)
            delta_c = s0_comp - s_comp
            deg_rows.append({
                "method": label,
                "scenario": s_name,
                "delta_completion_mean": float(np.mean(delta_c)),
                "delta_completion_ci_low": bootstrap_ci(delta_c)[0],
                "delta_completion_ci_high": bootstrap_ci(delta_c)[1],
            })
    deg_df = pd.DataFrame(deg_rows)
    deg_df.to_csv(out_dir / "robustness_degradation_deltas.csv", index=False)
    log.info("Robustness benchmark complete! Saved to %s", out_dir)


def main():
    p = argparse.ArgumentParser(description="Run 5-scenario robustness benchmark.")
    p.add_argument("--config", default="configs/evaluation.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--out-dir", default="results/robustness_benchmark_30seeds")
    args = p.parse_args()

    cfg = load_config(args.config)
    run_robustness_benchmark(cfg, args.checkpoint, args.seeds, Path(args.out_dir))


if __name__ == "__main__":
    main()
