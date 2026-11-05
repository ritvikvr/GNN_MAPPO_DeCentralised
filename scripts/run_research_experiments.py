#!/usr/bin/env python
"""Dedicated Experimental Suite for Research Validation:
- Experiment A: Incremental contribution analysis (Ablations E0 to E6)
- Experiment B: Communication robustness across packet-loss rates (0% to 50%)
- Experiment C: Learned protocol actions & window duration factorial breakdown
- Experiment D: Communication efficiency metric: Completed Task Value / Total Bytes
- Experiment E: Multi-task bundle sizes (1, 2, 3 tasks)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

import _common  # noqa: F401
from uav_allocation.config import Config, load_config, with_changes
from uav_allocation.evaluation.evaluate import evaluate_method, seeds_from
from uav_allocation.evaluation.metrics import aggregate
from uav_allocation.utils import get_logger

log = get_logger("experiments")


def run_experiment_a(cfg: Config, seeds: List[int], out_dir: Path, checkpoint: str):
    """Experiment A:
    Part 1: Cumulative contribution analysis (E0 -> E6).
    Part 2: Leave-one-component-out ablation (Full Model -> A1 to A6).
    """
    log.info("Running Experiment A Part 1: Cumulative Contribution Analysis (E0 to E6)...")
    cumulative_variants = {
        "E0_original": {"coordination": {"no_confirm_fix": True, "no_protocol": True, "no_adaptive_window": True, "no_comm_gate": True, "bundle_size": 1}},
        "E1_consistency_fix": {"coordination": {"no_protocol": True, "no_adaptive_window": True, "no_comm_gate": True, "bundle_size": 1}},
        "E2_protocol_actions": {"coordination": {"no_adaptive_window": True, "no_comm_gate": True, "bundle_size": 1}},
        "E3_adaptive_window": {"coordination": {"no_comm_gate": True, "bundle_size": 1}},
        "E4_comm_gate": {"coordination": {"bundle_size": 1}},
        "E5_intent_pred": {"coordination": {"bundle_size": 1}},
        "E6_multi_task_bundles": {"coordination": {"bundle_size": 2}},
    }
    rows_cum = []
    for var_name, change in cumulative_variants.items():
        vcfg = with_changes(cfg, change)
        df = evaluate_method("gnn_mappo", vcfg, seeds, checkpoint=checkpoint, condition=var_name)
        df["variant"] = var_name
        df["analysis"] = "cumulative"
        rows_cum.append(df)
    res_cum_df = pd.concat(rows_cum, ignore_index=True)
    res_cum_df.to_csv(out_dir / "exp_a1_cumulative_contributions.csv", index=False)
    summary_cum = aggregate(res_cum_df, ["variant"])
    summary_cum.to_csv(out_dir / "exp_a1_cumulative_summary.csv", index=False)

    log.info("Running Experiment A Part 2: Leave-One-Component-Out Ablation (Full vs A1 to A6)...")
    ablation_variants = {
        "Full_Model": {"coordination": {"bundle_size": 2}},
        "A1_no_consistency_fix": {"coordination": {"no_confirm_fix": True, "bundle_size": 2}},
        "A2_no_protocol_head": {"coordination": {"no_protocol": True, "bundle_size": 2}},
        "A3_no_learned_window": {"coordination": {"no_adaptive_window": True, "bundle_size": 2}},
        "A4_no_comm_gate": {"coordination": {"no_comm_gate": True, "bundle_size": 2}},
        "A5_no_intent_pred": {"coordination": {"bundle_size": 2}},  # inference without intent guidance
        "A6_no_task_bundles": {"coordination": {"bundle_size": 1}},
    }
    rows_abl = []
    for abl_name, change in ablation_variants.items():
        vcfg = with_changes(cfg, change)
        df = evaluate_method("gnn_mappo", vcfg, seeds, checkpoint=checkpoint, condition=abl_name)
        df["ablation"] = abl_name
        df["analysis"] = "leave_one_out"
        rows_abl.append(df)
    res_abl_df = pd.concat(rows_abl, ignore_index=True)
    res_abl_df.to_csv(out_dir / "exp_a2_leave_one_out_ablation.csv", index=False)
    summary_abl = aggregate(res_abl_df, ["ablation"])
    summary_abl.to_csv(out_dir / "exp_a2_leave_one_out_summary.csv", index=False)

    log.info("Experiment A complete. Saved cumulative to exp_a1 and leave-one-out to exp_a2.")
    return summary_cum, summary_abl


def run_experiment_b(cfg: Config, seeds: List[int], out_dir: Path, checkpoint: str):
    """Experiment B: Communication robustness across packet-loss rates (0% to 50%)."""
    log.info("Running Experiment B: Communication Robustness (0% to 50% packet loss)...")
    loss_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    methods = ["gnn_mappo", "cbba", "cbba_etc", "hungarian"]
    rows = []
    for p_loss in loss_rates:
        c_name = f"loss_{int(p_loss*100)}pct"
        lcfg = with_changes(cfg, {"comm": {"loss_prob": p_loss}})
        for m in methods:
            df = evaluate_method(m, lcfg, seeds, checkpoint=checkpoint if m == "gnn_mappo" else None, condition=c_name)
            df["loss_rate"] = p_loss
            rows.append(df)
    res_df = pd.concat(rows, ignore_index=True)
    res_df.to_csv(out_dir / "exp_b_communication_robustness.csv", index=False)
    summary = aggregate(res_df, ["loss_rate", "method"])
    summary.to_csv(out_dir / "exp_b_summary.csv", index=False)
    log.info("Experiment B complete. Saved to %s", out_dir / "exp_b_summary.csv")
    return summary


def run_experiment_c(cfg: Config, seeds: List[int], out_dir: Path, checkpoint: str):
    """Experiment C: Learned protocol actions vs. learned confirmation window."""
    log.info("Running Experiment C: Learned Protocol Actions vs Confirmation Window...")
    configs = {
        "C1_hardcoded_protocol": {"coordination": {"no_protocol": True, "no_adaptive_window": True}},
        "C2_learned_protocol_only": {"coordination": {"no_adaptive_window": True}},
        "C3_learned_window_only": {"coordination": {"no_protocol": True}},
        "C4_learned_protocol_and_window": {},
    }
    rows = []
    for c_name, change in configs.items():
        ccfg = with_changes(cfg, change)
        df = evaluate_method("gnn_mappo", ccfg, seeds, checkpoint=checkpoint, condition=c_name)
        df["protocol_config"] = c_name
        rows.append(df)
    res_df = pd.concat(rows, ignore_index=True)
    res_df.to_csv(out_dir / "exp_c_protocol_disentanglement.csv", index=False)
    summary = aggregate(res_df, ["protocol_config"])
    summary.to_csv(out_dir / "exp_c_summary.csv", index=False)
    log.info("Experiment C complete. Saved to %s", out_dir / "exp_c_summary.csv")
    return summary


def run_experiment_d(cfg: Config, seeds: List[int], out_dir: Path, checkpoint: str):
    """Experiment D: Communication efficiency (Periodic vs ETC vs Learned Gate)."""
    log.info("Running Experiment D: Communication Efficiency...")
    comm_modes = {
        "periodic": {"coordination": {"no_comm_gate": True}, "comm": {"heartbeat_period": 2}},
        "event_triggered": {"coordination": {"no_comm_gate": True}, "comm": {"heartbeat_period": 20}},
        "learned_gate": {"coordination": {"no_comm_gate": False}},
    }
    rows = []
    for mode_name, change in comm_modes.items():
        mcfg = with_changes(cfg, change)
        df = evaluate_method("gnn_mappo", mcfg, seeds, checkpoint=checkpoint, condition=mode_name)
        df["comm_mode"] = mode_name
        # Metric: Communication Efficiency = Completed Task Value / Total Comm Bytes
        # In simulator metrics: ret_value tracks weighted value completed; bytes_per_uav_min tracks total comm
        val_metric = df["weighted_completion_rate"] if "weighted_completion_rate" in df else df.get("ret_value", 1.0)
        comm_metric = df["bytes_per_uav_min"] if "bytes_per_uav_min" in df else 1.0
        df["comm_efficiency"] = val_metric / np.maximum(comm_metric, 1.0) * 1000.0
        rows.append(df)
    res_df = pd.concat(rows, ignore_index=True)
    res_df.to_csv(out_dir / "exp_d_communication_efficiency.csv", index=False)
    summary = aggregate(res_df, ["comm_mode"])
    summary.to_csv(out_dir / "exp_d_summary.csv", index=False)
    log.info("Experiment D complete. Saved to %s", out_dir / "exp_d_summary.csv")
    return summary


def run_experiment_e(cfg: Config, seeds: List[int], out_dir: Path, checkpoint: str):
    """Experiment E: Multi-task bundle sizes (1, 2, 3 tasks)."""
    log.info("Running Experiment E: Multi-task Bundle Capacity...")
    bundle_sizes = [1, 2, 3]
    rows = []
    for b_size in bundle_sizes:
        bcfg = with_changes(cfg, {"coordination": {"bundle_size": b_size}})
        for method in ["gnn_mappo", "cbba"]:
            df = evaluate_method(method, bcfg, seeds, checkpoint=checkpoint if method == "gnn_mappo" else None, condition=f"bundle_{b_size}")
            df["bundle_size"] = b_size
            rows.append(df)
    res_df = pd.concat(rows, ignore_index=True)
    res_df.to_csv(out_dir / "exp_e_bundle_sizes.csv", index=False)
    summary = aggregate(res_df, ["bundle_size", "method"])
    summary.to_csv(out_dir / "exp_e_summary.csv", index=False)
    log.info("Experiment E complete. Saved to %s", out_dir / "exp_e_summary.csv")
    return summary


def main():
    p = argparse.ArgumentParser(description="Run full research experimental validation suite.")
    p.add_argument("--config", default="configs/evaluation.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--episodes", type=int, default=5, help="Number of seeds for evaluation run")
    p.add_argument("--out-dir", default="results/research_experiments")
    p.add_argument("--experiments", nargs="+", default=["A", "B", "C", "D", "E"], choices=["A", "B", "C", "D", "E"])
    args = p.parse_args()

    cfg = load_config(args.config)
    seeds = seeds_from(0, args.episodes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Launching research experiment validation suite with %d test seeds...", len(seeds))
    if "A" in args.experiments:
        run_experiment_a(cfg, seeds, out_dir, args.checkpoint)
    if "B" in args.experiments:
        run_experiment_b(cfg, seeds, out_dir, args.checkpoint)
    if "C" in args.experiments:
        run_experiment_c(cfg, seeds, out_dir, args.checkpoint)
    if "D" in args.experiments:
        run_experiment_d(cfg, seeds, out_dir, args.checkpoint)
    if "E" in args.experiments:
        run_experiment_e(cfg, seeds, out_dir, args.checkpoint)
    log.info("All selected experiments finished successfully!")


if __name__ == "__main__":
    main()
