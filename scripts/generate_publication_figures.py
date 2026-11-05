#!/usr/bin/env python
"""Generate comprehensive publication-grade plots comparing all algorithms across S0-S4 scenarios."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling setup
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Load 30-seed robustness benchmark results
df_sum = pd.read_csv("results/robustness_benchmark_30seeds/robustness_summary.csv")
df_ep = pd.read_csv("results/robustness_benchmark_30seeds/robustness_episodes.csv")
df_deg = pd.read_csv("results/robustness_benchmark_30seeds/robustness_degradation_deltas.csv")

methods = ["gnn_mappo_proposed", "gnn_mappo_original", "cbba", "cbba_etc", "greedy", "hungarian"]
method_labels = {
    "gnn_mappo_proposed": "GNN-MAPPO (Proposed)",
    "gnn_mappo_original": "GNN-MAPPO (Ablation)",
    "cbba": "CBBA",
    "cbba_etc": "CBBA-ETC",
    "greedy": "Greedy (Cent.)",
    "hungarian": "Hungarian (Global)"
}
colors = {
    "gnn_mappo_proposed": "#1f77b4",  # Blue
    "gnn_mappo_original": "#aec7e8",  # Light blue
    "cbba": "#ff7f0e",                # Orange
    "cbba_etc": "#ffbb78",            # Light orange
    "greedy": "#2ca02c",              # Green
    "hungarian": "#9467bd",           # Purple
}
scenarios = ["S0_normal", "S1_degraded_comm", "S2_partition", "S3_uav_failure", "S4_combined_stress"]
scenario_labels = ["S0: Normal", "S1: Degraded Comm", "S2: Partition", "S3: UAV Failure", "S4: Combined Stress"]

# ==============================================================================
# Plot 1: 4-Panel Multi-Metric Robustness Comparison
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 11))

# Panel 1: Weighted Completion Rate
ax = axs[0, 0]
x = np.arange(len(scenarios))
width = 0.13
for idx, m in enumerate(methods):
    sub = df_sum[df_sum["method"] == m].set_index("scenario").loc[scenarios]
    means = sub["weighted_completion_rate_mean"].values * 100
    yerr_low = (sub["weighted_completion_rate_mean"].values - sub["weighted_completion_rate_ci_low"].values) * 100
    yerr_high = (sub["weighted_completion_rate_ci_high"].values - sub["weighted_completion_rate_mean"].values) * 100
    ax.bar(x + idx * width - (len(methods)/2 - 0.5) * width, means, width,
           label=method_labels[m], color=colors[m], yerr=[yerr_low, yerr_high], capsize=3, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Weighted Task Completion (%)")
ax.set_title("A. Task Completion across Operating Scenarios (95% CI)")
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, rotation=15)
ax.set_ylim(35, 95)
ax.legend(frameon=True, loc="upper right")

# Panel 2: Duplicate Rate
ax = axs[0, 1]
for idx, m in enumerate(methods):
    sub = df_sum[df_sum["method"] == m].set_index("scenario").loc[scenarios]
    means = sub["duplicate_rate_mean"].values * 100
    ax.bar(x + idx * width - (len(methods)/2 - 0.5) * width, means, width,
           label=method_labels[m], color=colors[m], edgecolor="black", linewidth=0.5)
ax.set_ylabel("Duplicate Conflict Rate (%)")
ax.set_title("B. Reservation Conflict Rate (Lower is Better)")
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, rotation=15)
ax.set_ylim(0, 30)

# Panel 3: Communication Byte Traffic (Log Scale)
ax = axs[1, 0]
for idx, m in enumerate(methods):
    sub = df_sum[df_sum["method"] == m].set_index("scenario").loc[scenarios]
    means = sub["bytes_per_uav_min_mean"].values
    ax.bar(x + idx * width - (len(methods)/2 - 0.5) * width, means, width,
           label=method_labels[m], color=colors[m], edgecolor="black", linewidth=0.5)
ax.set_ylabel("Bandwidth (Bytes / UAV / min)")
ax.set_yscale("log")
ax.set_title("C. Communication Overhead (Logarithmic Scale)")
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, rotation=15)

# Panel 4: Energy per Completed Task
ax = axs[1, 1]
for idx, m in enumerate(methods):
    sub = df_sum[df_sum["method"] == m].set_index("scenario").loc[scenarios]
    means = sub["energy_per_task_mean"].values
    ax.bar(x + idx * width - (len(methods)/2 - 0.5) * width, means, width,
           label=method_labels[m], color=colors[m], edgecolor="black", linewidth=0.5)
ax.set_ylabel("Energy per Completed Task (kJ)")
ax.set_title("D. Operational Energy Efficiency (kJ / Task)")
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, rotation=15)
ax.set_ylim(0, 15)

plt.tight_layout()
fig.savefig(out_dir / "benchmark_multi_metric_comparison.png", dpi=300)
plt.close(fig)

# ==============================================================================
# Plot 2: Robustness Degradation Profiles vs Normal Baseline
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
stress_scenarios = ["S1_degraded_comm", "S2_partition", "S3_uav_failure", "S4_combined_stress"]
stress_labels = ["S1: Degraded Comm", "S2: Partition", "S3: UAV Failure", "S4: Combined Stress"]
x_s = np.arange(len(stress_scenarios))

for m in methods:
    sub = df_deg[df_deg["method"] == m].set_index("scenario").loc[stress_scenarios]
    deltas = sub["delta_completion_mean"].values * 100
    yerr_low = (sub["delta_completion_mean"].values - sub["delta_completion_ci_low"].values) * 100
    yerr_high = (sub["delta_completion_ci_high"].values - sub["delta_completion_mean"].values) * 100
    ax.plot(x_s, deltas, marker="o", linewidth=2.2, markersize=7, label=method_labels[m], color=colors[m])
    ax.fill_between(x_s, deltas - yerr_low, deltas + yerr_high, color=colors[m], alpha=0.15)

ax.set_ylabel(r"Task Completion Degradation $\Delta C_s = C_{\mathrm{normal}} - C_s$ (pp)")
ax.set_title("Resilience Profile: Performance Drop Relative to Normal Scenario (Lower is Better)")
ax.set_xticks(x_s)
ax.set_xticklabels(stress_labels)
ax.set_ylim(-5, 45)
ax.axhline(0, color="gray", linestyle=":", linewidth=1)
ax.legend(frameon=True, loc="upper left")

plt.tight_layout()
fig.savefig(out_dir / "robustness_degradation_profile.png", dpi=300)
plt.close(fig)

# ==============================================================================
# Plot 3: Cumulative Completion CDF & Distribution Across Seeds
# ==============================================================================
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Distribution in Combined Stress (S4)
ax = axs[0]
s4_df = df_ep[df_ep["scenario"] == "S4_combined_stress"]
bp_data = [s4_df[s4_df["method_label"] == m]["weighted_completion_rate"].values * 100 for m in methods]
bplot = ax.boxplot(bp_data, patch_artist=True, labels=[method_labels[m] for m in methods], medianprops=dict(color="black", linewidth=1.5))
for patch, m in zip(bplot["boxes"], methods):
    patch.set_facecolor(colors[m])
    patch.set_alpha(0.8)
ax.set_ylabel("Weighted Completion (%)")
ax.set_title("Distribution under Compound Stress (S4)")
ax.set_xticklabels([method_labels[m] for m in methods], rotation=25, ha="right")

# Subplot 2: Bandwidth vs Completion Trade-off Frontier
ax = axs[1]
for m in methods:
    sub = df_sum[df_sum["method"] == m]
    comp = sub["weighted_completion_rate_mean"].mean() * 100
    traffic = sub["bytes_per_uav_min_mean"].mean()
    ax.scatter(traffic, comp, s=180, color=colors[m], edgecolors="black", linewidth=1.2, label=method_labels[m], zorder=5)
    ax.annotate(method_labels[m], (traffic, comp), textcoords="offset points", xytext=(8, 4), fontsize=9)

ax.set_xscale("log")
ax.set_xlabel("Average Bandwidth (Bytes / UAV / min, Log Scale)")
ax.set_ylabel("Mean Completion Across All Scenarios (%)")
ax.set_title("Pareto Efficiency: Coordination Quality vs Channel Overhead")
ax.set_ylim(60, 85)

plt.tight_layout()
fig.savefig(out_dir / "stress_distribution_and_pareto.png", dpi=300)
plt.close(fig)

print("All publication-grade comparison plots successfully generated in results/figures/")
