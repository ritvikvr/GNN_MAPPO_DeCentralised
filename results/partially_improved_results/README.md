# Partially Improved Results Archive

This folder stores the graphical benchmarks, simulation trajectory animations, and high-definition 3D videos generated from the updated protocol and communication guardrail evaluation.

---

## 1. Directory Contents

### Graphical Figures (300 DPI)
- `benchmark_multi_metric_comparison.png`: Four-panel multi-metric evaluation across all 6 algorithms and 5 stress scenarios (Task Completion, Duplicate Rate, Bandwidth, and Energy Efficiency with 95% bootstrap CIs).
- `robustness_degradation_profile.png`: Paired performance degradation curves ($\Delta C_s = C_{\mathrm{normal}} - C_s$) across S1 through S4.
- `stress_distribution_and_pareto.png`: Boxplot variance under compound stress (S4) and multi-scenario Pareto coordination efficiency frontier.

### Rendered Simulations & Videos
- `gnn_mappo_improved_simulation.mp4`: High-definition 3D isometric simulation video (1500x1500px, 20 FPS, 284 frames) illustrating decentralized multi-UAV trajectory negotiation and task execution (85.2% completion).
- `simulation_run.gif`: 2D multi-agent tactical simulation animation showing flight trajectories, dynamic task claims, and conflict-free arrivals (97.8% completion).
- `uav_simulation_3d_isometric.mp4`: Isometric animation of fleet coordination.

---

## 2. Experimental Origin
- **Source Checkpoint**: `checkpoints/best.pt` (MD5: `e3e21098ef617ab4cef6e296bb3d6519`)
- **Protocol Guardrail**: Mutual DEFEND deterministic tie-breaking enforced in `src/uav_allocation/coordination/reservations.py`.
- **Sample Size**: 30 independent matched test seeds ($n = 30$, 900 total simulated missions).
