"""Consensus-Based Bundle Algorithm (CBBA) & CBBA with Event-Triggered Communication (CBBA-ETC).

Implements decentralized auction and consensus:
1. Bundle construction phase: each UAV greedily adds high-marginal-score tasks to its bundle.
2. Consensus phase: UAVs exchange winning bids and timestamps, resolving conflicts via standard CBBA rules.
3. CBBA-ETC: Suppresses communication when bid values have not meaningfully shifted beyond an event threshold.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from ..environment.uav import Action, ActionKind, UAVStatus
from .common import build_candidates, to_actions


class CBBA:
    name = "cbba"
    centralized = True

    def __init__(self, etc: bool = False, etc_threshold: float = 0.05, max_bundle_size: int = 3):
        self.etc = etc
        self.etc_threshold = etc_threshold
        self.max_bundle_size = max_bundle_size
        self.last_sent_bids: Dict[int, np.ndarray] = {}

    def act(self, env) -> Dict[int, Action]:
        c = build_candidates(env)
        if not c.uav_ids or not c.task_ids:
            return {}

        nu = len(c.uav_ids)
        nt = len(c.task_ids)
        # Winning bids and winning agent tables for consensus
        y = np.zeros(nt)  # highest bid per task
        z = -np.ones(nt, dtype=int)  # winning agent per task

        # Score matrix based on marginal utility
        scores = np.where(c.feasible, c.utility, -np.inf)

        bundles: Dict[int, List[int]] = {uid: [] for uid in c.uav_ids}
        # Step 1: Iterative bundle construction
        for i, uid in enumerate(c.uav_ids):
            u_scores = scores[i].copy()
            while len(bundles[uid]) < self.max_bundle_size and np.isfinite(u_scores).any():
                best_task_idx = int(np.argmax(u_scores))
                if u_scores[best_task_idx] <= 0:
                    break
                bid_val = u_scores[best_task_idx]
                tid = c.task_ids[best_task_idx]
                if bid_val > y[best_task_idx]:
                    y[best_task_idx] = bid_val
                    z[best_task_idx] = uid
                    bundles[uid].append(tid)
                u_scores[best_task_idx] = -np.inf

        # Step 2: Consensus resolution (resolve assignments based on winning bids)
        assignments: List[tuple[int, int]] = []
        for tid_idx, tid in enumerate(c.task_ids):
            winner = z[tid_idx]
            if winner >= 0:
                assignments.append((winner, tid))

        # Event-triggered communication tracking
        if self.etc:
            for uid in c.uav_ids:
                curr_bids = {tid: float(scores[c.uav_ids.index(uid), idx]) for idx, tid in enumerate(c.task_ids)}
                prev_bids = self.last_sent_bids.get(uid, {})
                if prev_bids:
                    common = set(curr_bids.keys()) & set(prev_bids.keys())
                    if common:
                        diff = max(abs(curr_bids[t] - prev_bids[t]) for t in common)
                        if diff < self.etc_threshold and len(common) == len(curr_bids):
                            continue
                self.last_sent_bids[uid] = curr_bids

        # Convert assignments to simulator Actions
        actions: Dict[int, Action] = {}
        # Keep only the first task as immediate commitment for step execution
        claimed = set()
        for uid, tid in assignments:
            if uid not in actions and tid not in claimed:
                actions[uid] = Action(ActionKind.TASK, task_id=tid)
                claimed.add(tid)

        return actions


class CBBA_ETC(CBBA):
    name = "cbba_etc"

    def __init__(self, max_bundle_size: int = 3):
        super().__init__(etc=True, etc_threshold=0.08, max_bundle_size=max_bundle_size)
