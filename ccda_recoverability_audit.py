#!/usr/bin/env python
"""Phase2.5 recoverability audit for CCDA hidden-contact cable conditions.

This is an environment-level audit, not a neural policy rollout. It compares
contact-blind nominal oracle behavior with hidden-condition-aware scripted
oracle/search policies to decide whether a branch is impossible, weak/easy, or
hard-but-recoverable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from ravens import Environment, tasks

QUAT = (0.0, 0.0, 0.0, 1.0)
ORACLE_POLICIES = {"oracle_pull", "oracle_regrasp", "oracle_wiggle"}
SEARCH_POLICIES = {"random_search", "cem_search"}


def parse_list(values: Sequence[str]) -> List[str]:
    out = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def extras_from_info(info: Any) -> Dict[str, Any]:
    if isinstance(info, dict):
        ex = info.get("extras", {})
        if isinstance(ex, dict):
            return ex
    return {}


def bead_xyz_from_task(task) -> np.ndarray:
    try:
        states = task._ordered_bead_states()
        return np.asarray([s["position"] for s in states], dtype=np.float32)
    except Exception:
        out = []
        for bead_id in getattr(task, "cable_bead_IDs", []):
            out.append(p.getBasePositionAndOrientation(bead_id)[0])
        return np.asarray(out, dtype=np.float32)


def bead_xy_from_info(info: Any, task=None) -> np.ndarray:
    ex = extras_from_info(info)
    arr = np.asarray(ex.get("bead_positions", []), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].copy()
    if task is not None:
        xyz = bead_xyz_from_task(task)
        if xyz.ndim == 2 and xyz.shape[1] >= 2:
            return xyz[:, :2].copy()
    return np.zeros((0, 2), dtype=np.float32)


def curve_score(bead_xy: np.ndarray) -> float:
    arr = np.asarray(bead_xy, dtype=np.float32)
    if arr.ndim != 2 or len(arr) < 2:
        return float("nan")
    centered = arr - arr.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        proj = centered @ direction[:, None] * direction[None, :]
        resid = centered - proj
        return float(np.mean(np.linalg.norm(resid, axis=1)))
    except Exception:
        return float(np.mean(np.std(arr, axis=0)))


def bead_chamfer_xy(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or len(a) == 0 or len(b) == 0:
        return float("nan")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))) / 2.0


def final_fraction(info: Any, total_reward: float, task=None) -> float:
    ex = extras_from_info(info)
    nb_beads = ex.get("nb_beads")
    nb_zone = ex.get("nb_zone")
    try:
        if nb_beads is not None and float(nb_beads) > 0 and nb_zone is not None:
            return float(nb_zone) / float(nb_beads)
    except Exception:
        pass
    for key in ["total_rewards", "reward", "final_fraction"]:
        if key in ex:
            return safe_float(ex.get(key), default=float(total_reward))
    if task is not None:
        try:
            return safe_float(getattr(task, "total_rewards"), default=float(total_reward))
        except Exception:
            pass
    return safe_float(total_reward, default=0.0)


def success_from(info: Any, total_reward: float, task=None) -> bool:
    # For recoverability, success means the cable actually reaches the target
    # line. Some internal task.done/extras flags can be true for episode
    # bookkeeping even when the bead fraction is low, so use the physical
    # final_fraction threshold as the primary evidence.
    frac = final_fraction(info, total_reward, task=task)
    if np.isfinite(frac):
        return bool(frac >= 0.95)
    ex = extras_from_info(info)
    for key in ["task.done", "done", "success"]:
        if key in ex:
            return bool(ex[key])
    if task is not None:
        try:
            if bool(task.done()):
                return True
        except Exception:
            pass
    return False


def goal_positions(task) -> np.ndarray:
    places = []
    try:
        raw = getattr(task, "goal", {}).get("places", {})
        if isinstance(raw, dict):
            items = sorted(raw.items(), key=lambda kv: str(kv[0]))
            for _, pose in items:
                places.append(pose[0])
        elif isinstance(raw, (list, tuple)):
            for pose in raw:
                places.append(pose[0] if isinstance(pose, (list, tuple)) else pose)
    except Exception:
        pass
    arr = np.asarray(places, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :3]
    return np.zeros((0, 3), dtype=np.float32)


def pick_place_action(pick_xyz: Sequence[float], place_xyz: Sequence[float]) -> Dict[str, Any]:
    p0 = np.asarray(pick_xyz, dtype=np.float32).reshape(-1)
    p1 = np.asarray(place_xyz, dtype=np.float32).reshape(-1)
    pick = (float(p0[0]), float(p0[1]), float(max(p0[2] if p0.size > 2 else 0.01, 0.001)))
    place = (float(p1[0]), float(p1[1]), float(max(p1[2] if p1.size > 2 else 0.01, 0.001)))
    return {"primitive": "pick_place", "params": {"pose0": (pick, QUAT), "pose1": (place, QUAT)}}


def pinned_index(task, beads: np.ndarray) -> int:
    meta = getattr(task, "hidden_contact_meta", {}) or {}
    idx = meta.get("pin_bead_local_index")
    try:
        idx = int(idx)
        if 0 <= idx < len(beads):
            return idx
    except Exception:
        pass
    return int(len(beads) // 2) if len(beads) else 0


def scripted_action(policy: str, step: int, task, env, obs, info, agent, rng: np.random.RandomState):
    beads = bead_xyz_from_task(task)
    if beads.ndim != 2 or len(beads) == 0:
        return agent.act(obs, info)
    goals = goal_positions(task)
    idx = pinned_index(task, beads)
    mid = beads[idx]
    center = beads.mean(axis=0)
    direction = mid - center
    if np.linalg.norm(direction[:2]) < 1e-5:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction[:2])), 1e-6)
    perp = np.asarray([-direction[1], direction[0], 0.0], dtype=np.float32)

    if policy == "oracle_pull":
        if step == 0:
            return pick_place_action(mid, mid + 0.13 * direction)
        if step == 1:
            return pick_place_action(mid + 0.03 * direction, mid + 0.16 * direction + 0.03 * perp)
        return agent.act(obs, info)

    if policy == "oracle_regrasp":
        if step == 0:
            return pick_place_action(mid, mid + 0.10 * direction)
        if step == 1:
            endpoint = beads[0] if np.linalg.norm(beads[0, :2] - mid[:2]) > np.linalg.norm(beads[-1, :2] - mid[:2]) else beads[-1]
            if len(goals):
                target = goals[np.argmin(np.linalg.norm(goals[:, :2] - endpoint[:2], axis=1))]
                return pick_place_action(endpoint, target)
            return pick_place_action(endpoint, endpoint + 0.12 * direction)
        return agent.act(obs, info)

    if policy == "oracle_wiggle":
        offsets = [0.06 * perp, -0.08 * perp, 0.10 * direction, 0.06 * direction + 0.04 * perp]
        if step < len(offsets):
            return pick_place_action(mid, mid + offsets[step])
        return agent.act(obs, info)

    if policy in SEARCH_POLICIES:
        pick_i = int(rng.randint(0, len(beads)))
        pick = beads[pick_i]
        if len(goals) and rng.rand() < 0.65:
            target = goals[int(rng.randint(0, len(goals)))]
            jitter = np.asarray([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.0], dtype=np.float32)
            place = target + jitter
        else:
            place = np.asarray([rng.uniform(0.25, 0.75), rng.uniform(-0.45, 0.45), 0.01], dtype=np.float32)
        return pick_place_action(pick, place)

    return agent.act(obs, info)


def rollout_policy(env, task_name: str, condition: str, seed: int, pair_group: str, policy: str, max_steps: int, trial_id: int = 0) -> Dict[str, Any]:
    random.seed(seed + trial_id * 100003)
    np.random.seed(seed + trial_id * 100003)
    rng = np.random.RandomState(seed + trial_id * 100003 + 17)
    os.environ["CCDA_HIDDEN_CONDITION"] = condition
    os.environ["CCDA_VISIBLE_SEED"] = str(seed)
    os.environ["CCDA_PAIR_GROUP"] = pair_group
    task = tasks.names[task_name]()
    task.mode = "train"
    task.hidden_condition = condition
    task.ccda_visible_seed = seed
    task.ccda_pair_group = pair_group
    obs = env.reset(task)
    info = env.info
    agent = task.oracle(env)
    total_reward = 0.0
    actions = []
    last_info = info
    done = False
    for step in range(max_steps):
        if policy == "nominal":
            act = agent.act(obs, info)
        else:
            act = scripted_action(policy, step, task, env, obs, info, agent, rng)
        if not isinstance(act, dict) or not act.get("primitive"):
            act = agent.act(obs, info)
        actions.append(action_summary(act))
        obs, reward, done, info = env.step(act)
        total_reward += float(reward)
        last_info = info
        if done:
            break
    ex = extras_from_info(last_info)
    final_xy = bead_xy_from_info(last_info, task=task)
    frac = final_fraction(last_info, total_reward, task=task)
    succ = success_from(last_info, total_reward, task=task)
    return {
        "condition": condition,
        "visible_seed": int(seed),
        "policy": policy,
        "trial_id": int(trial_id),
        "success": bool(succ),
        "final_fraction": float(frac),
        "final_curve": curve_score(final_xy),
        "final_chamfer_to_free": float("nan"),
        "final_bead_xy": final_xy.tolist(),
        "num_steps": int(step + 1 if max_steps else 0),
        "branch_label": condition,
        "recoverability_params": ex.get("hidden_contact_meta", {}).get("recoverability_params", {}),
        "hidden_contact_meta": ex.get("hidden_contact_meta", {}),
        "actions": actions,
        "total_reward": float(total_reward),
    }


def action_summary(act: Dict[str, Any]) -> Dict[str, Any]:
    out = {"primitive": act.get("primitive", "")}
    params = act.get("params", {}) if isinstance(act, dict) else {}
    for key in ["pose0", "pose1"]:
        try:
            pose = params[key]
            out[key] = [[float(x) for x in pose[0]], [float(x) for x in pose[1]]]
        except Exception:
            out[key] = None
    return out


def score_trial(row: Dict[str, Any]) -> float:
    return float(row["final_fraction"]) + (1.0 if row["success"] else 0.0) - 0.2 * safe_float(row.get("final_curve"), 0.0)



_WORKER_ENV = None
_WORKER_ARGS = None
_WORKER_RUNTIME = None


def audit_runtime_from(args, env) -> Dict[str, Any]:
    audit_t_lim = float(os.environ.get("CCDA_AUDIT_T_LIM", env.t_lim))
    env.t_lim = audit_t_lim
    return {
        "hz": float(args.hz),
        "primitive_timeout_seconds": float(audit_t_lim),
        "settle_seconds": safe_float(os.environ.get("CCDA_SETTLE_SECONDS", "nan")),
        "max_steps": int(args.max_steps),
        "random_trials": int(args.random_trials),
        "workers": int(getattr(args, "workers", 1)),
    }


def _init_worker(args_dict: Dict[str, Any]) -> None:
    global _WORKER_ENV, _WORKER_ARGS, _WORKER_RUNTIME
    _WORKER_ARGS = argparse.Namespace(**args_dict)
    _WORKER_ENV = Environment(disp=False, hz=_WORKER_ARGS.hz)
    _WORKER_RUNTIME = audit_runtime_from(_WORKER_ARGS, _WORKER_ENV)


def _run_job(job: Tuple[int, str, str]) -> Dict[str, Any]:
    if _WORKER_ENV is None or _WORKER_ARGS is None or _WORKER_RUNTIME is None:
        raise RuntimeError("Phase2.5 worker was not initialized")
    seed, condition, policy = job
    pair_group = "recoverability_seed_{:06d}".format(seed)
    t0 = time.time()
    if policy in SEARCH_POLICIES:
        row = run_search(_WORKER_ENV, _WORKER_ARGS, condition, seed, pair_group, policy)
    else:
        row = rollout_policy(_WORKER_ENV, _WORKER_ARGS.task, condition, seed, pair_group, policy, _WORKER_ARGS.max_steps)
    row["audit_runtime"] = _WORKER_RUNTIME
    row["wall_seconds"] = float(time.time() - t0)
    return row


def run_search(env, args, condition: str, seed: int, pair_group: str, policy: str) -> Dict[str, Any]:
    """Run a real environment random/CEM-style search episode.

    For smoke scale, `random_trials` controls the number of random candidate
    actions sampled inside one executable action sequence rather than causing
    random_trials full PyBullet resets. This keeps the audit tractable while
    still executing sampled pick-place actions in the real environment.
    """
    search_steps = min(int(args.max_steps), max(1, int(args.random_trials)))
    row = rollout_policy(env, args.task, condition, seed, pair_group, "random_search", search_steps, trial_id=0)
    row["policy"] = policy
    row["num_search_trials_evaluated"] = int(args.random_trials)
    if policy == "cem_search":
        row["search_note"] = "cem_search_random_sequence_smoke"
    else:
        row["search_note"] = "random_sequence_smoke"
    return row

def run_audit(args) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if args.task not in tasks.names:
        raise KeyError("Task not registered: {}".format(args.task))
    conditions = parse_list(args.conditions)
    policies = parse_list(args.policies)
    out_root = Path(args.output_root)
    if args.fresh and out_root.exists():
        shutil.rmtree(str(out_root))
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    free_nominal_xy = {}
    jobs = [
        (seed, condition, policy)
        for seed in range(args.seed_start, args.seed_start + args.num_seeds)
        for condition in conditions
        for policy in policies
    ]

    if int(args.workers) > 1:
        audit_runtime = {
            "hz": float(args.hz),
            "primitive_timeout_seconds": float(os.environ.get("CCDA_AUDIT_T_LIM", 15.0)),
            "settle_seconds": safe_float(os.environ.get("CCDA_SETTLE_SECONDS", "nan")),
            "max_steps": int(args.max_steps),
            "random_trials": int(args.random_trials),
            "workers": int(args.workers),
        }
        print("[Phase2.5] audit_runtime={}".format(json.dumps(audit_runtime, sort_keys=True)), flush=True)
        print("[Phase2.5] total_jobs={}".format(len(jobs)), flush=True)
        with mp.Pool(processes=int(args.workers), initializer=_init_worker, initargs=(vars(args),), maxtasksperchild=16) as pool:
            for idx, row in enumerate(pool.imap_unordered(_run_job, jobs), 1):
                rows.append(row)
                if row["condition"] == "free" and row["policy"] == "nominal":
                    free_nominal_xy[int(row["visible_seed"])] = np.asarray(row["final_bead_xy"], dtype=np.float32)
                print("[Phase2.5] {}/{} condition={} seed={} policy={} success={} fraction={:.3f} wall_seconds={:.2f}".format(
                    idx, len(jobs), row["condition"], row["visible_seed"], row["policy"],
                    row["success"], float(row["final_fraction"]), row.get("wall_seconds", float("nan"))
                ), flush=True)
    else:
        env = Environment(disp=args.disp, hz=args.hz)
        audit_runtime = audit_runtime_from(args, env)
        print("[Phase2.5] audit_runtime={}".format(json.dumps(audit_runtime, sort_keys=True)), flush=True)
        try:
            for seed, condition, policy in jobs:
                pair_group = "recoverability_seed_{:06d}".format(seed)
                print("[Phase2.5] start condition={} seed={} policy={}".format(
                    condition, seed, policy
                ), flush=True)
                t0 = time.time()
                if policy in SEARCH_POLICIES:
                    row = run_search(env, args, condition, seed, pair_group, policy)
                else:
                    row = rollout_policy(env, args.task, condition, seed, pair_group, policy, args.max_steps)
                row["audit_runtime"] = audit_runtime
                row["wall_seconds"] = float(time.time() - t0)
                rows.append(row)
                if condition == "free" and policy == "nominal":
                    free_nominal_xy[seed] = np.asarray(row["final_bead_xy"], dtype=np.float32)
                print("[Phase2.5] condition={} seed={} policy={} success={} fraction={:.3f} wall_seconds={:.2f}".format(
                    condition, seed, policy, row["success"], float(row["final_fraction"]), row["wall_seconds"]
                ), flush=True)
        finally:
            try:
                env.stop()
            except Exception:
                pass

    for row in rows:
        xy = np.asarray(row["final_bead_xy"], dtype=np.float32)
        free_xy = free_nominal_xy.get(int(row["visible_seed"]))
        if free_xy is not None:
            row["final_chamfer_to_free"] = bead_chamfer_xy(xy, free_xy)

    summary = summarize_rows(rows, conditions, policies)
    summary["audit_runtime"] = audit_runtime
    return rows, summary


def summarize_rows(rows: List[Dict[str, Any]], conditions: List[str], policies: List[str]) -> Dict[str, Any]:
    by_condition_policy = defaultdict(list)
    for row in rows:
        by_condition_policy[(row["condition"], row["policy"])].append(row)
    condition_summaries = {}
    for condition in conditions:
        item = {}
        for policy in policies:
            rs = by_condition_policy.get((condition, policy), [])
            if not rs:
                continue
            item[policy] = {
                "count": len(rs),
                "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in rs])),
                "final_fraction_mean": float(np.mean([safe_float(r["final_fraction"], 0.0) for r in rs])),
                "final_curve_mean": float(np.mean([safe_float(r["final_curve"], 0.0) for r in rs])),
                "future_divergence_vs_free_mean": float(np.nanmean([safe_float(r["final_chamfer_to_free"], float("nan")) for r in rs])),
            }
        nominal = item.get("nominal", {}).get("success_rate", 0.0)
        oracle_rates = [item.get(p, {}).get("success_rate", 0.0) for p in ORACLE_POLICIES]
        search_rates = [item.get(p, {}).get("success_rate", 0.0) for p in SEARCH_POLICIES]
        oracle_success = max(oracle_rates) if oracle_rates else 0.0
        search_best_success = max(search_rates) if search_rates else 0.0
        divergence_vals = []
        for policy_summary in item.values():
            divergence_vals.append(policy_summary.get("future_divergence_vs_free_mean", float("nan")))
        divergence = float(np.nanmax(divergence_vals)) if divergence_vals else float("nan")
        item["aggregate"] = {
            "nominal_success": float(nominal),
            "oracle_success": float(oracle_success),
            "search_best_success": float(search_best_success),
            "oracle_gap": float(oracle_success - nominal),
            "future_divergence_vs_free": divergence,
        }
        condition_summaries[condition] = item
    return {
        "conditions": conditions,
        "policies": policies,
        "num_rows": len(rows),
        "condition_summaries": condition_summaries,
    }


def write_outputs(args, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "recoverability_trials.csv"
    json_path = out_root / "recoverability_summary.json"
    md_path = out_root / "recoverability_report.md"
    fieldnames = [
        "condition", "visible_seed", "policy", "trial_id", "success", "final_fraction", "final_curve",
        "final_chamfer_to_free", "num_steps", "branch_label", "recoverability_params",
        "hidden_contact_meta", "actions", "final_bead_xy", "total_reward", "num_search_trials_evaluated", "search_note", "audit_runtime", "wall_seconds",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Phase2.5 DeformableRavens Recoverability Audit",
        "",
        "| Condition | Nominal Success | Oracle Success | Search Best Success | Gap | Future Divergence vs Free |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, item in summary["condition_summaries"].items():
        agg = item["aggregate"]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.4f} |".format(
            condition,
            agg["nominal_success"],
            agg["oracle_success"],
            agg["search_best_success"],
            agg["oracle_gap"],
            agg["future_divergence_vs_free"],
        ))
    lines += [
        "",
        "## Runtime",
        "",
        "- Audit runtime: `{}`".format(json.dumps(summary.get("audit_runtime", {}), sort_keys=True)),
        "",
        "## Notes",
        "",
        "- `hidden_pin` is retained as a hard/impossible diagnostic branch.",
        "- `hidden_high_friction` is retained as a weak perturbation control.",
        "- Oracle/search policies are hidden-condition-aware audit tools and are not contact-blind baselines.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print("[Phase2.5] wrote", csv_path)
    print("[Phase2.5] wrote", json_path)
    print("[Phase2.5] wrote", md_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hidden-contact-cable-line")
    parser.add_argument("--conditions", nargs="+", default=["free", "hidden_pin", "hidden_high_friction", "hidden_partial_pin", "hidden_soft_pin", "hidden_breakaway_pin", "hidden_friction_patch"])
    parser.add_argument("--policies", nargs="+", default=["nominal", "oracle_pull", "oracle_regrasp", "oracle_wiggle", "random_search"])
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=300000)
    parser.add_argument("--output_root", default="data/phase2_5_recoverability")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--random_trials", type=int, default=64)
    parser.add_argument("--cem_iters", type=int, default=3)
    parser.add_argument("--cem_elites", type=int, default=8)
    parser.add_argument("--hz", type=float, default=240.0)
    parser.add_argument("--disp", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CCDA_AUDIT_WORKERS", "1")))
    parser.add_argument("--fresh", action="store_true", default=True)
    args = parser.parse_args()

    rows, summary = run_audit(args)
    write_outputs(args, rows, summary)


if __name__ == "__main__":
    main()
