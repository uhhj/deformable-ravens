#!/usr/bin/env python
"""Record continuous PyBullet execution videos for Phase1.1 CCDA cable rollouts."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from ccda_continuous_video import CCDAVideoRecorder
from ravens import Environment, tasks


DEFAULT_CONDITIONS = [
    "free",
    "hidden_pin",
    "hidden_high_friction",
    "hidden_side_jam",
]


def parse_conditions(raw: List[str]) -> List[str]:
    out = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def make_report(path: Path, summary: Dict) -> None:
    lines = []
    lines.append("# Phase1.1 Continuous PyBullet Rollout Video Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report summarizes continuous PyBullet execution videos for the hidden-contact cable task.")
    lines.append("Unlike the earlier action-step dataset replay videos, these MP4 files are recorded while `Environment.pick_place()` is executing, so the UR5 approach, descent, grasp, movement, release, and settle phases are visible.")
    lines.append("")
    lines.append("## Camera")
    lines.append("")
    cam = summary.get("camera", {})
    lines.append("- View: side-top debug camera, not the policy RGB-D camera")
    lines.append("- Camera position: `{}`".format(cam.get("camera_position")))
    lines.append("- Camera target: `{}`".format(cam.get("camera_target")))
    lines.append("- Image size: `{}x{}`".format(cam.get("image_width"), cam.get("image_height")))
    lines.append("")
    lines.append("## Videos")
    lines.append("")
    lines.append("| Condition | Seed | Status | Action Steps | Frames | Reward | Done | Path |")
    lines.append("|---|---:|---|---:|---:|---:|---|---|")
    for item in summary.get("videos", []):
        lines.append(
            "| {condition} | {seed} | {status} | {steps} | {frames} | {reward:.4f} | {done} | `{path}` |".format(
                condition=item.get("condition"),
                seed=item.get("visible_seed"),
                status=item.get("status"),
                steps=item.get("executed_action_steps", 0),
                frames=item.get("frames_written", 0),
                reward=float(item.get("total_reward", 0.0)),
                done=item.get("done"),
                path=item.get("path"),
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- These videos are intended for qualitative inspection of continuous primitive execution and contact timing.")
    lines.append("- They complement, but do not replace, the Phase1 RGB-D checks, bead trajectory overlays, and Phase2 threshold-based CCDA pair mining.")
    lines.append("- This report does not claim that CCDA has been fully proven.")
    lines.append("")
    path.write_text("\n".join(lines))


def initial_observation(agent, env, obs, info):
    act = agent.act(obs, info)
    obs, reward, done, info = env.step(act)
    return obs, reward, done, info


def record_condition(env, args, condition: str) -> Dict:
    random.seed(args.visible_seed)
    np.random.seed(args.visible_seed)

    os.environ["CCDA_HIDDEN_CONDITION"] = condition
    os.environ["CCDA_VISIBLE_SEED"] = str(args.visible_seed)
    os.environ["CCDA_PAIR_GROUP"] = "visible_seed_{:06d}".format(args.visible_seed)

    task = tasks.names[args.task]()
    task.mode = "train"
    task.hidden_condition = condition
    task.ccda_visible_seed = args.visible_seed
    task.ccda_pair_group = os.environ["CCDA_PAIR_GROUP"]

    obs = env.reset(task)
    agent = task.oracle(env)
    info = env.info

    out_path = Path(args.outdir) / "seed_{}_{}.mp4".format(args.visible_seed, condition)
    recorder = CCDAVideoRecorder(
        path=str(out_path),
        condition=condition,
        visible_seed=args.visible_seed,
        fps=args.fps,
        stride=args.stride,
        image_width=args.image_width,
        image_height=args.image_height,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        camera_up=args.camera_up,
    )

    total_reward = 0.0
    done = False
    executed = 0
    status = "written"
    env.set_ccda_video_recorder(recorder)

    try:
        recorder.set_action_step(0)
        recorder.record("reset")
        obs, reward, done, info = initial_observation(agent, env, obs, info)
        total_reward += float(reward)

        max_steps = min(int(args.max_steps), int(task.max_steps))
        for action_step in range(max_steps):
            if done:
                break

            recorder.set_action_step(action_step)
            act = agent.act(obs, info)
            if not act.get("primitive"):
                obs, reward, done, info = env.step(act)
                total_reward += float(reward)
                continue

            recorder.record("action_start")
            obs, reward, done, info = env.step(act)
            total_reward += float(reward)
            executed += 1
            recorder.record("action_done")
    except Exception as exc:
        status = "failed"
        raise
    finally:
        env.set_ccda_video_recorder(None)
        recorder.close()

    item = recorder.summary()
    item.update(
        {
            "task": args.task,
            "condition": condition,
            "visible_seed": args.visible_seed,
            "status": status,
            "path": str(out_path),
            "executed_action_steps": executed,
            "total_reward": total_reward,
            "done": bool(done),
        }
    )
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hidden-contact-cable-line")
    parser.add_argument("--visible_seed", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--hz", type=float, default=240.0)
    parser.add_argument("--outdir", default="/data/state_diff2/reports/phase1_continuous_videos")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--image_width", type=int, default=960)
    parser.add_argument("--image_height", type=int, default=720)
    parser.add_argument("--camera_position", nargs=3, type=float, default=[0.55, -0.75, 0.45])
    parser.add_argument("--camera_target", nargs=3, type=float, default=[0.50, 0.00, 0.02])
    parser.add_argument("--camera_up", nargs=3, type=float, default=[0, 0, 1])
    parser.add_argument("--disp", action="store_true")
    args = parser.parse_args()

    if args.task not in tasks.names:
        raise KeyError("Task not registered: {}".format(args.task))

    conditions = parse_conditions(args.conditions)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    report_root = outdir.parent

    env = Environment(disp=args.disp, hz=args.hz)
    videos = []
    try:
        for condition in conditions:
            print("[Phase1.1] recording condition={} seed={}".format(condition, args.visible_seed))
            item = record_condition(env, args, condition)
            videos.append(item)
            print("[Phase1.1] video summary:", item)
    finally:
        env.stop()

    summary = {
        "task": args.task,
        "visible_seed": args.visible_seed,
        "conditions": conditions,
        "hz": args.hz,
        "max_steps": args.max_steps,
        "fps": args.fps,
        "stride": args.stride,
        "outdir": str(outdir),
        "camera": {
            "camera_position": list(args.camera_position),
            "camera_target": list(args.camera_target),
            "camera_up": list(args.camera_up),
            "image_width": args.image_width,
            "image_height": args.image_height,
        },
        "videos": videos,
    }

    json_path = report_root / "phase1_continuous_video_summary.json"
    md_path = report_root / "phase1_continuous_video_report.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    make_report(md_path, summary)

    print("[Phase1.1] wrote {}".format(json_path))
    print("[Phase1.1] wrote {}".format(md_path))
    print("[Phase1.1] wrote videos to {}".format(outdir))

    written = [x for x in videos if x.get("status") == "written" and x.get("frames_written", 0) > 0]
    if not written:
        raise SystemExit("[Phase1.1][FAIL] no continuous videos written")


if __name__ == "__main__":
    main()
