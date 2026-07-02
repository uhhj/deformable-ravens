#!/usr/bin/env python
"""Generate small paired datasets for the Phase1 CCDA hidden-contact cable task."""

import argparse
import os
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np

from ravens import Dataset, Environment, tasks


def parse_conditions(raw: List[str]) -> List[str]:
    out = []
    for x in raw:
        for y in x.split(","):
            y = y.strip()
            if y:
                out.append(y)
    return out


def has_rgbd(obs):
    return isinstance(obs, dict) and "color" in obs and "depth" in obs


def rollout(agent, env, task):
    episode = []
    total_reward = 0
    obs = env.reset(task)
    info = env.info
    last_valid_obs = obs if has_rgbd(obs) else None
    last_stuff = (obs, info)

    for t in range(task.max_steps):
        act = agent.act(obs, info)

        if len(obs) > 0 and act.get("primitive"):
            episode.append((obs, act, info))
            if has_rgbd(obs):
                last_valid_obs = obs

        obs, reward, done, info = env.step(act)
        total_reward += reward
        if has_rgbd(obs):
            last_valid_obs = obs
            last_stuff = (obs, info)
        elif last_valid_obs is not None:
            last_stuff = (last_valid_obs, info)
        else:
            last_stuff = (obs, info)

        if done:
            break

    return total_reward, t, episode, last_stuff


def condition_dataset_path(root: str, task_name: str, condition: str) -> str:
    return os.path.join(root, task_name, condition)


def generate_for_condition(args, condition: str) -> Dict:
    path = condition_dataset_path(args.output_root, args.task, condition)

    if args.fresh and os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)
    dataset = Dataset(path)

    env = Environment(disp=args.disp, hz=args.hz)
    added = 0
    skipped = 0

    for local_i in range(args.num_demos):
        seed = args.seed_start + local_i

        random.seed(seed)
        np.random.seed(seed)

        os.environ["CCDA_HIDDEN_CONDITION"] = condition
        os.environ["CCDA_VISIBLE_SEED"] = str(seed)
        os.environ["CCDA_PAIR_GROUP"] = "visible_seed_{:06d}".format(seed)

        task = tasks.names[args.task]()
        task.mode = "train"
        task.hidden_condition = condition
        task.ccda_visible_seed = seed
        task.ccda_pair_group = "visible_seed_{:06d}".format(seed)

        agent = task.oracle(env)
        total_reward, t, episode, last_stuff = rollout(agent, env, task)

        if len(episode) == 0:
            skipped += 1
            print(
                "[Phase1] skip empty episode condition={} seed={} reward={}".format(
                    condition, seed, total_reward
                )
            )
            continue

        dataset.add(episode, last_stuff=last_stuff)
        added += 1
        print(
            "[Phase1] condition={} seed={} len={} reward={:.4f}".format(
                condition, seed, len(episode), float(total_reward)
            )
        )

    env.stop()

    return {
        "condition": condition,
        "path": path,
        "added": added,
        "skipped": skipped,
        "seed_start": args.seed_start,
        "seed_end_exclusive": args.seed_start + args.num_demos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hidden-contact-cable-line")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["free", "hidden_pin", "hidden_high_friction"],
    )
    parser.add_argument("--num_demos", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--hz", type=float, default=240.0)
    parser.add_argument("--disp", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--output_root", default="data")
    args = parser.parse_args()

    conditions = parse_conditions(args.conditions)

    if args.task not in tasks.names:
        raise KeyError("Task not registered: {}".format(args.task))

    summaries = []
    for condition in conditions:
        summaries.append(generate_for_condition(args, condition))

    print("[Phase1] generation summary:")
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
