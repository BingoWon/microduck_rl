#!/usr/bin/env python3
"""Evaluate selected checkpoints locally and publish results to TensorBoard."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("tensorboard_dir", type=Path)
    parser.add_argument("--steps", default="250,500,750,1000")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    steps = tuple(int(value) for value in args.steps.split(","))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]

    while True:
        for step in steps:
            checkpoint = args.checkpoint_dir / f"model_{step}.pt"
            result = args.result_dir / f"model_{step}.json"
            complete = args.result_dir / f"model_{step}.complete"
            if not checkpoint.is_file() or complete.exists():
                continue
            temporary = result.with_suffix(".json.part")
            evaluated = subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts" / "eval_single_leg_jump.py"),
                    str(checkpoint),
                    "--num-envs",
                    str(args.num_envs),
                    "--episodes",
                    str(args.episodes),
                    "--device",
                    "cpu",
                    "--seed",
                    "42",
                ],
                stdout=temporary.open("w", encoding="utf-8"),
                stderr=(args.result_dir / f"model_{step}.stderr").open(
                    "a", encoding="utf-8"
                ),
                cwd=repo,
                timeout=1800,
                check=False,
            )
            if evaluated.returncode != 0:
                temporary.unlink(missing_ok=True)
                continue
            temporary.replace(result)
            published = subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts" / "log_single_leg_eval_tensorboard.py"),
                    str(args.tensorboard_dir),
                    f"{step}={result}",
                ],
                cwd=repo,
                timeout=60,
                check=False,
            )
            if published.returncode == 0:
                complete.write_text("ok\n", encoding="utf-8")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
