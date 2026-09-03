#!/usr/bin/env python3
"""Evaluate synchronized crouch checkpoints without interrupting training."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

SERIES = ("p0", "p10", "p25", "p50", "p75", "p90", "p100", "mean")


def publish(result_path: Path, tensorboard_dir: Path, step: int) -> None:
    results = json.loads(result_path.read_text(encoding="utf-8"))
    writer = SummaryWriter(tensorboard_dir)
    for result in results:
        side = result["side"]
        for metric in (
            "command_count",
            "successes",
            "success_rate",
            "nonfoot_contact_rate",
            "fell_over_rate",
        ):
            writer.add_scalar(
                f"Evaluation/{side}/{metric}",
                result[metric],
                step,
            )
        for metric in (
            "max_depth_m",
            "max_rise_m",
            "final_height_error_m",
            "episode_length_s",
        ):
            for series in SERIES:
                writer.add_scalar(
                    f"Evaluation/{side}/{metric}/{series}",
                    result[metric][series],
                    step,
                )
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("tensorboard_dir", type=Path)
    parser.add_argument("--steps", default="25,50,75,99")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    steps = tuple(int(value) for value in args.steps.split(","))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name("eval.py")

    while True:
        for step in steps:
            checkpoint = args.checkpoint_dir / f"model_{step}.pt"
            result = args.result_dir / f"model_{step}.json"
            complete = args.result_dir / f"model_{step}.complete"
            if not checkpoint.is_file() or complete.exists():
                continue
            temporary = result.with_suffix(".json.part")
            with temporary.open("w", encoding="utf-8") as stdout, (
                args.result_dir / f"model_{step}.stderr"
            ).open("a", encoding="utf-8") as stderr:
                evaluated = subprocess.run(
                    [
                        "/usr/bin/nice",
                        "-n",
                        "10",
                        sys.executable,
                        str(evaluator),
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
                    stdout=stdout,
                    stderr=stderr,
                    timeout=1800,
                    check=False,
                )
            if evaluated.returncode != 0:
                temporary.unlink(missing_ok=True)
                continue
            temporary.replace(result)
            publish(result, args.tensorboard_dir, step)
            complete.write_text("ok\n", encoding="utf-8")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
