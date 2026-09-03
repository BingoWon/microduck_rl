#!/usr/bin/env python3
"""Continuously mirror one remote TensorBoard run without copying checkpoints."""

import argparse
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("remote")
    parser.add_argument("local", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    args.local.mkdir(parents=True, exist_ok=True)

    while True:
        result = subprocess.run(
            [
                "rsync",
                "-az",
                "--include=events.out.tfevents*",
                "--exclude=*",
                args.remote.rstrip("/") + "/",
                str(args.local) + "/",
            ],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            print(result.stderr.strip(), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
