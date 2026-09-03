#!/usr/bin/env python3
"""Print one compact, machine-readable status snapshot for a remote jump run."""

import argparse
import json
import re
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="vast-microduck")
    parser.add_argument("--pid", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    command = f"""
alive=0
kill -0 {int(args.pid)} 2>/dev/null && alive=1
printf 'ALIVE=%s\\n' "$alive"
grep -a 'Learning iteration' {args.log} | tail -1 || true
grep -a 'Steps per second' {args.log} | tail -1 || true
grep -a 'Episode_Termination/nan_state' {args.log} | tail -1 || true
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits
"""
    result = subprocess.run(
        ["ssh", args.host, command],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or f"ssh exited {result.returncode}")
    text = result.stdout
    iteration = re.search(r"Learning iteration\s+(\d+)/(\d+)", text)
    steps = re.search(r"Steps per second:\s+(\d+)", text)
    nan = re.search(r"nan_state:\s+([0-9.]+)", text)
    gpu = re.search(
        r"^([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)$",
        text,
        re.MULTILINE,
    )
    print(
        json.dumps(
            {
                "alive": "ALIVE=1" in text,
                "iteration": int(iteration.group(1)) if iteration else None,
                "target": int(iteration.group(2)) if iteration else None,
                "steps_per_second": int(steps.group(1)) if steps else None,
                "nan_state": float(nan.group(1)) if nan else None,
                "gpu_util_percent": float(gpu.group(1)) if gpu else None,
                "gpu_memory_used_mib": float(gpu.group(2)) if gpu else None,
                "gpu_memory_total_mib": float(gpu.group(3)) if gpu else None,
                "gpu_power_w": float(gpu.group(4)) if gpu else None,
                "phase": "training" if iteration else "initializing",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
