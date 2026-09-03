#!/usr/bin/env python3
"""Record one fixed-side single-leg jump rollout."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder


TASK = "Mjlab-SingleLegJump-Flat-MicroDuck"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.seconds <= 0.0:
        parser.error("--seconds must be positive")

    side = -1 if args.side == "left" else 1
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    cfg.commands["twist"].fixed_side = side
    cfg.commands["twist"].fixed_mode = 1
    cfg.viewer.width = args.width
    cfg.viewer.height = args.height
    agent_cfg = load_rl_cfg(TASK)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
    steps = max(1, round(args.seconds / raw.step_dt))
    recorded = VideoRecorder(
        raw,
        video_folder=args.output_dir,
        step_trigger=lambda step: step == 0,
        video_length=steps,
        name_prefix=f"single-leg-jump-{args.side}",
    )
    env = RslRlVecEnvWrapper(recorded, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(args.checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, _, _ = env.step(policy(obs))
    env.close()
    print(
        (
            args.output_dir
            / f"single-leg-jump-{args.side}-step-0.mp4"
        ).resolve()
    )


if __name__ == "__main__":
    main()
