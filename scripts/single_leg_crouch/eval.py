#!/usr/bin/env python3
"""Deterministic fixed-side evaluation for single-leg crouch and return."""

import argparse
import contextlib
import json
import sys
from dataclasses import asdict

TASK = "Mjlab-SingleLegCrouch-Flat-MicroDuck"
PERCENTILES = (0, 10, 25, 50, 75, 90, 100)


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    result = {"mean": sum(ordered) / len(ordered)}
    for percentile in PERCENTILES:
        position = (len(ordered) - 1) * percentile / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        result[f"p{percentile}"] = (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )
    return result


def evaluate(
    checkpoint: str,
    side: int,
    num_envs: int,
    episodes: int,
    device: str,
    seed: int,
) -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner

    cfg = load_env_cfg(TASK, play=True)
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.scene.num_envs = num_envs
    cfg.commands["twist"].fixed_side = side
    cfg.events["reset_single_leg_crouch"].params["fixed_side"] = side
    agent_cfg = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device=device),
        clip_actions=agent_cfg.clip_actions,
    )
    runner_cls = load_runner_cls(TASK) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        checkpoint,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()
    raw = env.unwrapped
    dt = float(raw.step_dt)

    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    completed = successes = nonfoot_failures = fell_over_failures = 0
    depths: list[float] = []
    rises: list[float] = []
    final_errors: list[float] = []
    episode_lengths: list[float] = []
    termination_counts = {
        name: 0 for name in raw.termination_manager.active_terms
    }

    with torch.inference_mode():
        while completed < episodes:
            obs, _, dones, _ = env.step(policy(obs))
            episode_steps += 1
            done_ids = torch.nonzero(dones, as_tuple=False).flatten()
            for idx in done_ids.tolist():
                if completed >= episodes:
                    break
                completed += 1
                success = bool(
                    raw.termination_manager.get_term("crouch_success")[idx]
                )
                successes += int(success)
                nonfoot_failures += int(
                    raw.termination_manager.get_term("nonfoot_contact")[idx]
                )
                fell_over_failures += int(
                    raw.termination_manager.get_term("fell_over")[idx]
                )
                depths.append(float(raw._slc_best_depth[idx]))
                rises.append(float(raw._slc_best_rise[idx]))
                final_errors.append(float(raw._slc_final_height_error[idx]))
                episode_lengths.append(float(episode_steps[idx]) * dt)
                for name in termination_counts:
                    termination_counts[name] += int(
                        raw.termination_manager.get_term(name)[idx]
                    )

            if len(done_ids) == 0:
                continue
            reset_obs, _ = raw.reset(env_ids=done_ids)
            for group in obs:
                obs[group][done_ids] = reset_obs[group][done_ids]
            episode_steps[done_ids] = 0

    env.close()
    return {
        "side": "left" if side == -1 else "right",
        "command_count": completed,
        "successes": successes,
        "success_rate": successes / completed,
        "nonfoot_contact_rate": nonfoot_failures / completed,
        "fell_over_rate": fell_over_failures / completed,
        "max_depth_m": summarize(depths),
        "max_rise_m": summarize(rises),
        "final_height_error_m": summarize(final_errors),
        "episode_length_s": summarize(episode_lengths),
        "termination_counts": termination_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.episodes <= 0:
        parser.error("--num-envs and --episodes must be positive")

    with contextlib.redirect_stdout(sys.stderr):
        results = [
            evaluate(
                args.checkpoint,
                side,
                args.num_envs,
                args.episodes,
                args.device,
                args.seed,
            )
            for side in (-1, 1)
        ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
