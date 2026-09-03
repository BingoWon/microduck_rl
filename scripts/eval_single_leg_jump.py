#!/usr/bin/env python3
"""Headless ground-truth evaluation for one-shot left/right jumps."""

import argparse
import contextlib
import json
import sys
from dataclasses import asdict


TASK = "Mjlab-SingleLegJump-Flat-MicroDuck"
PERCENTILES = (0, 10, 25, 50, 75, 90, 100)


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    result = {"mean": sum(ordered) / len(ordered)}
    for percentile in PERCENTILES:
        position = (len(ordered) - 1) * percentile / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        result[f"p{percentile}"] = (
            ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
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
    from rsl_rl.runners import OnPolicyRunner

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    cfg = load_env_cfg(TASK, play=True)
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.scene.num_envs = num_envs
    cfg.commands["twist"].fixed_side = side
    cfg.commands["twist"].fixed_mode = 1
    cfg.commands["twist"].resampling_time_range = (6.0, 6.0)
    cfg.events["reset_single_leg_jump"].params["fixed_side"] = side
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
    support = 0 if side == -1 else 1
    swing = 1 - support

    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    took_off = torch.zeros(num_envs, dtype=torch.bool, device=device)
    landed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    swing_contacted = torch.zeros(num_envs, dtype=torch.bool, device=device)
    nonfoot_contacted = torch.zeros(num_envs, dtype=torch.bool, device=device)
    landing_step = torch.full(
        (num_envs,), -1, dtype=torch.long, device=device
    )
    completed = 0
    successes = takeoffs = landings = 0
    swing_contact_failures = nonfoot_contact_failures = 0
    peak_height_gains: list[float] = []
    recovery_times: list[float] = []
    episode_lengths: list[float] = []
    termination_counts = {
        name: 0 for name in raw.termination_manager.active_terms
    }

    with torch.inference_mode():
        while completed < episodes:
            obs, _, dones, _ = env.step(policy(obs))
            episode_steps += 1
            command = raw.command_manager.get_term("twist")
            active = command.is_jump & (command.alpha >= 0.99)
            contacts = (
                raw.scene.sensors["feet_ground_contact"]
                .data.found.reshape(num_envs, -1)[:, :2]
                .bool()
            )
            swing_contacted |= active & contacts[:, swing]
            nonfoot_contacted |= active & (
                raw.scene.sensors["nonfoot_ground_contact"]
                .data.found.reshape(num_envs, -1)
                .sum(dim=1)
                > 0
            )
            took_off |= raw._slj_took_off
            first_landing = raw._slj_landed & ~landed
            landing_step[first_landing] = episode_steps[first_landing]
            landed |= raw._slj_landed

            done_ids = torch.nonzero(dones, as_tuple=False).flatten()
            for idx in done_ids.tolist():
                if completed >= episodes:
                    break
                completed += 1
                success = bool(
                    raw.termination_manager.get_term("jump_success")[idx]
                )
                successes += int(success)
                takeoffs += int(took_off[idx])
                landings += int(landed[idx])
                swing_contact_failures += int(swing_contacted[idx])
                nonfoot_contact_failures += int(nonfoot_contacted[idx])
                peak_height_gains.append(float(raw._slj_peak_height_gain[idx]))
                episode_lengths.append(float(episode_steps[idx]) * dt)
                if success and landing_step[idx] >= 0:
                    recovery_times.append(
                        float(episode_steps[idx] - landing_step[idx]) * dt
                    )
                for name in termination_counts:
                    termination_counts[name] += int(
                        raw.termination_manager.get_term(name)[idx]
                    )

            if len(done_ids) == 0:
                continue
            reset_obs, _ = raw.reset(env_ids=done_ids)
            for group in obs.keys():
                obs[group][done_ids] = reset_obs[group][done_ids]
            episode_steps[done_ids] = 0
            took_off[done_ids] = False
            landed[done_ids] = False
            swing_contacted[done_ids] = False
            nonfoot_contacted[done_ids] = False
            landing_step[done_ids] = -1

    env.close()
    return {
        "side": "left" if side == -1 else "right",
        "command_count": completed,
        "successes": successes,
        "success_rate": successes / completed,
        "true_takeoffs": takeoffs,
        "true_takeoff_rate": takeoffs / completed,
        "same_foot_landings": landings,
        "same_foot_landing_rate": landings / completed,
        "landing_per_takeoff": landings / max(takeoffs, 1),
        "recovery_completions": successes,
        "recovery_completion_rate": successes / max(landings, 1),
        "completion_per_takeoff": successes / max(takeoffs, 1),
        "swing_foot_contact_rate": swing_contact_failures / completed,
        "nonfoot_contact_rate": nonfoot_contact_failures / completed,
        "peak_root_height_gain_m": summarize(peak_height_gains),
        "recovery_time_s": summarize(recovery_times),
        "episode_length_s": summarize(episode_lengths),
        "termination_counts": termination_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
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
