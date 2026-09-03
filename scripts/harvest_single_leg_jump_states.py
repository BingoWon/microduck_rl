#!/usr/bin/env python3
"""Harvest reverse-curriculum states from successful one-shot jump rollouts."""

import argparse
import contextlib
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


TASK = "Mjlab-SingleLegJump-Flat-MicroDuck"
CATEGORIES = ("compressed", "airborne", "landing")


def _snapshot(raw) -> dict[str, torch.Tensor]:
    asset = raw.scene["robot"]
    servo_ids, _ = asset.find_joints(r"^(?!passive_).*")
    return {
        "root_pos": (
            asset.data.root_link_pos_w - raw.scene.terrain.env_origins
        ).clone(),
        "root_quat": asset.data.root_link_quat_w.clone(),
        "root_lin_vel": asset.data.root_link_lin_vel_w.clone(),
        "root_ang_vel": asset.data.root_link_ang_vel_w.clone(),
        "joint_pos": asset.data.joint_pos[:, servo_ids].clone(),
        "joint_vel": asset.data.joint_vel[:, servo_ids].clone(),
        "baseline_z": raw._slj_baseline_z.clone(),
        "peak_height_gain": raw._slj_peak_height_gain.clone(),
    }


def _store(
    records: dict[str, dict[str, torch.Tensor]],
    category: str,
    snapshot: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> None:
    pending = mask & ~records[category]["present"]
    if not bool(pending.any()):
        return
    records[category]["present"][pending] = True
    for name, values in snapshot.items():
        records[category][name][pending] = values[pending]


def _new_records(snapshot: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    n = snapshot["root_pos"].shape[0]
    return {
        category: {
            "present": torch.zeros(
                n, dtype=torch.bool, device=snapshot["root_pos"].device
            ),
            **{name: torch.zeros_like(values) for name, values in snapshot.items()},
        }
        for category in CATEGORIES
    }


def _serialize(
    records: dict[str, dict[str, torch.Tensor]],
    selected: torch.Tensor,
) -> dict[str, list[dict]]:
    output = {}
    for category in CATEGORIES:
        ids = selected[records[category]["present"][selected]]
        states = []
        for idx in ids.tolist():
            states.append(
                {
                    name: (
                        float(values[idx])
                        if values[idx].ndim == 0
                        else values[idx].cpu().tolist()
                    )
                    for name, values in records[category].items()
                    if name != "present"
                }
            )
        output[category] = states
    return output


def harvest_side(
    checkpoint: str,
    side: int,
    num_envs: int,
    max_states: int,
    device: str,
    seed: int,
) -> dict[str, list[dict]]:
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.commands["twist"].fixed_side = side
    env_cfg.commands["twist"].fixed_mode = 1
    env_cfg.commands["twist"].resampling_time_range = (6.0, 6.0)
    env_cfg.terminations.pop("jump_success", None)
    env_cfg.terminations.pop("jump_failure", None)
    agent_cfg = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=device),
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
    snapshot = _snapshot(raw)
    records = _new_records(snapshot)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)

    with torch.inference_mode():
        for _ in range(round(6.0 / raw.step_dt) - 1):
            before = _snapshot(raw)
            phase_before = obs["actor"][:, 50].clone()
            obs, _, dones, _ = env.step(policy(obs))
            after = _snapshot(raw)
            phase_after = obs["actor"][:, 50]

            compressed = (phase_before < 0.0) & (phase_after > 0.0) & alive
            _store(records, "compressed", after, compressed)
            _store(records, "airborne", after, raw._slj_takeoff_event & alive)
            _store(records, "landing", before, raw._slj_landing_event & alive)

            alive &= ~dones.bool()
    complete = alive & raw._slj_completed
    valid = complete.clone()
    for category in CATEGORIES:
        valid &= records[category]["present"]
    selected = torch.nonzero(valid, as_tuple=False).flatten()[:max_states]
    if len(selected) == 0:
        raise RuntimeError(
            f"No complete {'left' if side == -1 else 'right'} jump trajectories"
        )
    output = _serialize(records, selected)
    env.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-states", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    with contextlib.redirect_stdout(sys.stderr):
        states = {
            "left": harvest_side(
                str(args.checkpoint),
                -1,
                args.num_envs,
                args.max_states,
                args.device,
                args.seed,
            ),
            "right": harvest_side(
                str(args.checkpoint),
                1,
                args.num_envs,
                args.max_states,
                args.device,
                args.seed,
            ),
        }
    payload = {
        "version": 1,
        "task": TASK,
        "source_checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "states": states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "counts": {
                    side: {
                        category: len(values)
                        for category, values in categories.items()
                    }
                    for side, categories in states.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
