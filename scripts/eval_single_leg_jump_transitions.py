#!/usr/bin/env python3
"""Evaluate all stand/jump transactions and the return to the resume side."""

import argparse
import contextlib
import json
import sys
from dataclasses import asdict

TASK = "Mjlab-SingleLegJumpTransitions-Flat-MicroDuck"
TRANSACTIONS = (
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 1),
    (-1, 1, 1),
    (1, -1, 1),
    (1, 1, 1),
)


def _name(resume_side: int, target_side: int, is_jump: int) -> str:
    side = {-1: "left", 1: "right"}
    mode = "jump" if is_jump else "stand"
    return (
        f"{side[resume_side]}_stand->{side[target_side]}_{mode}"
        f"->{side[resume_side]}_stand"
    )


def evaluate(
    checkpoint: str,
    resume_side: int,
    target_side: int,
    is_jump: int,
    num_envs: int,
    episodes: int,
    device: str,
    seed: int,
) -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner

    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = load_env_cfg(TASK, play=True)
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.scene.num_envs = num_envs
    command_cfg = cfg.commands["twist"]
    command_cfg.fixed_resume_side = resume_side
    command_cfg.fixed_target_side = target_side
    command_cfg.fixed_transaction_mode = is_jump
    cfg.events["reset_single_leg_jump"].params["fixed_side"] = resume_side
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
    feet_cfg = SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
    )
    target_hold = torch.zeros(num_envs, device=device)
    return_hold = torch.zeros(num_envs, device=device)
    max_target_hold = torch.zeros(num_envs, device=device)
    max_return_hold = torch.zeros(num_envs, device=device)
    took_off = torch.zeros(num_envs, dtype=torch.bool, device=device)
    landed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    completed_jump = torch.zeros(
        num_envs, dtype=torch.bool, device=device
    )
    completed = target_successes = return_successes = failures = 0
    takeoffs = landings = jump_completions = 0
    termination_counts = {
        name: 0 for name in raw.termination_manager.active_terms
    }

    with torch.inference_mode():
        while completed < episodes:
            obs, _, dones, _ = env.step(policy(obs))
            command = raw.command_manager.get_term("twist")
            valid = microduck_mdp.single_leg_success_state(
                raw,
                command_name="twist",
                sensor_name="feet_ground_contact",
                asset_cfg=feet_cfg,
                nonfoot_sensor_name="nonfoot_ground_contact",
                min_clearance=0.003,
                max_tilt_deg=35.0,
                max_lin_vel=0.08,
                max_ang_vel=0.8,
                require_com_inside=False,
            ).bool()
            target_active = (
                (command._elapsed >= command.cfg.initial_hold_s)
                & ~command.returning
            )
            target_valid = valid & target_active & ~command.transaction_is_jump
            return_valid = valid & command.returning
            target_hold = torch.where(
                target_valid,
                target_hold + raw.step_dt,
                torch.zeros_like(target_hold),
            )
            return_hold = torch.where(
                return_valid,
                return_hold + raw.step_dt,
                torch.zeros_like(return_hold),
            )
            max_target_hold = torch.maximum(max_target_hold, target_hold)
            max_return_hold = torch.maximum(max_return_hold, return_hold)
            took_off |= raw._slj_took_off
            landed |= raw._slj_landed
            completed_jump |= raw._slj_completed

            done_ids = torch.nonzero(dones, as_tuple=False).flatten()
            for idx in done_ids.tolist():
                if completed >= episodes:
                    break
                completed += 1
                target_ok = (
                    bool(completed_jump[idx])
                    if is_jump
                    else bool(max_target_hold[idx] >= 1.0)
                )
                return_ok = bool(max_return_hold[idx] >= 1.0)
                target_successes += int(target_ok)
                return_successes += int(return_ok)
                failures += int(
                    raw.termination_manager.get_term("jump_failure")[idx]
                )
                takeoffs += int(took_off[idx])
                landings += int(landed[idx])
                jump_completions += int(completed_jump[idx])
                for name in termination_counts:
                    termination_counts[name] += int(
                        raw.termination_manager.get_term(name)[idx]
                    )

            if len(done_ids) == 0:
                continue
            reset_obs, _ = raw.reset(env_ids=done_ids)
            for group in obs.keys():  # noqa: SIM118 - TensorDict iterates values.
                obs[group][done_ids] = reset_obs[group][done_ids]
            for tensor in (
                target_hold,
                return_hold,
                max_target_hold,
                max_return_hold,
            ):
                tensor[done_ids] = 0.0
            for tensor in (took_off, landed, completed_jump):
                tensor[done_ids] = False

    env.close()
    return {
        "transaction": _name(resume_side, target_side, is_jump),
        "episodes": completed,
        "target_success_rate": target_successes / completed,
        "return_success_rate": return_successes / completed,
        "true_takeoff_rate": takeoffs / completed,
        "same_foot_landing_rate": landings / completed,
        "jump_completion_rate": jump_completions / completed,
        "failure_rate": failures / completed,
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
                resume_side,
                target_side,
                is_jump,
                args.num_envs,
                args.episodes,
                args.device,
                args.seed,
            )
            for resume_side, target_side, is_jump in TRANSACTIONS
        ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
