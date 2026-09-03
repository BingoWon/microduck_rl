#!/usr/bin/env python3
"""View one single-leg crouch checkpoint with the real episode cadence."""

import argparse
from dataclasses import asdict

import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViserPlayViewer
from rsl_rl.runners import OnPolicyRunner

TASK = "Mjlab-SingleLegCrouch-Flat-MicroDuck"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    args = parser.parse_args()
    side = -1 if args.side == "left" else 1

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["twist"].fixed_side = side
    env_cfg.events["reset_single_leg_crouch"].params["fixed_side"] = side
    agent_cfg = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=args.device),
        clip_actions=agent_cfg.clip_actions,
    )
    runner_cls = load_runner_cls(TASK) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        args.checkpoint,
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    server = viser.ViserServer(
        port=args.port,
        label="Microduck single-leg crouch",
    )
    ViserPlayViewer(env, policy, viser_server=server).run()
    env.close()


if __name__ == "__main__":
    main()
