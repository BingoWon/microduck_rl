#!/usr/bin/env python3
"""Open a four-command single-leg jump checkpoint in the Viser viewer."""

import argparse
from dataclasses import asdict

import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViserPlayViewer
from rsl_rl.runners import OnPolicyRunner

TASK = "Mjlab-SingleLegJump-Flat-MicroDuck"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 3600.0
    command_cfg = env_cfg.commands["twist"]
    command_cfg.fixed_side = -1
    command_cfg.fixed_mode = 1
    command_cfg.resampling_time_range = (3600.0, 3600.0)
    env_cfg.events["reset_single_leg_jump"].params["fixed_side"] = -1
    env_cfg.terminations.pop("jump_success", None)

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
    server = viser.ViserServer(port=args.port, label="Microduck single-leg jump")
    ViserPlayViewer(env, policy, viser_server=server).run()
    env.close()


if __name__ == "__main__":
    main()
