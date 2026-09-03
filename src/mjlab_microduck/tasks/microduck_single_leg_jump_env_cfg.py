"""One-shot commanded left/right single-leg jump."""

import os
from copy import deepcopy
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_single_leg_stand_env_cfg import (
    COMMAND_NAME,
    FEET_CFG,
    FEET_SENSOR,
    NONFOOT_SENSOR,
    MicroduckSingleLegStandRlCfg,
    make_microduck_single_leg_stand_strict_env_cfg,
)


EPISODE_LENGTH_S = 6.0
TRANSITION_EPISODE_LENGTH_S = 10.0
RESET_STATE_BANK = (
    Path(__file__).resolve().parent / "data" / "single_leg_jump_reset_states.json"
)


def _fixed_play_mode() -> int:
    value = os.environ.get("SINGLE_LEG_JUMP_PLAY_MODE", "jump").lower()
    if value in ("random", "-1"):
        return -1
    if value in ("stand", "0"):
        return 0
    if value in ("jump", "1"):
        return 1
    raise ValueError("SINGLE_LEG_JUMP_PLAY_MODE must be stand, jump, or random")


def make_microduck_single_leg_jump_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_single_leg_stand_strict_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    stand_command = cfg.commands[COMMAND_NAME]
    command_kwargs = vars(stand_command).copy()
    command_kwargs["resampling_time_range"] = (
        EPISODE_LENGTH_S,
        EPISODE_LENGTH_S,
    )
    cfg.commands[COMMAND_NAME] = microduck_mdp.SingleLegJumpCommandCfg(
        **command_kwargs,
        jump_prob=0.75,
        fixed_mode=_fixed_play_mode() if play else -1,
        prepare_s=1.5,
        crouch_s=0.22,
        extend_s=0.12,
    )

    cfg.rewards["strict_single_leg_hold"].params["required_mode"] = "stand"
    cfg.rewards["strict_single_leg_hold"].weight = 2.0
    for metric_name in (
        "single_leg_success",
        "single_leg_success_left",
        "single_leg_success_right",
    ):
        cfg.metrics[metric_name].params["required_mode"] = "stand"
    cfg.rewards.pop("failed_episode", None)
    jump_params = {
        "command_name": COMMAND_NAME,
        "sensor_name": FEET_SENSOR,
        "nonfoot_sensor_name": NONFOOT_SENSOR,
        "asset_cfg": FEET_CFG,
        "min_takeoff_velocity": 0.02,
        "min_height_gain": 0.003,
        "recovery_s": 0.5,
    }
    cfg.rewards["jump_completion"] = RewardTermCfg(
        func=microduck_mdp.single_leg_jump_completion_reward,
        weight=20.0,
        params=jump_params,
    )
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=microduck_mdp.single_leg_jump_banked_height_reward,
        weight=1.0,
        params={**jump_params, "target_height_gain": 0.01},
    )
    cfg.rewards["jump_compression_progress"] = RewardTermCfg(
        func=microduck_mdp.single_leg_jump_compression_progress,
        weight=0.5,
        params={**jump_params, "target_compression": 0.01},
    )
    cfg.rewards["jump_upward_progress"] = RewardTermCfg(
        func=microduck_mdp.single_leg_jump_upward_progress,
        weight=0.5,
        params={**jump_params, "target_velocity": 0.15},
    )
    cfg.rewards["jump_takeoff"] = RewardTermCfg(
        func=microduck_mdp.single_leg_jump_takeoff_event,
        weight=1.0,
        params=jump_params,
    )
    cfg.rewards["jump_landing"] = RewardTermCfg(
        func=microduck_mdp.single_leg_jump_landing_event,
        weight=1.0,
        params=jump_params,
    )
    cfg.rewards["jump_recovery_progress"] = RewardTermCfg(
        func=microduck_mdp.single_leg_jump_recovery_progress,
        weight=5.0,
        params=jump_params,
    )
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["joint_torque_rate_l2"].weight = -5e-4

    cfg.events["reset_single_leg_jump"] = EventTermCfg(
        func=microduck_mdp.reset_single_leg_jump_state,
        mode="reset",
        params={
            "state_bank_path": str(RESET_STATE_BANK),
            "standing_prob": 0.35 if not play else 1.0,
            "compressed_prob": 0.15 if not play else 0.0,
            "airborne_prob": 0.50 if not play else 0.0,
        },
    )
    cfg.terminations["jump_success"] = TerminationTermCfg(
        func=microduck_mdp.single_leg_jump_success,
        time_out=False,
        params=jump_params,
    )
    cfg.terminations["jump_failure"] = TerminationTermCfg(
        func=microduck_mdp.single_leg_jump_failure,
        time_out=False,
        params=jump_params,
    )
    for side_name, support_side in (("left", -1), ("right", 1)):
        for metric in (
            "command_count",
            "takeoff_rate",
            "landing_rate",
            "completion_rate",
            "failure_rate",
            "peak_height_gain",
        ):
            cfg.metrics[f"jump_{metric}_{side_name}"] = MetricsTermCfg(
                func=microduck_mdp.single_leg_jump_metric,
                reduce="last",
                params={
                    **jump_params,
                    "metric": metric,
                    "support_side": support_side,
                },
            )
    if not play:
        cfg.curriculum["jump_reset_mix"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={
                "event_name": "reset_single_leg_jump",
                "param_stages": [
                    {
                        "step": 0,
                        "params": {
                            "standing_prob": 0.35,
                            "compressed_prob": 0.15,
                            "airborne_prob": 0.50,
                        },
                    },
                    {
                        "step": 1000 * 24,
                        "params": {
                            "standing_prob": 0.60,
                            "compressed_prob": 0.15,
                            "airborne_prob": 0.25,
                        },
                    },
                    {
                        "step": 2000 * 24,
                        "params": {
                            "standing_prob": 0.90,
                            "compressed_prob": 0.06,
                            "airborne_prob": 0.04,
                        },
                    },
                    {
                        "step": 3000 * 24,
                        "params": {
                            "standing_prob": 1.0,
                            "compressed_prob": 0.0,
                            "airborne_prob": 0.0,
                        },
                    },
                ],
            },
        )
        for reward_name, initial_weight in (
            ("jump_compression_progress", 0.5),
            ("jump_upward_progress", 0.5),
            ("jump_takeoff", 1.0),
            ("jump_landing", 1.0),
            ("jump_recovery_progress", 5.0),
        ):
            cfg.curriculum[f"{reward_name}_weight"] = CurriculumTermCfg(
                func=microduck_mdp.reward_weight,
                params={
                    "reward_name": reward_name,
                    "weight_stages": [
                        {"step": 0, "weight": initial_weight},
                        {"step": 1000 * 24, "weight": initial_weight * 0.5},
                        {"step": 2000 * 24, "weight": initial_weight * 0.25},
                        {"step": 3000 * 24, "weight": 0.0},
                    ],
                },
            )
        cfg.curriculum["jump_action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 2000 * 24, "weight": -0.03},
                    {"step": 3000 * 24, "weight": -0.05},
                ],
            },
        )
    return cfg


def make_microduck_single_leg_jump_transition_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Continuation task for stand/jump transactions with a return side."""
    cfg = make_microduck_single_leg_jump_env_cfg(play=play)
    cfg.episode_length_s = TRANSITION_EPISODE_LENGTH_S
    command_kwargs = vars(cfg.commands[COMMAND_NAME]).copy()
    command_kwargs["resampling_time_range"] = (
        TRANSITION_EPISODE_LENGTH_S,
        TRANSITION_EPISODE_LENGTH_S,
    )
    cfg.commands[COMMAND_NAME] = (
        microduck_mdp.SingleLegJumpTransitionCommandCfg(
            **command_kwargs,
            transaction_jump_prob=0.75,
            cross_side_prob=0.5,
            initial_hold_s=1.5,
            recovery_s=1.5,
        )
    )
    cfg.events["reset_single_leg_jump"].params.update(
        standing_prob=1.0,
        compressed_prob=0.0,
        airborne_prob=0.0,
        fixed_side=0,
    )
    cfg.curriculum.pop("jump_reset_mix", None)
    cfg.terminations.pop("jump_success", None)
    return_params = {
        "command_name": COMMAND_NAME,
        "sensor_name": FEET_SENSOR,
        "nonfoot_sensor_name": NONFOOT_SENSOR,
        "asset_cfg": FEET_CFG,
        "min_takeoff_velocity": 0.02,
        "min_height_gain": 0.003,
        "recovery_s": 0.5,
        "return_hold_s": 1.0,
    }
    cfg.rewards["return_completion"] = RewardTermCfg(
        func=microduck_mdp.single_leg_jump_return_reward,
        weight=5.0,
        params=return_params,
    )
    cfg.metrics["transaction_success"] = MetricsTermCfg(
        func=microduck_mdp.single_leg_jump_return_success,
        reduce="last",
        params=return_params,
    )
    cfg.terminations["transaction_success"] = TerminationTermCfg(
        func=microduck_mdp.single_leg_jump_return_success,
        time_out=False,
        params=return_params,
    )
    return cfg


MicroduckSingleLegJumpRlCfg = deepcopy(MicroduckSingleLegStandRlCfg)
MicroduckSingleLegJumpRlCfg.experiment_name = "single_leg_jump"
MicroduckSingleLegJumpRlCfg.run_name = "single_leg_jump"
MicroduckSingleLegJumpRlCfg.max_iterations = 6_000
MicroduckSingleLegJumpRlCfg.actor.distribution_cfg["init_std"] = 0.005
MicroduckSingleLegJumpRlCfg.algorithm.learning_rate = 5e-5
MicroduckSingleLegJumpRlCfg.algorithm.entropy_coef = 0.0
MicroduckSingleLegJumpRlCfg.algorithm.schedule = "fixed"

MicroduckSingleLegJumpTransitionRlCfg = deepcopy(MicroduckSingleLegJumpRlCfg)
MicroduckSingleLegJumpTransitionRlCfg.experiment_name = (
    "single_leg_jump_transitions"
)
MicroduckSingleLegJumpTransitionRlCfg.run_name = "single_leg_jump_transitions"
MicroduckSingleLegJumpTransitionRlCfg.max_iterations = 3_000
