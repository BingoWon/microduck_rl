"""One-shot commanded left/right single-leg jump."""

import os
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
        weight=10.0,
        params=jump_params,
    )
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=microduck_mdp.single_leg_jump_banked_height_reward,
        weight=1.0,
        params={**jump_params, "target_height_gain": 0.01},
    )

    cfg.events["reset_single_leg_jump"] = EventTermCfg(
        func=microduck_mdp.reset_single_leg_jump_state,
        mode="reset",
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
    return cfg


MicroduckSingleLegJumpRlCfg = deepcopy(MicroduckSingleLegStandRlCfg)
MicroduckSingleLegJumpRlCfg.experiment_name = "single_leg_jump"
MicroduckSingleLegJumpRlCfg.run_name = "single_leg_jump"
MicroduckSingleLegJumpRlCfg.max_iterations = 6_000
