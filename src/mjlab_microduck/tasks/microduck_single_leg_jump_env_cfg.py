"""One-shot commanded left/right single-leg jump."""

import os
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_single_leg_stand_env_cfg import (
    COMMAND_NAME,
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
    return cfg


MicroduckSingleLegJumpRlCfg = deepcopy(MicroduckSingleLegStandRlCfg)
MicroduckSingleLegJumpRlCfg.experiment_name = "single_leg_jump"
MicroduckSingleLegJumpRlCfg.run_name = "single_leg_jump"
MicroduckSingleLegJumpRlCfg.max_iterations = 6_000
