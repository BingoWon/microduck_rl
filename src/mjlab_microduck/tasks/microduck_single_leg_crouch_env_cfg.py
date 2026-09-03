"""Commanded single-leg crouch and return from a verified standing state."""

from copy import deepcopy
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
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
    make_microduck_single_leg_stand_env_cfg,
)

EPISODE_LENGTH_S = 5.5
INITIAL_HOLD_S = 0.5
CROUCH_S = 2.0
RETURN_S = 2.0
FINAL_HOLD_S = 1.0
HEIGHT_UNIT = 0.01
RETURN_TOLERANCE = 0.005
RETURN_HOLD_S = 0.5
RESET_STATE_BANK = (
    Path(__file__).resolve().parent / "data" / "single_leg_jump_reset_states.json"
)


def make_microduck_single_leg_crouch_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_single_leg_stand_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    command_kwargs = vars(cfg.commands[COMMAND_NAME]).copy()
    command_kwargs["resampling_time_range"] = (
        EPISODE_LENGTH_S,
        EPISODE_LENGTH_S,
    )
    cfg.commands[COMMAND_NAME] = microduck_mdp.SingleLegCrouchCommandCfg(
        **command_kwargs,
        jump_prob=1.0,
        fixed_mode=1,
        prepare_s=INITIAL_HOLD_S,
        crouch_s=CROUCH_S,
        extend_s=RETURN_S,
    )

    cfg.rewards.clear()
    task_params = {
        "command_name": COMMAND_NAME,
        "sensor_name": FEET_SENSOR,
        "nonfoot_sensor_name": NONFOOT_SENSOR,
        "asset_cfg": FEET_CFG,
        "height_unit": HEIGHT_UNIT,
        "return_tolerance": RETURN_TOLERANCE,
        "return_hold_s": RETURN_HOLD_S,
    }
    cfg.rewards["crouch_depth_progress"] = RewardTermCfg(
        func=microduck_mdp.single_leg_crouch_depth_progress,
        weight=5.0,
        params=task_params,
    )
    cfg.rewards["return_height_progress"] = RewardTermCfg(
        func=microduck_mdp.single_leg_crouch_return_progress,
        weight=5.0,
        params=task_params,
    )
    cfg.rewards["return_completion"] = RewardTermCfg(
        func=microduck_mdp.single_leg_crouch_completion_reward,
        weight=10.0,
        params=task_params,
    )

    cfg.events["reset_single_leg_crouch"] = EventTermCfg(
        func=microduck_mdp.reset_single_leg_crouch_state,
        mode="reset",
        params={
            "state_bank_path": str(RESET_STATE_BANK),
            "fixed_side": cfg.commands[COMMAND_NAME].fixed_side,
        },
    )

    cfg.terminations.pop("root_too_low", None)
    cfg.terminations["crouch_success"] = TerminationTermCfg(
        func=microduck_mdp.single_leg_crouch_success,
        time_out=False,
        params=task_params,
    )
    cfg.terminations["nonfoot_contact"] = TerminationTermCfg(
        func=microduck_mdp.single_leg_crouch_failure,
        time_out=False,
        params=task_params,
    )

    cfg.events.pop("push_robot", None)
    cfg.metrics.clear()
    for side_name, support_side in (("left", -1), ("right", 1)):
        for metric in (
            "command_count",
            "completion_rate",
            "failure_rate",
            "max_depth",
            "max_rise",
            "final_height_error",
        ):
            cfg.metrics[f"crouch_{metric}_{side_name}"] = MetricsTermCfg(
                func=microduck_mdp.single_leg_crouch_metric,
                reduce="last",
                params={
                    **task_params,
                    "metric": metric,
                    "support_side": support_side,
                },
            )
    cfg.curriculum.clear()
    return cfg


MicroduckSingleLegCrouchRlCfg = deepcopy(MicroduckSingleLegStandRlCfg)
MicroduckSingleLegCrouchRlCfg.experiment_name = "single_leg_crouch"
MicroduckSingleLegCrouchRlCfg.run_name = "single_leg_crouch"
MicroduckSingleLegCrouchRlCfg.max_iterations = 100
MicroduckSingleLegCrouchRlCfg.save_interval = 25
MicroduckSingleLegCrouchRlCfg.actor.distribution_cfg["init_std"] = 0.02
MicroduckSingleLegCrouchRlCfg.algorithm.learning_rate = 5e-5
MicroduckSingleLegCrouchRlCfg.algorithm.entropy_coef = 0.0
MicroduckSingleLegCrouchRlCfg.algorithm.num_learning_epochs = 1
MicroduckSingleLegCrouchRlCfg.algorithm.schedule = "fixed"
