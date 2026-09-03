import pytest
from mjlab.tasks.registry import list_tasks, load_runner_cls

from mjlab_microduck.tasks import (
    MicroduckActorWarmStartRunner,
    mdp as microduck_mdp,
)
from mjlab_microduck.tasks.microduck_single_leg_jump_env_cfg import (
    EPISODE_LENGTH_S,
    MicroduckSingleLegJumpRlCfg,
    make_microduck_single_leg_jump_env_cfg,
)


def test_task_skeleton_keeps_shared_contract():
    cfg = make_microduck_single_leg_jump_env_cfg()
    command = cfg.commands["twist"]
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 6.0
    assert isinstance(command, microduck_mdp.SingleLegJumpCommandCfg)
    assert command.jump_prob == 0.75
    assert command.prepare_s == 1.5
    assert command.crouch_s == 0.22
    assert command.extend_s == 0.12
    assert cfg.observations["actor"].terms["head_command"].params["dim"] == 4
    assert cfg.observations["actor"].terms["body_command"].params["dim"] == 6


def test_four_semantic_commands_share_one_actor():
    command = make_microduck_single_leg_jump_env_cfg().commands["twist"]
    assert command.left_prob == 0.5
    assert command.ranges.lin_vel_y == (-1.0, 1.0)
    assert command.fixed_mode == -1
    assert MicroduckSingleLegJumpRlCfg.algorithm.symmetry_cfg is not None


def test_play_mode_can_be_fixed(monkeypatch):
    monkeypatch.setenv("SINGLE_LEG_JUMP_PLAY_MODE", "stand")
    assert make_microduck_single_leg_jump_env_cfg(play=True).commands[
        "twist"
    ].fixed_mode == 0
    monkeypatch.setenv("SINGLE_LEG_JUMP_PLAY_MODE", "jump")
    assert make_microduck_single_leg_jump_env_cfg(play=True).commands[
        "twist"
    ].fixed_mode == 1
    monkeypatch.setenv("SINGLE_LEG_JUMP_PLAY_MODE", "invalid")
    with pytest.raises(ValueError):
        make_microduck_single_leg_jump_env_cfg(play=True)


def test_task_uses_actor_only_warm_start_runner():
    assert "Mjlab-SingleLegJump-Flat-MicroDuck" in list_tasks()
    assert (
        load_runner_cls("Mjlab-SingleLegJump-Flat-MicroDuck")
        is MicroduckActorWarmStartRunner
    )
    assert MicroduckSingleLegJumpRlCfg.experiment_name == "single_leg_jump"
