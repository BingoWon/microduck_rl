import pytest
import torch
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


def test_terminal_rewards_are_banked_until_success():
    cfg = make_microduck_single_leg_jump_env_cfg()
    assert cfg.rewards["strict_single_leg_hold"].params["required_mode"] == "stand"
    assert "failed_episode" not in cfg.rewards
    assert cfg.rewards["jump_completion"].weight == 10.0
    assert cfg.rewards["jump_height"].weight == 1.0
    assert "jump_success" in cfg.terminations
    assert "jump_failure" in cfg.terminations
    assert "reset_single_leg_jump" in cfg.events


def test_transition_requires_true_takeoff_and_same_foot_landing():
    stage = torch.tensor([0, 0, 1, 1, 1])
    initial = torch.tensor([True, True, True, True, True])
    previous_support = torch.tensor([True, True, False, False, False])
    support = torch.tensor([False, False, True, True, False])
    swing = torch.tensor([False, False, False, True, True])
    nonfoot_clear = torch.tensor([True, True, True, True, False])
    vz = torch.tensor([0.03, 0.0, 0.0, 0.0, 0.0])
    peak = torch.tensor([0.0, 0.0, 0.004, 0.004, 0.004])

    takeoff, landing, failed = microduck_mdp.single_leg_jump_transition_flags(
        stage,
        initial,
        previous_support,
        support,
        swing,
        nonfoot_clear,
        vz,
        peak,
        min_takeoff_velocity=0.02,
        min_height_gain=0.003,
    )
    assert takeoff.tolist() == [True, False, False, False, False]
    assert landing.tolist() == [False, False, True, False, False]
    assert failed.tolist() == [False, False, False, True, True]


def test_short_contact_flicker_is_not_a_landing():
    takeoff, landing, failed = microduck_mdp.single_leg_jump_transition_flags(
        stage=torch.tensor([1]),
        initial_grounded=torch.tensor([True]),
        previous_support_contact=torch.tensor([False]),
        support_contact=torch.tensor([True]),
        swing_contact=torch.tensor([False]),
        nonfoot_clear=torch.tensor([True]),
        upward_velocity=torch.tensor([0.0]),
        peak_height_gain=torch.tensor([0.001]),
        min_takeoff_velocity=0.02,
        min_height_gain=0.003,
    )
    assert not bool(takeoff[0])
    assert not bool(landing[0])
    assert not bool(failed[0])
