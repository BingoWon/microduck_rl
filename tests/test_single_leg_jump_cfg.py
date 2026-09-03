import json
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import list_tasks, load_runner_cls

from mjlab_microduck.tasks import (
    MicroduckActorWarmStartRunner,
    MicroduckOnPolicyRunner,
    mdp as microduck_mdp,
)
from mjlab_microduck.tasks.microduck_single_leg_jump_env_cfg import (
    EPISODE_LENGTH_S,
    RESET_STATE_BANK,
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


def test_task_uses_jump_warm_start_runner():
    assert "Mjlab-SingleLegJump-Flat-MicroDuck" in list_tasks()
    assert (
        load_runner_cls("Mjlab-SingleLegJump-Flat-MicroDuck")
        is MicroduckActorWarmStartRunner
    )
    assert MicroduckSingleLegJumpRlCfg.experiment_name == "single_leg_jump"


def _runner_stub():
    runner = object.__new__(MicroduckActorWarmStartRunner)
    runner.current_learning_iteration = 0
    runner.env = SimpleNamespace(
        unwrapped=SimpleNamespace(common_step_counter=19)
    )
    return runner


def test_native_jump_checkpoint_uses_full_resume(monkeypatch, tmp_path):
    checkpoint = tmp_path / "native.pt"
    torch.save({"infos": {}}, checkpoint)
    calls = []

    def fake_load(self, path, load_cfg=None, strict=True, map_location=None):
        calls.append(load_cfg)
        self.current_learning_iteration = 123
        return {"native": True}

    monkeypatch.setattr(MicroduckOnPolicyRunner, "load", fake_load)
    runner = _runner_stub()
    assert runner.load(checkpoint) == {"native": True}
    assert calls == [None]
    assert runner.current_learning_iteration == 123
    assert runner.env.unwrapped.common_step_counter == 19


def test_v6_checkpoint_forces_actor_only_warm_start(monkeypatch, tmp_path):
    checkpoint = tmp_path / "v6.pt"
    torch.save({"infos": {"hop_command_pretrained": True}}, checkpoint)
    calls = []

    def fake_load(self, path, load_cfg=None, strict=True, map_location=None):
        calls.append(load_cfg)
        self.current_learning_iteration = 123
        return {"v6": True}

    monkeypatch.setattr(MicroduckOnPolicyRunner, "load", fake_load)
    runner = _runner_stub()
    assert runner.load(checkpoint) == {"v6": True}
    assert calls == [{"actor": True}]
    assert runner.current_learning_iteration == 0
    assert runner.env.unwrapped.common_step_counter == 0


def test_explicit_stand_warm_start_resets_new_command_inputs(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "stand.pt"
    torch.save({"infos": {}}, checkpoint)
    monkeypatch.setenv("SINGLE_LEG_JUMP_ACTOR_WARM_START", "1")
    monkeypatch.setattr(
        MicroduckOnPolicyRunner,
        "load",
        lambda self, *args, **kwargs: {},
    )
    runner = _runner_stub()
    runner.alg = SimpleNamespace(
        actor=SimpleNamespace(
            mlp=[SimpleNamespace(weight=torch.ones(2, 61))],
            obs_normalizer=SimpleNamespace(
                _mean=torch.ones(61),
                _var=torch.full((61,), 2.0),
                _std=torch.full((61,), 3.0),
            ),
        )
    )
    runner.load(checkpoint)
    assert torch.equal(
        runner.alg.actor.mlp[0].weight[:, [48, 50]], torch.zeros(2, 2)
    )
    assert torch.equal(
        runner.alg.actor.obs_normalizer._mean[[48, 50]], torch.zeros(2)
    )
    assert torch.equal(
        runner.alg.actor.obs_normalizer._var[[48, 50]], torch.ones(2)
    )
    assert torch.equal(
        runner.alg.actor.obs_normalizer._std[[48, 50]], torch.ones(2)
    )


def test_invalid_explicit_warm_start_value_is_rejected(monkeypatch, tmp_path):
    checkpoint = tmp_path / "stand.pt"
    torch.save({"infos": {}}, checkpoint)
    monkeypatch.setenv("SINGLE_LEG_JUMP_ACTOR_WARM_START", "sometimes")
    with pytest.raises(ValueError):
        _runner_stub().load(checkpoint)


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


def test_wrong_contact_fails_before_takeoff_once_single_leg_is_established():
    _, _, failed = microduck_mdp.single_leg_jump_transition_flags(
        stage=torch.tensor([0, 0, 0]),
        initial_grounded=torch.tensor([True, True, False]),
        previous_support_contact=torch.tensor([True, True, True]),
        support_contact=torch.tensor([True, True, True]),
        swing_contact=torch.tensor([True, False, True]),
        nonfoot_clear=torch.tensor([True, False, True]),
        upward_velocity=torch.zeros(3),
        peak_height_gain=torch.zeros(3),
        min_takeoff_velocity=0.02,
        min_height_gain=0.003,
    )
    assert failed.tolist() == [True, True, False]


def test_reverse_curriculum_uses_harvested_states_and_anneals_to_standing():
    cfg = make_microduck_single_leg_jump_env_cfg()
    params = cfg.events["reset_single_leg_jump"].params
    assert params["state_bank_path"] == str(RESET_STATE_BANK)
    assert sum(
        params[name]
        for name in (
            "standing_prob",
            "compressed_prob",
            "airborne_prob",
            "landing_prob",
        )
    ) == pytest.approx(1.0)
    stages = cfg.curriculum["jump_reset_mix"].params["param_stages"]
    assert stages[0]["params"] == {
        "standing_prob": 0.60,
        "compressed_prob": 0.25,
        "airborne_prob": 0.075,
        "landing_prob": 0.075,
    }
    assert stages[-1]["params"] == {
        "standing_prob": 1.0,
        "compressed_prob": 0.0,
        "airborne_prob": 0.0,
        "landing_prob": 0.0,
    }
    play = make_microduck_single_leg_jump_env_cfg(play=True)
    assert "jump_reset_mix" not in play.curriculum
    assert play.events["reset_single_leg_jump"].params["standing_prob"] == 1.0


def test_harvested_state_bank_is_balanced_and_well_formed():
    payload = json.loads(RESET_STATE_BANK.read_text())
    assert payload["version"] == 1
    for side in ("left", "right"):
        for category in ("compressed", "airborne", "landing"):
            states = payload["states"][side][category]
            assert states
            for state in states:
                assert len(state["root_pos"]) == 3
                assert len(state["root_quat"]) == 4
                assert len(state["root_lin_vel"]) == 3
                assert len(state["root_ang_vel"]) == 3
                assert len(state["joint_pos"]) == 14
                assert len(state["joint_vel"]) == 14
                assert torch.isfinite(
                    torch.tensor(
                        [
                            *state["root_pos"],
                            *state["root_quat"],
                            *state["root_lin_vel"],
                            *state["root_ang_vel"],
                            *state["joint_pos"],
                            *state["joint_vel"],
                            state["baseline_z"],
                            state["peak_height_gain"],
                        ]
                    )
                ).all()


def test_discovery_shaping_is_bounded_and_anneals_to_zero():
    cfg = make_microduck_single_leg_jump_env_cfg()
    shaping = {
        "jump_compression_progress": 0.5,
        "jump_upward_progress": 0.5,
        "jump_takeoff": 1.0,
        "jump_landing": 1.0,
        "jump_recovery_progress": 1.0,
    }
    assert sum(shaping.values()) == 4.0
    assert cfg.rewards["jump_completion"].weight == 10.0
    assert cfg.rewards["jump_height"].weight == 1.0
    assert cfg.rewards["strict_single_leg_hold"].weight == 2.0
    for name, weight in shaping.items():
        assert cfg.rewards[name].weight == weight
        stages = cfg.curriculum[f"{name}_weight"].params["weight_stages"]
        assert stages[0]["weight"] == weight
        assert stages[-1]["weight"] == 0.0
    assert cfg.rewards["action_rate_l2"].weight == -0.01


def test_frontier_progress_pays_only_new_maximum():
    frontier = torch.tensor([0.0, 0.005, 0.02, 0.005])
    paid = torch.tensor([0.0, 0.0, 0.01, 0.006])
    reward = microduck_mdp._single_leg_jump_frontier_delta(
        frontier, paid, target=0.01
    )
    assert torch.allclose(reward, torch.tensor([0.0, 0.5, 0.0, 0.0]))


def test_frontier_reward_survives_reward_manager_dt_scaling(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp, "_update_single_leg_jump", lambda *args, **kwargs: None
    )
    env = SimpleNamespace(
        step_dt=0.02,
        _slj_max_compression=torch.tensor([0.005]),
        _slj_paid_compression=torch.tensor([0.0]),
    )
    raw_rate = microduck_mdp.single_leg_jump_compression_progress(
        env,
        command_name="twist",
        sensor_name="feet",
        nonfoot_sensor_name="nonfoot",
        asset_cfg=None,
        target_compression=0.01,
    )
    scaled_reward = raw_rate * 0.5 * env.step_dt
    assert torch.allclose(scaled_reward, torch.tensor([0.25]))


def test_terminal_reward_is_exact_and_failed_height_bank_is_discarded(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp, "_update_single_leg_jump", lambda *args, **kwargs: None
    )
    env = SimpleNamespace(
        step_dt=0.02,
        _slj_completed=torch.tensor([True, False]),
        _slj_peak_height_gain=torch.tensor([0.006, 0.010]),
    )
    params = {
        "command_name": "twist",
        "sensor_name": "feet",
        "nonfoot_sensor_name": "nonfoot",
        "asset_cfg": None,
    }
    completion_rate = microduck_mdp.single_leg_jump_completion_reward(env, **params)
    height_rate = microduck_mdp.single_leg_jump_banked_height_reward(
        env, **params, target_height_gain=0.01
    )
    scaled = completion_rate * 10.0 * env.step_dt + height_rate * env.step_dt
    assert torch.allclose(scaled, torch.tensor([10.6, 0.0]))


def test_partial_reset_clears_jump_latches():
    env = SimpleNamespace(num_envs=3, device="cpu")
    microduck_mdp.reset_single_leg_jump_state(
        env, torch.arange(3), state_bank_path=None
    )
    env._slj_completed[:] = True
    env._slj_failed[:] = True
    env._slj_took_off[:] = True
    env._slj_landed[:] = True
    env._slj_peak_height_gain[:] = 0.01
    microduck_mdp.reset_single_leg_jump_state(
        env, torch.tensor([1]), state_bank_path=None
    )
    assert env._slj_completed.tolist() == [True, False, True]
    assert env._slj_failed.tolist() == [True, False, True]
    assert env._slj_took_off.tolist() == [True, False, True]
    assert env._slj_landed.tolist() == [True, False, True]
    assert env._slj_peak_height_gain.tolist() == pytest.approx([0.01, 0.0, 0.01])


def test_training_metrics_are_split_by_support_side():
    cfg = make_microduck_single_leg_jump_env_cfg()
    for name in (
        "single_leg_success",
        "single_leg_success_left",
        "single_leg_success_right",
    ):
        assert cfg.metrics[name].params["required_mode"] == "stand"
    for side in ("left", "right"):
        for metric in (
            "command_count",
            "takeoff_rate",
            "landing_rate",
            "completion_rate",
            "failure_rate",
            "peak_height_gain",
        ):
            term = cfg.metrics[f"jump_{metric}_{side}"]
            assert term.func is microduck_mdp.single_leg_jump_metric
            assert term.reduce == "last"
            assert term.params["support_side"] == (-1 if side == "left" else 1)


def test_training_metrics_return_per_environment_values(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp, "_update_single_leg_jump", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_single_leg_command_state",
        lambda *args: (
            torch.tensor([-1.0, -1.0, 1.0, -1.0]),
            None,
            None,
            None,
        ),
    )
    env = SimpleNamespace(
        num_envs=4,
        command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(
                is_jump=torch.tensor([True, False, True, True])
            )
        ),
        _slj_took_off=torch.tensor([True, True, False, False]),
        _slj_landed=torch.tensor([True, False, False, False]),
        _slj_completed=torch.tensor([False, False, False, False]),
        _slj_failed=torch.tensor([False, False, True, True]),
        _slj_peak_height_gain=torch.tensor([0.01, 0.02, 0.03, 0.04]),
    )
    params = {
        "command_name": "twist",
        "sensor_name": "feet",
        "nonfoot_sensor_name": "nonfoot",
        "asset_cfg": None,
        "support_side": -1,
    }
    expected = {
        "command_count": [1.0, 0.0, 0.0, 1.0],
        "takeoff_rate": [1.0, 0.0, 0.0, 0.0],
        "landing_rate": [1.0, 0.0, 0.0, 0.0],
        "completion_rate": [0.0, 0.0, 0.0, 0.0],
        "failure_rate": [0.0, 0.0, 0.0, 1.0],
        "peak_height_gain": [0.01, 0.0, 0.0, 0.04],
    }
    for metric, values in expected.items():
        actual = microduck_mdp.single_leg_jump_metric(
            env, metric=metric, **params
        )
        assert actual.tolist() == pytest.approx(values)


def test_jump_task_is_documented_in_readme():
    readme = (RESET_STATE_BANK.parents[4] / "README.md").read_text()
    assert "Mjlab-SingleLegJump-Flat-MicroDuck" in readme
    assert "[1, side, -1]" in readme
    assert "[1, side, +1]" in readme
