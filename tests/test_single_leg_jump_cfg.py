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
    TRANSITION_EPISODE_LENGTH_S,
    MicroduckSingleLegJumpRlCfg,
    MicroduckSingleLegJumpTransitionRlCfg,
    make_microduck_single_leg_jump_env_cfg,
    make_microduck_single_leg_jump_transition_env_cfg,
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


def test_transition_task_trains_return_to_resume_side():
    cfg = make_microduck_single_leg_jump_transition_env_cfg()
    command = cfg.commands["twist"]
    assert cfg.episode_length_s == TRANSITION_EPISODE_LENGTH_S == 10.0
    assert isinstance(
        command, microduck_mdp.SingleLegJumpTransitionCommandCfg
    )
    assert command.transaction_jump_prob == 0.75
    assert command.cross_side_prob == 0.5
    assert command.initial_hold_s == 1.5
    assert command.recovery_s == 1.5
    assert cfg.events["reset_single_leg_jump"].params == {
        **make_microduck_single_leg_jump_env_cfg().events[
            "reset_single_leg_jump"
        ].params,
        "standing_prob": 1.0,
        "compressed_prob": 0.0,
        "airborne_prob": 0.0,
        "fixed_side": 0,
    }
    assert "jump_reset_mix" not in cfg.curriculum
    assert "jump_success" not in cfg.terminations
    assert "transaction_success" in cfg.terminations
    assert cfg.rewards["return_completion"].weight == 5.0
    assert (
        MicroduckSingleLegJumpTransitionRlCfg.experiment_name
        == "single_leg_jump_transitions"
    )
    assert "Mjlab-SingleLegJumpTransitions-Flat-MicroDuck" in list_tasks()


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
    assert MicroduckSingleLegJumpRlCfg.actor.distribution_cfg["init_std"] == 0.005
    assert MicroduckSingleLegJumpRlCfg.algorithm.learning_rate == 5e-5
    assert MicroduckSingleLegJumpRlCfg.algorithm.entropy_coef == 0.0
    assert MicroduckSingleLegJumpRlCfg.algorithm.schedule == "fixed"


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
    monkeypatch.setenv("SINGLE_LEG_JUMP_ACTOR_WARM_START", "stand")
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


def test_actor_warm_start_preserves_jump_command_inputs(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "jump.pt"
    torch.save({"infos": {}}, checkpoint)
    monkeypatch.setenv("SINGLE_LEG_JUMP_ACTOR_WARM_START", "actor")
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
        runner.alg.actor.mlp[0].weight[:, [48, 50]], torch.ones(2, 2)
    )
    assert torch.equal(
        runner.alg.actor.obs_normalizer._mean[[48, 50]], torch.ones(2)
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


def test_full_takeoff_landing_recovery_transition(monkeypatch):
    class Scene(dict):
        pass

    command = torch.tensor([[1.0, -1.0, 0.0]])
    term = SimpleNamespace(is_jump=torch.tensor([True]))
    command_manager = SimpleNamespace(
        get_command=lambda name: command,
        get_term=lambda name: term,
    )
    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.12]]),
        root_link_lin_vel_w=torch.zeros(1, 3),
    )
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        common_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        command_manager=command_manager,
        scene=scene,
        _test_contacts=torch.tensor([[1.0, 0.0]]),
        _test_recovery_valid=torch.tensor([False]),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_single_leg_command_state",
        lambda *args: (
            torch.tensor([-1.0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1.0]),
        ),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_single_leg_contacts",
        lambda test_env, sensor_name: test_env._test_contacts,
    )
    monkeypatch.setattr(
        microduck_mdp,
        "any_contact_cost",
        lambda *args: torch.zeros(1),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "single_leg_success_state",
        lambda test_env, **kwargs: test_env._test_recovery_valid.float(),
    )
    params = {
        "command_name": "twist",
        "sensor_name": "feet",
        "nonfoot_sensor_name": "nonfoot",
        "asset_cfg": SimpleNamespace(name="robot"),
    }

    microduck_mdp.single_leg_jump_success(env, **params)
    assert env._slj_initial_grounded.tolist() == [True]

    env.common_step_counter = 1
    command[:, 2] = -1.0
    microduck_mdp.single_leg_jump_success(env, **params)
    assert env._slj_attempt_started.tolist() == [True]

    env.common_step_counter = 2
    command[:, 2] = 1.0
    data.root_link_pos_w[:, 2] = 0.115
    data.root_link_lin_vel_w[:, 2] = 0.05
    microduck_mdp.single_leg_jump_success(env, **params)

    env.common_step_counter = 3
    env._test_contacts[:] = torch.tensor([[0.0, 0.0]])
    data.root_link_pos_w[:, 2] = 0.124
    assert not bool(microduck_mdp.single_leg_jump_success(env, **params)[0])
    assert env._slj_takeoff_event.tolist() == [True]
    assert env._slj_took_off.tolist() == [True]

    env.common_step_counter = 4
    command[:, 2] = 0.0
    env._test_contacts[:] = torch.tensor([[1.0, 0.0]])
    env._test_recovery_valid[:] = True
    data.root_link_pos_w[:, 2] = 0.123
    data.root_link_lin_vel_w[:, 2] = -0.02
    assert not bool(microduck_mdp.single_leg_jump_success(env, **params)[0])
    assert env._slj_landing_event.tolist() == [True]
    assert env._slj_landed.tolist() == [True]

    for step in range(5, 29):
        env.common_step_counter = step
        success = microduck_mdp.single_leg_jump_success(env, **params)
    assert bool(success[0])
    assert env._slj_completion_event.tolist() == [True]
    assert not env._slj_failed.any()

    env.common_step_counter = 29
    microduck_mdp.single_leg_jump_success(env, **params)
    assert env._slj_completion_event.tolist() == [False]

    microduck_mdp.reset_single_leg_jump_state(
        env, torch.tensor([0]), state_bank_path=None
    )
    term.is_jump[:] = False
    term.transaction_is_jump = torch.tensor([True])
    term.returning = torch.tensor([True])
    env.common_step_counter = 30
    assert bool(microduck_mdp.single_leg_jump_failure(env, **params)[0])


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
        )
    ) == pytest.approx(1.0)
    stages = cfg.curriculum["jump_reset_mix"].params["param_stages"]
    assert stages[0]["params"] == {
        "standing_prob": 0.60,
        "compressed_prob": 0.25,
        "airborne_prob": 0.15,
    }
    assert stages[-1]["params"] == {
        "standing_prob": 1.0,
        "compressed_prob": 0.0,
        "airborne_prob": 0.0,
    }
    play = make_microduck_single_leg_jump_env_cfg(play=True)
    assert "jump_reset_mix" not in play.curriculum
    assert play.events["reset_single_leg_jump"].params["standing_prob"] == 1.0


def test_harvested_state_bank_is_balanced_and_well_formed():
    payload = json.loads(RESET_STATE_BANK.read_text())
    assert payload["version"] == 2
    spec = make_microduck_single_leg_jump_env_cfg(
        play=True
    ).scene.entities["robot"].spec_fn()
    joint_ranges = [
        tuple(joint.range)
        for joint in spec.joints
        if joint.name != "trunk_base_freejoint"
        and not joint.name.startswith("passive_")
    ]
    assert len(joint_ranges) == 14
    for side in ("left", "right"):
        serialized = set()
        for category in ("standing", "compressed", "airborne"):
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
                assert torch.linalg.vector_norm(
                    torch.tensor(state["root_quat"])
                ).item() == pytest.approx(1.0, abs=1e-5)
                for position, (lower, upper) in zip(
                    state["joint_pos"], joint_ranges, strict=True
                ):
                    assert lower - 1e-5 <= position <= upper + 1e-5
                for position, velocity, (lower, upper) in zip(
                    state["joint_pos"],
                    state["joint_vel"],
                    joint_ranges,
                    strict=True,
                ):
                    if position <= lower + 1e-5:
                        assert velocity >= 0.0
                    if position >= upper - 1e-5:
                        assert velocity <= 0.0
                assert not state["swing_contact"]
                assert not state["nonfoot_contact"]
                if category == "standing":
                    assert state["support_contact"]
                elif category == "compressed":
                    assert state["support_contact"]
                    assert state["root_lin_vel"][2] <= 0.0
                    assert (
                        state["baseline_z"] - state["root_pos"][2]
                        >= 0.003
                    )
                elif category == "airborne":
                    assert not state["support_contact"]
                    assert state["root_lin_vel"][2] >= 0.02
                key = json.dumps(state, sort_keys=True)
                assert key not in serialized
                serialized.add(key)


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
        _slj_completion_event=torch.tensor([True, False]),
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
    env._slj_completion_event[:] = True
    env._slj_peak_height_gain[:] = 0.01
    microduck_mdp.reset_single_leg_jump_state(
        env, torch.tensor([1]), state_bank_path=None
    )
    assert env._slj_completed.tolist() == [True, False, True]
    assert env._slj_failed.tolist() == [True, False, True]
    assert env._slj_took_off.tolist() == [True, False, True]
    assert env._slj_landed.tolist() == [True, False, True]
    assert env._slj_completion_event.tolist() == [True, False, True]
    assert env._slj_peak_height_gain.tolist() == pytest.approx([0.01, 0.0, 0.01])


def test_banked_reset_aligns_command_side_mode_and_phase(monkeypatch):
    def fake_stand_resample(term, env_ids):
        term.vel_command_b[env_ids] = 0.0
        term.vel_command_b[env_ids, 1] = 1.0
        term._alpha[env_ids] = 0.0

    monkeypatch.setattr(
        microduck_mdp.SingleLegStandCommand,
        "_resample_command",
        fake_stand_resample,
    )
    term = object.__new__(microduck_mdp.SingleLegJumpCommand)
    term.cfg = SimpleNamespace(
        fixed_mode=0,
        prepare_s=1.5,
        crouch_s=0.22,
        extend_s=0.12,
    )
    term._jump_prob = 0.75
    term._env = SimpleNamespace(
        device="cpu",
        _slj_reset_kind=torch.tensor([0, 1, 2]),
        _slj_reset_side=torch.tensor([-1.0, 1.0, -1.0]),
    )
    term.vel_command_b = torch.zeros(3, 3)
    term._alpha = torch.zeros(3)
    term._is_jump = torch.ones(3, dtype=torch.bool)
    term._elapsed = torch.full((3,), 99.0)

    term._resample_command(torch.arange(3))

    assert term.vel_command_b[:, 1].tolist() == [-1.0, 1.0, -1.0]
    assert term._alpha.tolist() == [1.0, 1.0, 1.0]
    assert term._is_jump.tolist() == [False, True, True]
    assert term._elapsed.tolist() == pytest.approx([0.0, 1.72, 1.84])


def test_transition_command_runs_target_then_returns_resume(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp.SingleLegStandCommand,
        "compute",
        lambda term, dt: None,
    )
    term = object.__new__(microduck_mdp.SingleLegJumpTransitionCommand)
    term.cfg = SimpleNamespace(
        initial_hold_s=1.5,
        prepare_s=1.5,
        crouch_s=0.22,
        extend_s=0.12,
        recovery_s=1.5,
    )
    term._elapsed = torch.zeros(1)
    term._resume_side = torch.tensor([-1.0])
    term._target_side = torch.tensor([1.0])
    term._transaction_is_jump = torch.tensor([True])
    term._is_jump = torch.tensor([False])
    term._returning = torch.tensor([False])
    term._alpha = torch.ones(1)
    term.vel_command_b = torch.zeros(1, 3)
    term.vel_command_b[:, 1] = -1.0

    cases = (
        (0.0, [0.0, -1.0, 0.0], False),
        (1.5, [1.0, 1.0, 0.0], False),
        (3.0, [1.0, 1.0, -1.0], False),
        (3.23, [1.0, 1.0, 1.0], False),
        (4.84, [0.0, -1.0, 0.0], True),
    )
    for elapsed, expected, returning in cases:
        term._elapsed[:] = elapsed
        term.compute(0.0)
        assert term.vel_command_b[0].tolist() == pytest.approx(expected)
        assert bool(term.returning[0]) is returning


def test_completion_reward_is_an_event_not_a_post_success_jackpot(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp, "_update_single_leg_jump", lambda *args, **kwargs: None
    )
    env = SimpleNamespace(
        step_dt=0.02,
        _slj_completed=torch.tensor([True]),
        _slj_completion_event=torch.tensor([False]),
    )
    reward = microduck_mdp.single_leg_jump_completion_reward(
        env,
        command_name="twist",
        sensor_name="feet",
        nonfoot_sensor_name="nonfoot",
        asset_cfg=None,
    )
    assert reward.tolist() == [0.0]


def test_return_reward_is_paid_once_after_a_valid_return(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp, "_update_single_leg_jump", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        microduck_mdp,
        "single_leg_success_state",
        lambda *args, **kwargs: torch.ones(1),
    )
    term = SimpleNamespace(
        transaction_is_jump=torch.tensor([True]),
        returning=torch.tensor([True]),
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.5,
        common_step_counter=0,
        episode_length_buf=torch.tensor([2]),
        command_manager=SimpleNamespace(get_term=lambda name: term),
        _slj_completed=torch.tensor([True]),
    )
    params = {
        "command_name": "twist",
        "sensor_name": "feet",
        "nonfoot_sensor_name": "nonfoot",
        "asset_cfg": None,
        "return_hold_s": 1.0,
    }
    assert microduck_mdp.single_leg_jump_return_reward(
        env, **params
    ).tolist() == [0.0]
    env.common_step_counter = 1
    assert microduck_mdp.single_leg_jump_return_reward(
        env, **params
    ).tolist() == [2.0]
    env.common_step_counter = 2
    assert microduck_mdp.single_leg_jump_return_reward(
        env, **params
    ).tolist() == [0.0]
    assert env._slj_return_completed.tolist() == [True]


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
    assert "Mjlab-SingleLegJumpTransitions-Flat-MicroDuck" in readme
    assert "[1, side, -1]" in readme
    assert "[1, side, +1]" in readme
    assert "left stand -> right jump -> left stand" in readme
    assert "SINGLE_LEG_JUMP_ACTOR_WARM_START=actor" in readme
    assert "runtime leg action low-pass disabled" in readme
