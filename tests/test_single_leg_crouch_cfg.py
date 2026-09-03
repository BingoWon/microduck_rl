from types import SimpleNamespace

import torch
from mjlab.tasks.registry import list_tasks, load_runner_cls

from mjlab_microduck.tasks import (
    MicroduckCrouchWarmStartRunner,
)
from mjlab_microduck.tasks import (
    mdp as microduck_mdp,
)
from mjlab_microduck.tasks.microduck_single_leg_crouch_env_cfg import (
    CROUCH_S,
    EPISODE_LENGTH_S,
    FINAL_HOLD_S,
    HEIGHT_UNIT,
    INITIAL_HOLD_S,
    RESET_STATE_BANK,
    RETURN_HOLD_S,
    RETURN_S,
    RETURN_TOLERANCE,
    MicroduckSingleLegCrouchRlCfg,
    make_microduck_single_leg_crouch_env_cfg,
)


def test_crouch_task_contract():
    cfg = make_microduck_single_leg_crouch_env_cfg()
    command = cfg.commands["twist"]
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 5.5
    assert INITIAL_HOLD_S + CROUCH_S + RETURN_S + FINAL_HOLD_S == 5.5
    assert isinstance(command, microduck_mdp.SingleLegCrouchCommandCfg)
    assert command.jump_prob == 1.0
    assert command.fixed_mode == 1
    assert command.prepare_s == INITIAL_HOLD_S
    assert command.crouch_s == CROUCH_S
    assert command.extend_s == RETURN_S
    assert cfg.events["reset_single_leg_crouch"].params == {
        "state_bank_path": str(RESET_STATE_BANK),
        "fixed_side": 0,
    }


def test_crouch_rewards_and_terminations_are_minimal():
    cfg = make_microduck_single_leg_crouch_env_cfg()
    assert {name: term.weight for name, term in cfg.rewards.items()} == {
        "crouch_depth_progress": 5.0,
        "return_height_progress": 5.0,
        "return_completion": 10.0,
    }
    assert "root_too_low" not in cfg.terminations
    assert "fell_over" in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert "out_of_terrain_bounds" in cfg.terminations
    assert "time_out" in cfg.terminations
    assert "crouch_success" in cfg.terminations
    assert "nonfoot_contact" in cfg.terminations
    assert "push_robot" not in cfg.events
    assert cfg.curriculum == {}


def test_crouch_training_configuration():
    assert "Mjlab-SingleLegCrouch-Flat-MicroDuck" in list_tasks()
    assert (
        load_runner_cls("Mjlab-SingleLegCrouch-Flat-MicroDuck")
        is MicroduckCrouchWarmStartRunner
    )
    assert MicroduckSingleLegCrouchRlCfg.experiment_name == "single_leg_crouch"
    assert MicroduckSingleLegCrouchRlCfg.max_iterations == 100
    assert MicroduckSingleLegCrouchRlCfg.save_interval == 25
    assert (
        MicroduckSingleLegCrouchRlCfg.actor.distribution_cfg["init_std"]
        == 0.02
    )
    assert (
        MicroduckSingleLegCrouchRlCfg.actor.class_name
        == MicroduckSingleLegCrouchRlCfg.critic.class_name
    )
    assert (
        "Teacher"
        not in MicroduckSingleLegCrouchRlCfg.algorithm.class_name
    )
    assert MicroduckSingleLegCrouchRlCfg.algorithm.learning_rate == 5e-5
    assert MicroduckSingleLegCrouchRlCfg.algorithm.entropy_coef == 0.0
    assert MicroduckSingleLegCrouchRlCfg.algorithm.num_learning_epochs == 1
    assert MicroduckSingleLegCrouchRlCfg.algorithm.schedule == "fixed"
    assert MicroduckSingleLegCrouchRlCfg.algorithm.symmetry_cfg is not None


def _crouch_env() -> SimpleNamespace:
    command = torch.tensor([[1.0, -1.0, -1.0]])
    feet = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0, 0.0]])))
    nonfoot = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0.0]])))
    robot = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.12]]))
    )

    class Scene:
        def __init__(self):
            self.sensors = {
                "feet_ground_contact": feet,
                "nonfoot_ground_contact": nonfoot,
            }
            self.terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))

        def __getitem__(self, name):
            assert name == "robot"
            return robot

    term = SimpleNamespace(alpha=torch.ones(1))
    manager = SimpleNamespace(
        get_command=lambda name: command,
        get_term=lambda name: term,
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        common_step_counter=1,
        command_manager=manager,
        scene=Scene(),
    )
    microduck_mdp._init_single_leg_crouch_buffers(env)
    env._slc_baseline_z[:] = 0.12
    env._slc_lowest_z[:] = 0.12
    return env


def _params() -> dict:
    return {
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
        "nonfoot_sensor_name": "nonfoot_ground_contact",
        "asset_cfg": SimpleNamespace(name="robot"),
        "height_unit": HEIGHT_UNIT,
        "return_tolerance": RETURN_TOLERANCE,
        "return_hold_s": RETURN_HOLD_S,
    }


def test_height_progress_is_continuous_uncapped_and_not_repaid():
    env = _crouch_env()
    env.scene["robot"].data.root_link_pos_w[:, 2] = 0.10
    reward = microduck_mdp.single_leg_crouch_depth_progress(env, **_params())
    assert torch.allclose(reward * env.step_dt, torch.tensor([2.0]))

    env.common_step_counter += 1
    reward = microduck_mdp.single_leg_crouch_depth_progress(env, **_params())
    assert torch.equal(reward, torch.zeros(1))

    env.command_manager.get_command("twist")[:, 2] = 1.0
    env.scene["robot"].data.root_link_pos_w[:, 2] = 0.13
    env.common_step_counter += 1
    reward = microduck_mdp.single_leg_crouch_return_progress(env, **_params())
    assert torch.allclose(reward * env.step_dt, torch.tensor([3.0]))


def test_completion_requires_real_crouch_and_height_band_hold():
    env = _crouch_env()
    params = _params()
    env.command_manager.get_command("twist")[:, 2] = 1.0
    env.common_step_counter += 1
    microduck_mdp.single_leg_crouch_return_progress(env, **params)
    env.command_manager.get_command("twist")[:, 2] = 0.0
    for _ in range(30):
        env.common_step_counter += 1
        reward = microduck_mdp.single_leg_crouch_completion_reward(
            env, **params
        )
    assert torch.equal(reward, torch.zeros(1))
    assert not bool(env._slc_completed[0])

    env._slc_best_depth[:] = RETURN_TOLERANCE
    env._slc_lowest_z[:] = env._slc_baseline_z - RETURN_TOLERANCE
    env._slc_return_hold_s.zero_()
    for _ in range(25):
        env.common_step_counter += 1
        reward = microduck_mdp.single_leg_crouch_completion_reward(
            env, **params
        )
    assert torch.allclose(reward * env.step_dt, torch.ones(1))
    assert bool(env._slc_completed[0])


def test_side_metrics_do_not_average_in_other_side_zeroes():
    env = _crouch_env()
    env.num_envs = 2
    env.command_manager = SimpleNamespace(
        get_command=lambda name: torch.tensor(
            [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0]]
        ),
        get_term=lambda name: SimpleNamespace(alpha=torch.ones(2)),
    )
    env.scene.sensors["feet_ground_contact"].data.found = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    env.scene.sensors["nonfoot_ground_contact"].data.found = torch.zeros(2, 1)
    env.scene.terrain.env_origins = torch.zeros(2, 3)
    env.scene["robot"].data.root_link_pos_w = torch.tensor(
        [[0.0, 0.0, 0.11], [0.0, 0.0, 0.10]]
    )
    for name, value in vars(env).copy().items():
        if name.startswith("_slc_") and isinstance(value, torch.Tensor):
            setattr(env, name, value.repeat(2))
    env._slc_baseline_z[:] = 0.12
    env._slc_lowest_z[:] = 0.12
    metric = microduck_mdp.single_leg_crouch_metric(
        env,
        **_params(),
        metric="max_depth",
        support_side=-1,
    )
    assert torch.allclose(metric, torch.full((2,), 0.01))
