"""CPU tests for the hostile-terrain walking env (no GPU: cfg construction, terrain compile,
reward knobs, curriculum freeze, ladder rule on a fake env)."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab.terrains.terrain_generator import TerrainGenerator

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.hostile_terrains import HfRubbleTerrainCfg, hostile_subterrains
from mjlab_microduck.tasks.microduck_velocity_hostile_env_cfg import (
    FOOT_TARGET_HEIGHT,
    HIP_ROLL_STD,
    LADDER_ROWS,
    TRACK_LIN_STD,
    make_hostile_terrains_cfg,
    make_microduck_velocity_hostile_env_cfg,
)


@pytest.mark.parametrize("finetune,feet,track", [(True, False, False), (True, True, False), (False, True, False), (True, True, True)])
def test_variants_build(finetune, feet, track):
    cfg = make_microduck_velocity_hostile_env_cfg(finetune=finetune, feet=feet, track=track)
    gen = cfg.scene.terrain.terrain_generator
    assert gen.curriculum is True and gen.num_rows == LADDER_ROWS
    assert set(gen.sub_terrains) == {"flat", "grid", "stairs", "stones", "rubble", "slopes"}
    assert cfg.curriculum["terrain_levels"].func is microduck_mdp.hostile_terrain_levels
    assert cfg.scene.terrain.max_init_terrain_level == (2 if finetune else 0)
    # reward knobs
    fc = cfg.rewards["foot_clearance"].params["target_height"]
    sw = cfg.rewards["foot_swing_height"].params["target_height"]
    hip = cfg.rewards["pose"].params["std_walking"][r".*hip_roll.*"]
    assert (fc, sw, hip) == ((FOOT_TARGET_HEIGHT, FOOT_TARGET_HEIGHT, HIP_ROLL_STD) if feet else (0.02, 0.02, 0.05))
    assert cfg.rewards["pose"].params["std_standing"][r".*hip_roll.*"] == 0.05  # standing stays tight
    std = cfg.rewards["track_linear_velocity"].params["std"]
    assert std == pytest.approx(TRACK_LIN_STD if track else np.sqrt(0.1))
    # penalties keep their sign
    for name in ("foot_clearance", "foot_swing_height", "action_rate_l2", "foot_slip"):
        assert cfg.rewards[name].weight < 0
    assert cfg.rewards["track_linear_velocity"].weight > 0


def test_finetune_freezes_step_curricula():
    ft = make_microduck_velocity_hostile_env_cfg(finetune=True)
    stages = ft.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages == [{"step": 0, "weight": -1.0}]
    standing = ft.curriculum["standing_envs"].params["standing_stages"]
    assert len(standing) == 1 and standing[0]["step"] == 0 and standing[0]["rel_standing_envs"] >= 0.2
    scratch = make_microduck_velocity_hostile_env_cfg(finetune=False)
    assert len(scratch.curriculum["action_rate_weight"].params["weight_stages"]) > 1
    assert scratch.curriculum["action_rate_weight"].params["weight_stages"][0]["weight"] == -0.1


def test_terrain_compiles_on_cpu_and_stays_cheap():
    spec = mujoco.MjSpec()
    gen = TerrainGenerator(make_hostile_terrains_cfg(), device="cpu")
    gen.compile(spec)
    model = spec.compile()
    assert model.ngeom < 6000, model.ngeom
    assert model.nhfield == 3 * LADDER_ROWS  # stones + rubble + slopes per row
    assert gen.terrain_origins.shape == (LADDER_ROWS, 6, 3)


def test_rubble_origin_is_above_the_bumps():
    cfg = hostile_subterrains()["rubble"]
    assert isinstance(cfg, HfRubbleTerrainCfg)
    cfg.size = (4.0, 4.0)
    for d in (0.0, 1.0):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        out = cfg.function(d, spec, np.random.default_rng(0))
        amplitude = (cfg.noise_range_easy[1] - cfg.noise_range_easy[0]) if d == 0 else (cfg.noise_range[1] - cfg.noise_range[0])
        assert out.origin[2] >= amplitude  # surface spans [0, amplitude]; origin at/above its top


def _fake_env(fell, timed_out, tracking_fraction, weight=2.0, T=20.0):
    n = len(fell)
    levels = torch.zeros(n, dtype=torch.long)
    calls = {}

    def update_env_origins(env_ids, move_up, move_down):
        calls["up"], calls["down"] = move_up.clone(), move_down.clone()

    terrain = SimpleNamespace(update_env_origins=update_env_origins, terrain_levels=levels)
    term = SimpleNamespace(get_term=lambda name: {"fell_over": torch.tensor(fell), "time_out": torch.tensor(timed_out)}[name])
    sums = torch.tensor(tracking_fraction) * weight * T
    rew = SimpleNamespace(_episode_sums={"track_linear_velocity": sums}, get_term_cfg=lambda n: SimpleNamespace(weight=weight))
    env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain), termination_manager=term, reward_manager=rew, max_episode_length_s=T)
    return env, calls


def test_ladder_rule():
    #            fell   timeout  tracking
    fell =      [True,  False,   False, False]
    timed_out = [False, True,    True,  False]
    tracking =  [0.9,   0.8,     0.3,   0.9]   # 0: fell → down; 1: good → up; 2: stood still → stay; 3: died otherwise → stay
    env, calls = _fake_env(fell, timed_out, tracking)
    out = microduck_mdp.hostile_terrain_levels(env, torch.arange(4), min_tracking=0.55)
    assert calls["up"].tolist() == [False, True, False, False]
    assert calls["down"].tolist() == [True, False, False, False]
    assert out["promoted"].item() == pytest.approx(0.25) and out["demoted"].item() == pytest.approx(0.25)
