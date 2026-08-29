"""Hostile-terrain walking (2026-08-29): the walking recipe on a harder, laddered terrain menu.

What this file does, in plain English
-------------------------------------
It takes the normal walking environment (`make_microduck_velocity_env_cfg(rough=True)`, the
recipe that produced the shipped `alpha_walking` policy) and changes four things:

1. **The ground.** Instead of one mild mix (1 cm grid + 1.5 cm stairs), the robot trains on a
   "ladder" of terrain rows (see `hostile_terrains.py`): flat, grid, stairs, stones, rubble,
   slopes. Row 0 is about today's difficulty; the last row is about twice as hard.

2. **A ladder that actually moves.** A "curriculum" is a rule that changes the task difficulty
   during training. The old rule needed the robot to walk 4 m in one episode, which almost
   never happened, so nobody ever climbed. The new rule (`hostile_terrain_levels` in mdp.py):
   fall → go one row down; finish the 20 s episode upright while tracking the commanded speed
   reasonably → go one row up; otherwise stay.

3. **Two reward changes** (switchable, so they can be tested one at a time):
   - `feet`: the two foot-height terms asked for a 2 cm swing and punished higher swings.
     A 3 cm step needs a higher swing. Target raised to 3.5 cm. The hip-roll posture
     tolerance is also loosened (0.05 → 0.12 rad): with no ankle roll, hip roll is the only
     joint that can adapt sideways to uneven ground.
   - `track`: the speed-tracking reward is made stricter (std 0.32 → 0.22 m/s) so "stand
     still instead of walking" earns less. Riskier; tested separately.
   - `progress` (added after run A, 2026-08-30): run A learned to walk in circles on the flat
     spawn platform — speed tracking is measured in the robot's own frame, so circling tracks
     perfectly and survives, and the ladder promoted it. Two counters: promotion now also
     requires real displacement (≥ 30 % of the commanded distance), and turning while
     commanded straight costs more (yaw-tracking std 0.71 → 0.39 rad/s).

4. **Fine-tuning mode.** When resuming from the shipped policy's checkpoint, every
   iteration-keyed curriculum (action-rate tax ramp, standing fraction, head ranges, CoM
   randomization ramps) is frozen at its FINAL stage from step 0 — the gait already exists,
   we do not want the smoothing tax to drop back to −0.1 and let it turn jerky.

Glossary: "reward term" = one line in the score the policy maximizes; "std" = the width of a
bell-curve reward (a wider std forgives larger errors); "episode" = one 20 s attempt.
"""
from __future__ import annotations

import dataclasses
import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.hostile_terrains import hostile_subterrains
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Reward knobs (see the docstring).
FOOT_TARGET_HEIGHT = 0.035   # m, was 0.02
HIP_ROLL_STD = 0.12          # rad, was 0.05 (walking / running regimes only; standing stays tight)
TRACK_LIN_STD = math.sqrt(0.05)  # m/s, was sqrt(0.1)
TRACK_ANG_STD = math.sqrt(0.15)  # rad/s, was sqrt(0.5): a 0.3 rad/s unwanted turn costs 45 % instead of 16 % of the yaw reward
MIN_PROGRESS = 0.3               # promotion needs ≥ 30 % of the commanded distance actually covered (anti-circling)
# Terrain ladder.
LADDER_ROWS = 8              # difficulty rows: row r has difficulty in [r/8, (r+1)/8)
PATCH_SIZE = 4.0             # m, one terrain patch (the base env used 8 m; 4 m = 4× fewer boxes)
MIN_TRACKING_TO_PROMOTE = 0.55   # fraction of the max speed-tracking reward, averaged over the episode
SPAWN_XY = 0.2               # m, random spawn offset from the patch centre (base env: 0.5)


def make_hostile_terrains_cfg(rows: int = LADDER_ROWS, scale: float = 1.0) -> TerrainGeneratorCfg:
    """One column per terrain type, ``rows`` difficulty rows (curriculum=True is what makes rows
    mean difficulty; with curriculum=False every patch gets a random difficulty)."""
    return TerrainGeneratorCfg(
        size=(PATCH_SIZE, PATCH_SIZE),
        border_width=4.0,
        num_rows=rows,
        num_cols=6,          # ignored in curriculum mode (one column per sub-terrain) but kept explicit
        curriculum=True,
        sub_terrains=hostile_subterrains(scale),
        add_lights=False,
    )


def _freeze_step_curricula(cfg) -> None:
    """Fine-tune helper: every curriculum stage list ``[{step, ...}, ...]`` collapses to its last
    stage applied at step 0. Performance-based curricula (the terrain ladder) have no stage list
    and are untouched."""
    for term in cfg.curriculum.values():
        for key, val in list(term.params.items()):
            if isinstance(val, list) and val and isinstance(val[0], dict) and "step" in val[0]:
                last = dict(val[-1])
                last["step"] = 0
                term.params[key] = [last]


def make_microduck_velocity_hostile_env_cfg(
    play: bool = False,
    finetune: bool = True,
    feet: bool = True,
    track: bool = False,
    progress: bool = False,
    terrain_scale: float = 1.0,
    max_init_level: int | None = None,
):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=True)

    # --- ground -----------------------------------------------------------------------------
    cfg.scene.terrain.terrain_generator = make_hostile_terrains_cfg(rows=4 if play else LADDER_ROWS, scale=terrain_scale)
    if max_init_level is None:
        max_init_level = 2 if finetune else 0   # rows a robot can START on (0-based, inclusive)
    cfg.scene.terrain.max_init_terrain_level = max_init_level
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=microduck_mdp.hostile_terrain_levels,
        params={"min_tracking": MIN_TRACKING_TO_PROMOTE, "min_progress": MIN_PROGRESS if progress else 0.0},
    )
    pose_range = cfg.events["reset_base"].params["pose_range"]
    pose_range["x"] = (-SPAWN_XY, SPAWN_XY)
    pose_range["y"] = (-SPAWN_XY, SPAWN_XY)

    # --- rewards ----------------------------------------------------------------------------
    if feet:
        cfg.rewards["foot_clearance"].params["target_height"] = FOOT_TARGET_HEIGHT
        cfg.rewards["foot_swing_height"].params["target_height"] = FOOT_TARGET_HEIGHT
        for regime in ("std_walking", "std_running"):
            stds = dict(cfg.rewards["pose"].params[regime])
            stds[r".*hip_roll.*"] = HIP_ROLL_STD
            cfg.rewards["pose"].params[regime] = stds
    if track:
        cfg.rewards["track_linear_velocity"].params["std"] = TRACK_LIN_STD
    if progress:
        # anti-circling, part 2: turning while commanded straight must cost more (run A lesson)
        cfg.rewards["track_angular_velocity"].params["std"] = TRACK_ANG_STD

    # --- fine-tune: curricula at their final stage from step 0 -------------------------------
    if finetune:
        _freeze_step_curricula(cfg)
    return cfg


MicroduckHostileRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    experiment_name="velocity_hostile",
    run_name="hostile",
)
