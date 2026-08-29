"""Sub-terrains for the hostile-terrain walking task (2026-08-29).

Plain-English glossary used across this file:
  * "terrain patch": one square of ground (here 4 m × 4 m) with one kind of obstacle on it.
  * "difficulty": a number from 0 to 1. Every obstacle size below is interpolated with it
    (0 = the easy end, 1 = the hard end). The curriculum ("ladder") decides which
    difficulty row each robot lives on.
  * "heightfield": ground described as a grid of heights (one collision object, cheap).
    Box terrains are made of many boxes (one collision object per box, expensive).

These configs are the SAME objects previewed by `notes/tools/duck_terrain.py`, so the
studio pictures are what the GPU environment compiles.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainOutput


@dataclass(kw_only=True)
class HfRubbleTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
    """Continuous bumps ("rubble", a gravel/grass proxy) whose height scales with difficulty.

    mjlab's HfRandomUniformTerrainCfg ignores difficulty. Here ``noise_range`` is the range at
    difficulty 1 and ``noise_range_easy`` the range at difficulty 0; in between we interpolate.
    The spawn origin is raised to the highest bump so the robot never starts with a foot
    inside the ground (mjlab puts the origin at the MEAN height).
    """

    noise_range_easy: tuple[float, float] = (-0.005, 0.005)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        d = float(np.clip(difficulty, 0.0, 1.0))
        lo = self.noise_range_easy[0] + d * (self.noise_range[0] - self.noise_range_easy[0])
        hi = self.noise_range_easy[1] + d * (self.noise_range[1] - self.noise_range_easy[1])
        saved = self.noise_range
        self.noise_range = (lo, hi)
        try:
            out = super().function(difficulty, spec, rng)
        finally:
            self.noise_range = saved
        # mjlab places the surface between 0 and (hi - lo) and the origin at the mean; lift the
        # origin to the top of the tallest bump (+ a hair) so every spawn starts above ground.
        out.origin[2] = max(float(out.origin[2]), hi - lo) + 0.002
        return out


def hostile_subterrains(scale: float = 1.0) -> dict[str, terrain_gen.SubTerrainCfg]:
    """The menu used for training. ``scale`` multiplies every obstacle size (1.0 = the numbers
    below, agreed with Rémi as "about 2× today's terrain at the hard end").

    Row 0 of the ladder (difficulty ≈ 0) is deliberately at or below today's training terrain
    (1 cm grid cells, 1.5 cm stairs), so a fine-tuned policy starts on ground it already walks.
    """
    s = scale
    return {
        # Today's terrain, kept so the policy never forgets it (proportion = share of robots).
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.15),
        "grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.015 * s),
            platform_width=1.0, border_width=0.25),
        # Pyramid stairs: 12 cm treads, step height 1 cm → 3 cm.
        "stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.20, step_height_range=(0.01 * s, 0.03 * s), step_width=0.12,
            platform_width=0.8, border_width=0.2),
        # Flat stones 6–16 cm wide, 0.8 cm → 2.5 cm high (heightfield, 2 cm resolution).
        "stones": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20, obstacle_width_range=(0.06, 0.16),
            obstacle_height_range=(0.008 * s, 0.025 * s), num_obstacles=140,
            obstacle_height_mode="fixed", horizontal_scale=0.02, vertical_scale=0.001,
            platform_width=0.6, border_width=0.2, origin_z_offset=0.025 * s + 0.002),
        # Rubble: bumps every 6 cm, ±0.5 cm → ±2 cm.
        "rubble": HfRubbleTerrainCfg(
            proportion=0.15, noise_range=(-0.02 * s, 0.02 * s), noise_range_easy=(-0.005 * s, 0.005 * s),
            noise_step=0.002, downsampled_scale=0.06, horizontal_scale=0.02, vertical_scale=0.001,
            border_width=0.2),
        # Pyramid slopes: 3° → 9° (rise/run 0.05 → 0.16).
        "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.05 * s, 0.16 * s), platform_width=0.8,
            horizontal_scale=0.05, vertical_scale=0.001),
    }
