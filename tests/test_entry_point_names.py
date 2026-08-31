"""Console-script names must not collide with mjlab's.

Two installed distributions declaring the same `[project.scripts]` name is
last-writer-wins: whichever is installed last writes `.venv/bin/<name>`, and
the other one silently disappears. This project once declared `train` — meant
to "shadow" mjlab's `train` and add a --hf-jobs flag — and `uv sync` put
mjlab's shim there instead, so `uv run train ... --hf-jobs` hit mjlab's tyro
parser and died with `Unrecognized options: --hf-jobs` (2026-08-31).

Nothing warns about this: `uv sync` succeeds, the flag is simply gone.
"""

import tomllib
from importlib.metadata import distribution
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _our_scripts():
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    return pyproject["project"].get("scripts", {})


def _mjlab_scripts():
    return {
        ep.name
        for ep in distribution("mjlab").entry_points
        if ep.group == "console_scripts"
    }


def test_no_script_name_collides_with_mjlab():
    clashes = set(_our_scripts()) & _mjlab_scripts()
    assert not clashes, (
        f"[project.scripts] re-declares mjlab's console script(s) {sorted(clashes)}. "
        "This does NOT shadow mjlab: install order decides which shim lands in "
        "bin/, so the command silently becomes mjlab's (or ours) at random. "
        "Pick a distinct name instead."
    )


def test_hf_submission_entry_point_exists():
    """The documented remote-training command must keep resolving."""
    scripts = _our_scripts()
    assert "train-hf" in scripts, (
        "`train-hf` is the documented HF Jobs command (AGENTS.md, README.md, "
        "scripts/hf/README.md) — renaming it breaks those docs."
    )
    module, _, attr = scripts["train-hf"].partition(":")
    mod = __import__(module, fromlist=[attr])
    assert callable(getattr(mod, attr)), f"{scripts['train-hf']} is not callable"
