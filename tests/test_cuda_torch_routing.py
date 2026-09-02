"""On linux-aarch64 (DGX Spark / GB10) AND on Windows, PyPI's torch wheel is
CPU-ONLY: torch.version.cuda is None -> torch.cuda.device_count() == 0 ->
mjlab's select_gpus() indexes an empty list and dies with
`IndexError: list index out of range` BEFORE the first training step
(mjlab/utils/gpu.py:70).

The fix (pyproject.toml) routes torch to PyTorch's CUDA indexes on exactly
those platforms: cu129 on aarch64 (the CUDA 12.9 toolkit warp 1.12.0
bundles) and cu128 on Windows (cu129 publishes no win_amd64 wheel for 2.9.1;
cu128 is the closest same-major build). It has SILENT break points, locked
in by these tests — in every case `uv sync` succeeds and you only find out
when you launch a run:

1. `torch` must stay a DIRECT dependency: uv applies [tool.uv.sources] to
   direct dependencies only, so deleting the `torch==...` line (which looks
   redundant, since torch already comes in via mjlab/rsl_rl) makes the
   source binding a no-op without any warning.
2. The linux-x86_64 resolution must stay on PyPI, otherwise HF Jobs silently
   switch wheels.
3. Every routed platform must resolve to a CUDA 12.x build, so the zero-copy
   warp<->torch interop stays on one runtime major version.
"""

import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CUDA_INDEX = "https://download.pytorch.org/whl/cu"
_AARCH64 = "sys_platform == 'linux' and platform_machine == 'aarch64'"
_WIN32 = "sys_platform == 'win32'"


def _packages(name):
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    return [p for p in lock["package"] if p["name"] == name]


def _registry(pkg):
    return pkg.get("source", {}).get("registry", "")


def _markers(pkg):
    return " ".join(pkg.get("resolution-markers", []))


def _is_aarch64_entry(pkg):
    m = _markers(pkg)
    return "platform_machine == 'aarch64'" in m and "sys_platform == 'linux'" in m


def _is_win32_entry(pkg):
    return "sys_platform == 'win32'" in _markers(pkg)


def _single(pkgs, pred, what):
    hits = [p for p in pkgs if pred(p)]
    assert len(hits) == 1, f"expected 1 {what} torch entry, found {len(hits)}"
    return hits[0]


def _torch_sources():
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    uv_cfg = pyproject["tool"]["uv"]
    assert "torch" in uv_cfg.get("sources", {}), (
        "[tool.uv.sources] no longer has a torch entry -> aarch64 and Windows "
        "fall back to PyPI's CPU wheel and `train` dies with IndexError in "
        "select_gpus()."
    )
    indexes = {p["name"]: p["url"] for p in uv_cfg.get("index", [])}
    return {src["marker"]: indexes[src["index"]] for src in uv_cfg["sources"]["torch"]}


def test_torch_is_a_direct_dependency():
    """Without this, [tool.uv.sources] for torch is a silent no-op."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert any(d.split("=")[0].split("[")[0].strip() == "torch" for d in deps), (
        "torch must stay in [project.dependencies]: uv applies "
        "[tool.uv.sources] to DIRECT dependencies only. Removing it silently "
        "drops aarch64 and Windows back onto PyPI's CPU-only wheel."
    )


def test_torch_sources_cover_exactly_aarch64_and_windows():
    sources = _torch_sources()
    assert set(sources) == {_AARCH64, _WIN32}, (
        f"torch sources are scoped to {sorted(sources)}; expected exactly the "
        "aarch64 and win32 markers — anything wider would move linux-x86_64 "
        "(HF Jobs) off PyPI."
    )
    for marker, url in sources.items():
        assert url.startswith(_CUDA_INDEX), f"{marker!r} -> {url} is not a PyTorch CUDA index"
        assert url.startswith(_CUDA_INDEX + "12"), (
            f"{marker!r} -> {url} is not a CUDA 12.x index; warp 1.12.0 bundles "
            "toolkit 12.9, so torch must stay on the same runtime major."
        )


def test_lockfile_routes_aarch64_torch_to_cuda_wheels():
    aarch64 = _single(_packages("torch"), _is_aarch64_entry, "aarch64")
    assert _registry(aarch64).startswith(_CUDA_INDEX), (
        f"torch on aarch64 comes from {_registry(aarch64)!r} — a CPU wheel. "
        "Re-run `uv lock` after checking [tool.uv.sources]."
    )
    wheels = " ".join(w["url"] for w in aarch64["wheels"])
    assert "aarch64" in wheels, "no aarch64 wheel in the aarch64 torch entry"
    assert "%2Bcu" in wheels or "+cu" in wheels, (
        "the aarch64 wheel has no +cuXXX local version -> CPU build"
    )


def test_lockfile_routes_windows_torch_to_cuda_wheels():
    win = _single(_packages("torch"), _is_win32_entry, "win32")
    assert _registry(win).startswith(_CUDA_INDEX), (
        f"torch on Windows comes from {_registry(win)!r} — a CPU wheel. "
        "Re-run `uv lock` after checking [tool.uv.sources]."
    )
    wheels = " ".join(w["url"] for w in win["wheels"])
    assert "win_amd64" in wheels, (
        "no win_amd64 wheel in the win32 torch entry: the chosen CUDA index "
        "does not publish Windows builds for this torch version (cu129 does "
        "not) -> `uv sync` fails on Windows."
    )
    assert "%2Bcu12" in wheels or "+cu12" in wheels, (
        "the Windows wheel is not a +cu12x build (CPU, or a CUDA 13 build "
        "that mismatches warp's bundled toolkit)"
    )


def test_linux_x86_64_resolution_stays_on_pypi():
    """HF Jobs run on linux-x86_64: their resolution must not move."""
    others = [
        p
        for p in _packages("torch")
        if not _is_aarch64_entry(p) and not _is_win32_entry(p)
    ]
    assert others, "no linux-x86_64 torch entry found"
    for pkg in others:
        assert _registry(pkg) == "https://pypi.org/simple", (
            f"linux-x86_64 torch moved to {_registry(pkg)!r} — HF Jobs would "
            "switch wheels."
        )
        assert "+cu" not in pkg["version"], "linux-x86_64 torch must not be CUDA-pinned"
        markers = _markers(pkg)
        assert "sys_platform == 'linux'" in markers and "!= 'aarch64'" in markers, (
            f"the PyPI torch entry no longer selects linux-x86_64: {markers!r}"
        )


def test_torch_version_identical_across_platforms():
    """The fix changes only the wheel's SOURCE, not its version: the CUDA
    index carries newer builds than the PyPI pin, so a `>=` drags torch
    2.9.1 -> 2.13.0 with nothing having validated that bump."""
    versions = {p["version"].split("+")[0] for p in _packages("torch")}
    assert len(versions) == 1, f"torch versions diverge across platforms: {versions}"


def _has_nvidia_gpu():
    return (
        shutil.which("nvidia-smi") is not None
        and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    )


def _on_routed_platform():
    return sys.platform == "win32" or (
        sys.platform == "linux" and platform.machine() == "aarch64"
    )


@pytest.mark.skipif(
    not (_on_routed_platform() and _has_nvidia_gpu()),
    reason="not a linux-aarch64 or Windows machine with an NVIDIA GPU",
)
def test_installed_torch_actually_sees_the_gpu():
    """Direct reproduction of the crash: this is exactly what select_gpus() reads."""
    import torch

    assert torch.cuda.device_count() > 0, (
        f"torch {torch.__version__} (cuda={torch.version.cuda}) sees no GPU "
        "although nvidia-smi reports one -> select_gpus() will raise IndexError."
    )
