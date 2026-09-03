#!/usr/bin/env python3
"""Keep the local viewer on the newest complete remote checkpoint."""

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from shlex import quote


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def newest_checkpoint(host: str, remote_run: str) -> tuple[int, str, int]:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            (
                f"find {quote(remote_run)} -maxdepth 1 -type f "
                "-name 'model_*.pt' -printf '%f %s\\n'"
            ),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    candidates = []
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"model_(\d+)\.pt (\d+)", line)
        if match:
            step, size = map(int, match.groups())
            candidates.append((step, f"{remote_run}/model_{step}.pt", size))
    if not candidates:
        raise RuntimeError("remote run has no checkpoints")
    return max(candidates)


def valid_checkpoint(path: Path, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return archive.testzip() is None


def sync_checkpoint(
    host: str,
    remote_path: str,
    expected_size: int,
    local_dir: Path,
) -> Path:
    destination = local_dir / Path(remote_path).name
    if valid_checkpoint(destination, expected_size):
        return destination
    temporary = destination.with_suffix(".part")
    subprocess.run(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            f"{host}:{remote_path}",
            str(temporary),
        ],
        timeout=120,
        check=True,
    )
    if not valid_checkpoint(temporary, expected_size):
        raise RuntimeError(f"incomplete checkpoint: {temporary}")
    temporary.replace(destination)
    return destination


def stop_viewer(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def port_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def start_viewer(
    checkpoint: Path,
    port: int,
    device: str,
    log_path: Path,
) -> subprocess.Popen:
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        [
            "/usr/bin/nice",
            "-n",
            "10",
            sys.executable,
            str(Path(__file__).with_name("play_checkpoint.py")),
            str(checkpoint),
            "--port",
            str(port),
            "--device",
            device,
        ],
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError(f"viewer exited with code {process.returncode}")
        if port_ready(port):
            return process
        time.sleep(0.1)
    stop_viewer(process)
    raise RuntimeError("viewer did not open its port")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("remote_run")
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("state_dir", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.state_dir / "viewer-status.json"
    viewer = None
    shown_step = -1

    while True:
        try:
            step, remote_path, size = newest_checkpoint(
                args.host,
                args.remote_run,
            )
            viewer_alive = viewer is not None and viewer.poll() is None
            if step != shown_step or not viewer_alive:
                checkpoint = sync_checkpoint(
                    args.host,
                    remote_path,
                    size,
                    args.local_dir,
                )
                stop_viewer(viewer)
                viewer = start_viewer(
                    checkpoint,
                    args.port,
                    args.device,
                    args.state_dir / "viewer.log",
                )
                shown_step = step
                write_json(
                    status_path,
                    {
                        "latest_remote_step": step,
                        "shown_step": step,
                        "checkpoint": str(checkpoint),
                        "sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                        "viewer_pid": viewer.pid,
                        "url": f"http://127.0.0.1:{args.port}",
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            zipfile.BadZipFile,
        ) as error:
            write_json(
                args.state_dir / "viewer-attention.json",
                {
                    "error": str(error),
                    "shown_step": shown_step,
                    "at": datetime.now(UTC).isoformat(),
                },
            )
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
