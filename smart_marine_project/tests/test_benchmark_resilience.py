import os
import subprocess
import sys
from pathlib import Path

import pytest


def _script_path() -> str:
    root = Path(__file__).resolve().parents[1]
    return str(root / "scripts" / "benchmark_inference.py")


def test_benchmark_video_missing_path_fails_cleanly():
    script = _script_path()

    p = subprocess.run(
        [sys.executable, script, "--video", "/this/path/does/not/exist.mp4"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert p.returncode != 0
    assert "Video file not found" in p.stdout


@pytest.mark.skipif(os.name == "nt", reason="uses /dev/null")
def test_benchmark_video_invalid_file_fails_cleanly(tmp_path: Path):
    script = _script_path()

    fake = tmp_path / "fake.mp4"
    fake.write_text("not a real video")

    p = subprocess.run(
        [sys.executable, script, "--video", str(fake)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert p.returncode != 0
    assert "Could not open video source" in p.stdout
