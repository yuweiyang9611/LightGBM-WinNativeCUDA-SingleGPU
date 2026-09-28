"""Version synchronization must update package metadata together."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("version", ["9.9.9", "9.9.9.99", "9.9.9rc1"])
def test_update_version_updates_python_and_r(tmp_path: Path, version: str) -> None:
    """Run the real release script against isolated copies of its inputs."""
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("Version synchronization requires a POSIX shell on PATH")
    for name in [".appveyor.yml", "python-package/pyproject.toml", "R-package/DESCRIPTION", "R-package/configure.ac"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / name, destination)
    (tmp_path / "VERSION.txt").write_text(version + "\n", encoding="utf-8")
    subprocess.run([shell, str(REPO_ROOT / ".ci/update-version.sh")], cwd=tmp_path, env=os.environ.copy(), check=True)
    assert f'version = "{version}"' in (tmp_path / "python-package/pyproject.toml").read_text(encoding="utf-8")
    assert f"version: {version}.{{build}}" in (tmp_path / ".appveyor.yml").read_text(encoding="utf-8")
    r_version = version.replace("rc", "-")
    assert f"Version: {r_version}\n" in (tmp_path / "R-package/DESCRIPTION").read_text(encoding="utf-8")
    assert f"AC_INIT([lightgbm], [{r_version}]," in (tmp_path / "R-package/configure.ac").read_text(encoding="utf-8")
