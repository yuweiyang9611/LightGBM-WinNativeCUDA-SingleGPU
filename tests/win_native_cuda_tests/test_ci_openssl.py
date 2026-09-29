"""Homebrew cleanup must only remove the known legacy OpenSSL link."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("target", ["legacy", "current", "regular"])
def test_homebrew_openssl_link_cleanup(tmp_path: Path, target: str) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("Requires a POSIX shell on PATH")
    prefix = tmp_path / "homebrew"
    (prefix / "bin").mkdir(parents=True)
    openssl = prefix / "bin/openssl"
    if target == "regular":
        openssl.write_text("keep this file", encoding="utf-8")
    else:
        version = "1.1" if target == "legacy" else "3"
        try:
            openssl.symlink_to(f"../opt/openssl@{version}/bin/openssl")
        except OSError:
            pytest.skip("Creating symbolic links is not permitted on this host")
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    brew = mock_bin / "brew"
    brew.write_text('#!/bin/sh\nprintf "%s/homebrew\\n" "$PWD"\n', encoding="utf-8", newline="\n")
    brew.chmod(0o755)
    subprocess.run(
        [
            shell,
            "-c",
            'PATH="$PWD/bin:$PATH"; export PATH; exec sh "$1"',
            "openssl-test",
            str(REPO_ROOT / ".ci/prepare-homebrew-openssl.sh"),
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        check=True,
    )
    if target == "legacy":
        assert not openssl.is_symlink()
    elif target == "current":
        assert openssl.is_symlink()
    else:
        assert openssl.read_text(encoding="utf-8") == "keep this file"
