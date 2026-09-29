"""CI must retry invalid download bodies instead of executing them."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["html_once", "http_once", "always_html"])
def test_cmake_download_retries_invalid_responses(tmp_path: Path, mode: str) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("Requires a POSIX shell on PATH")
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    curl = mock_bin / "curl"
    curl.write_text(
        """#!/bin/sh
count=0
if test -f attempts; then count=$(cat attempts); fi
count=$((count + 1))
echo "$count" > attempts
while test "$1" != --output; do shift; done
output=$2
if test "$TEST_DOWNLOAD_MODE" = always_html || { test "$TEST_DOWNLOAD_MODE" = html_once && test "$count" = 1; }; then
    echo '<html>temporary error</html>' > "$output"
elif test "$TEST_DOWNLOAD_MODE" = http_once && test "$count" = 1; then
    exit 22
else
    printf '#!/bin/sh\\nexit 0\\n' > "$output"
fi
""",
        encoding="utf-8",
        newline="\n",
    )
    sleep = mock_bin / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")
    curl.chmod(0o755)
    sleep.chmod(0o755)
    env = dict(os.environ, TEST_DOWNLOAD_MODE=mode)
    result = subprocess.run(
        [
            shell,
            "-c",
            'PATH="$PWD/bin:$PATH"; export PATH; exec sh "$@"',
            "download-test",
            str(REPO_ROOT / ".ci/download-cmake.sh"),
            "3.30.0",
            "x86_64",
            "installer.sh",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if mode == "always_html":
        assert result.returncode != 0
        assert "Could not download a valid CMake installer" in result.stderr
        assert (tmp_path / "attempts").read_text().strip() == "3"
    else:
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "attempts").read_text().strip() == "2"
        assert (tmp_path / "installer.sh").read_text().startswith("#!/bin/sh\n")
