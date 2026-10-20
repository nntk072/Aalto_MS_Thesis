"""Accelerator-aware interpreter selection for Triton allocations."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVATE_SCRIPT = REPO_ROOT / "scripts" / "compute" / "slurm" / "activate_venv.sh"


def _write_fake_venv(root: Path, name: str) -> Path:
    venv = root / name
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    (bin_dir / "activate").write_text(f"export SELECTED_TEST_VENV={name}\n", encoding="utf-8")
    return venv


def _run_activation(tmp_path: Path, partition: str, *, include_mi210: bool = True):
    repo = tmp_path / "repo"
    (repo / "scripts" / "compute" / "slurm").mkdir(parents=True)
    x86 = _write_fake_venv(tmp_path, "x86-test-venv")
    mi210 = (
        _write_fake_venv(tmp_path, "mi210-test-venv")
        if include_mi210
        else tmp_path / "missing-mi210"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text(
        '#!/bin/sh\nif [ "${1:-}" = "-m" ]; then echo x86_64; else exec /usr/bin/uname "$@"; fi\n',
        encoding="utf-8",
    )
    fake_uname.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "TRITON_REPO": str(repo),
            "X86_VENV": str(x86),
            "MI210_VENV": str(mi210),
            "SLURM_JOB_PARTITION": partition,
        }
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$SELECTED_TEST_VENV"',
            "test",
            str(ACTIVATE_SCRIPT),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_gpu_amd_partition_selects_rocm_mi210_environment(tmp_path: Path) -> None:
    result = _run_activation(tmp_path, "gpu-amd")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "mi210-test-venv"


def test_nvidia_x86_partition_selects_cuda_x86_environment(tmp_path: Path) -> None:
    result = _run_activation(tmp_path, "gpu-h200-141g-m")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "x86-test-venv"


def test_gpu_amd_missing_environment_reports_rocm_setup(tmp_path: Path) -> None:
    result = _run_activation(tmp_path, "gpu-amd", include_mi210=False)

    assert result.returncode != 0
    assert "MI210 ROCm venv not found" in result.stderr
    assert "tools/setup-venv-mi210" in result.stderr
