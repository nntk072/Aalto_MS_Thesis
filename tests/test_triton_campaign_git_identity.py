from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        check=True,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_campaign_resume_shim_scopes_recorded_sha_to_controller(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], cwd=repo)
    _run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "init",
        ],
        cwd=repo,
    )
    actual_sha = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    recorded_sha = "a" * 40

    campaign_dir = repo / "outputs/gpu_campaigns/test-campaign"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text(
        json.dumps({"git_sha": recorded_sha}), encoding="utf-8"
    )
    controller = tmp_path / "campaign_controller.py"
    controller.write_text(
        "import subprocess\n"
        "print(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CAMPAIGN_ID": "test-campaign",
            "TRITON_REPO": str(repo),
            "TMPDIR": str(tmp_path),
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{environment['PATH']}",
        }
    )

    script = Path(__file__).parents[1] / "scripts/triton/campaign_git_identity.sh"
    result = _run(
        [
            "bash",
            "-c",
            'cd "$TRITON_REPO"; source "$1"; source "$1"; git rev-parse HEAD; "$3" "$2"',
            "campaign-identity-test",
            str(script),
            str(controller),
            sys.executable,
        ],
        env=environment,
    ).splitlines()

    assert result == [
        f"Campaign resume will retain recorded Git identity {recorded_sha}; source hashes remain enforced.",
        actual_sha,
        recorded_sha,
    ]
