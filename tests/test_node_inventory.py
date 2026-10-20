from __future__ import annotations

import json

from scripts.compute.slurm.node_inventory import print_pending_gpu_jobs


def test_pending_gpu_jobs_reports_requested_gpu_and_reason(monkeypatch, capsys) -> None:
    payload = {
        "jobs": [
            {
                "partition": "gpu-grace-h200-141g",
                "job_id": 20157390,
                "user_name": "lintune5",
                "name": "iclr_cont_rcwf",
                "tres_per_job": "gres/gpu:gh200:1",
                "tres_req_str": ("cpu=8,mem=160G,node=1,billing=141,gres/gpu=1,gres/gpu:gh200=1"),
                "cpus_per_task": {"number": 8},
                "state_reason": "AssocGrpGRESRunMinutes",
            }
        ]
    }
    monkeypatch.setattr(
        "scripts.compute.slurm.node_inventory.run",
        lambda _command: json.dumps(payload),
    )

    print_pending_gpu_jobs({"gpu-grace-h200-141g"}, show_examples=True)

    output = capsys.readouterr().out
    assert "job=20157390" in output
    assert "user=lintune5" in output
    assert "name=iclr_cont_rcwf" in output
    assert "gpu=gres/gpu:gh200:1" in output
    assert "cpu=8 mem=160G" in output
    assert "reason=AssocGrpGRESRunMinutes" in output


def test_pending_gpu_jobs_falls_back_to_requested_tres(monkeypatch, capsys) -> None:
    payload = {
        "jobs": [
            {
                "partition": "gpu-grace-h200-141g",
                "job_id": 12345,
                "user_name": "user1",
                "name": "gpu_job",
                "tres_per_job": "",
                "tres_per_node": "",
                "tres_per_task": "",
                "tres_req_str": "cpu=4,mem=16G,node=1,gres/gpu=1,gres/gpu:gh200=1",
                "cpus_per_task": {"number": 4},
                "state_reason": "Resources",
            }
        ]
    }
    monkeypatch.setattr(
        "scripts.compute.slurm.node_inventory.run",
        lambda _command: json.dumps(payload),
    )

    print_pending_gpu_jobs({"gpu-grace-h200-141g"}, show_examples=True)

    output = capsys.readouterr().out
    assert "job=12345" in output
    assert "gres/gpu:gh200=1" in output
    assert "cpu=4 mem=16G" in output
