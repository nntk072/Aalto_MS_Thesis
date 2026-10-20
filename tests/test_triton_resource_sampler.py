from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from compute.slurm import resources as sample_campaign_resources
from compute.slurm.resources import allocation_cgroup_file, cgroup_file


def test_cgroup_file_follows_nested_v2_process_cgroup(tmp_path: Path) -> None:
    root = tmp_path / "sys" / "fs" / "cgroup"
    nested = root / "slurm" / "uid_1" / "job_2" / "step_0"
    nested.mkdir(parents=True)
    (nested / "memory.current").write_text("1234\n")
    proc = tmp_path / "proc" / "self" / "cgroup"
    proc.parent.mkdir(parents=True)
    proc.write_text("0::/slurm/uid_1/job_2/step_0\n")

    assert cgroup_file("memory.current", proc_cgroup=proc, cgroup_root=root) == (
        nested / "memory.current"
    )


def test_allocation_cgroup_file_uses_slurm_job_parent(tmp_path: Path) -> None:
    root = tmp_path / "sys" / "fs" / "cgroup"
    job = root / "slurm" / "uid_1" / "job_2"
    step = job / "step_3" / "task_0"
    step.mkdir(parents=True)
    (job / "memory.current").write_text("9000\n")
    (step / "memory.current").write_text("1000\n")
    proc = tmp_path / "proc" / "worker" / "cgroup"
    proc.parent.mkdir(parents=True)
    proc.write_text("0::/slurm/uid_1/job_2/step_3/task_0\n")

    assert cgroup_file("memory.current", proc_cgroup=proc, cgroup_root=root) == (
        step / "memory.current"
    )
    assert allocation_cgroup_file("memory.current", proc_cgroup=proc, cgroup_root=root) == (
        job / "memory.current"
    )


def test_cgroup_file_follows_v1_memory_controller(tmp_path: Path) -> None:
    root = tmp_path / "sys" / "fs" / "cgroup"
    nested = root / "memory" / "slurm" / "job_2"
    nested.mkdir(parents=True)
    (nested / "memory.usage_in_bytes").write_text("5678\n")
    proc = tmp_path / "proc" / "self" / "cgroup"
    proc.parent.mkdir(parents=True)
    proc.write_text("5:cpu,memory:/slurm/job_2\n")

    assert (
        cgroup_file("memory.usage_in_bytes", proc_cgroup=proc, cgroup_root=root)
        == nested / "memory.usage_in_bytes"
    )


def test_cgroup_file_rejects_paths_escaping_mount(tmp_path: Path) -> None:
    root = tmp_path / "sys" / "fs" / "cgroup"
    root.mkdir(parents=True)
    outside = tmp_path / "secret"
    outside.write_text("not a cgroup value")
    proc = tmp_path / "proc" / "self" / "cgroup"
    proc.parent.mkdir(parents=True)
    proc.write_text("0::/../../secret\n")

    assert cgroup_file("secret", proc_cgroup=proc, cgroup_root=root) is None


def test_process_stat_cpu_usec_parses_user_and_system_time() -> None:
    fields = ["S"] + ["0"] * 12
    fields[11] = "125"
    fields[12] = "75"
    stat = f"123 (worker process) {' '.join(fields)}"

    assert sample_campaign_resources._process_stat_cpu_usec(stat, 100) == 2_000_000


def test_gpu_sample_preserves_each_device(monkeypatch) -> None:
    monkeypatch.setattr(
        "compute.slurm.resources.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="86, 44167\n81, 43583\n"),
    )

    assert sample_campaign_resources.gpu_sample() == ([86, 81], [44167, 43583])


def test_active_roots_selects_remote_workers_for_this_node(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("compute.slurm.resources.socket.gethostname", lambda: "gpuarm2.int")
    first = tmp_path / "gpuarm1.json"
    second = tmp_path / "gpuarm2.json"
    first.write_text(json.dumps({"hostname": "gpuarm1", "pid": 101}))
    second.write_text(json.dumps({"hostname": "gpuarm2", "pid": 202}))
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "jobs": [
                    {"status": "running", "remote_pid_file": str(first)},
                    {"status": "running", "remote_pid_file": str(second)},
                    {"status": "pending"},
                ]
            }
        )
    )

    roots, active = sample_campaign_resources.active_roots(campaign)

    assert roots == [202]
    assert active is True


def test_sampler_csv_records_cpu_time_and_memory_scope(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "resources.csv"
    campaign = tmp_path / "campaign.json"
    campaign.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample_campaign_resources.py",
            "--campaign",
            str(campaign),
            "--out",
            str(output),
        ],
    )
    monkeypatch.setattr(sample_campaign_resources, "active_roots", lambda _path: ([], False))
    monkeypatch.setattr(sample_campaign_resources, "read_process_table", lambda: {})
    monkeypatch.setattr(sample_campaign_resources, "process_io_bytes", lambda _pids: (1, 2, 3, 4))
    monkeypatch.setattr(sample_campaign_resources, "gpu_sample", lambda: ([75], [1234]))
    monkeypatch.setattr("compute.slurm.resources.os.getloadavg", lambda: (1.0, 0.5, 0.2))
    monkeypatch.setattr(sample_campaign_resources, "pss_bytes", lambda _pids: 5678)
    monkeypatch.setattr(
        sample_campaign_resources,
        "cgroup_value",
        lambda name: {
            "memory.current": "100",
            "memory.max": "200",
            "memory.peak": "150",
            "memory.events": "low 0",
        }.get(name, ""),
    )
    monkeypatch.setattr(sample_campaign_resources, "process_tree_cpu_usec", lambda _pids: 9_000_000)
    monkeypatch.setattr(sample_campaign_resources, "cgroup_memory_scope", lambda: "slurm_job")

    assert sample_campaign_resources.main() == 0

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["process_tree_cpu_time_usec"] == "9000000"
    assert rows[0]["cgroup_memory_scope"] == "slurm_job"
    assert rows[0]["cgroup_memory_peak"] == "150"
